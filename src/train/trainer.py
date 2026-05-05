import os,sys,time
import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
from .utils import BayesianSGD
from networks.FC import BayesianLinear
from tqdm import tqdm
import wandb


class Trainer(object):

    def __init__(self,model,args,lr_min=1e-6,lr_factor=3.0,lr_patience=5,clipgrad=1000):
        self.model=model
        self.device = args.device
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.lr_mu = args.lr_mu
        self.lr_sigma = args.lr_sigma
        self.sbatch=args.sbatch
        self.epochs=args.epochs

        self.arch=args.arch
        self.samples=args.samples
        self.lambda_=1.

        self.output=args.output
        self.checkpoint = args.checkpoint
        self.experiment=args.experiment
        self.num_tasks=args.num_tasks

        self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
        self.modules_names_without_cls = self.find_modules_names(with_classifier=False)
        self.static = getattr(args, 'static', False)
        self.growth_rate = getattr(args, 'growth_rate', 1)
        self.growth_saturation = getattr(args, 'growth_saturation', 0.5)
        self.growth_threshold = getattr(args, 'growth_threshold', 0.05)



    def train(self, t, xtrain, ytrain, xvalid, yvalid):

        # Clear new parameter masks so they become "old" for this task
        for name, m in self.model.named_modules():
            if hasattr(m, 'weight_mask_new') and m.weight_mask_new is not None:
                m.weight_mask_new.fill_(False)
            if hasattr(m, 'bias_mask_new') and m.bias_mask_new is not None:
                m.bias_mask_new.fill_(False)

        best_loss=np.inf

        # best_model=copy.deepcopy(self.model)
        best_model = copy.deepcopy(self.model.state_dict())
        lr_mu = self.lr_mu
        lr_sigma = self.lr_sigma
        patience = self.lr_patience


        # Loop epochs
        try:
            for e in range(self.epochs):
                global_e = t*self.epochs + e
                # Train
                clock0=time.time()
                # grow network size and update parameters at every epoch!
                self.grow(global_e) # if want static version, can just set this to 0
                params_dict = self.update_lr(global_e)
                self.optimizer = BayesianSGD(params=params_dict)

                # ── LR diagnostic (runs every epoch) ──
                if e == 0 or (e + 1) % max(1, self.epochs // 3) == 0:
                    print(f'\n  [lr-check] task={t} epoch={e} (update_lr arg={global_e})')
                    for gi, group in enumerate(self.optimizer.param_groups):
                        lr = group['lr']
                        if isinstance(lr, torch.Tensor):
                            n_unique = lr.unique().numel()
                            print(f'    group {gi}: Tensor lr shape={list(lr.shape)}, '
                                  f'unique={n_unique}/{lr.numel()}, '
                                  f'min={lr.min():.6e}, max={lr.max():.6e}')
                        else:
                            print(f'    group {gi}: scalar lr={lr}')


                self.train_epoch(t,xtrain,ytrain)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain)
                clock2=time.time()

                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),
                    train_loss,100*train_acc),end='')
                wandb.log({"task": t, "epoch": e, "train_loss": train_loss, "train_acc": train_acc})
                # Valid
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

                wandb.log({"task": t, "epoch": e, "valid_loss": valid_loss, "valid_acc": valid_acc})

                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("saved best model and quit because loss became nan")
                    break

                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=copy.deepcopy(self.model.state_dict())
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr_mu /= self.lr_factor
                        lr_sigma /= self.lr_factor
                        print(' lr_mu={:.1e} lr_sigma={:.1e}'.format(lr_mu, lr_sigma),end='')
                        if lr_mu < self.lr_min:
                            print()
                            break
                        patience=self.lr_patience

                        params_dict = self.update_lr(global_e, adaptive_lr=True, lr_mu=lr_mu, lr_sigma=lr_sigma)
                        self.optimizer=BayesianSGD(params=params_dict)

                # Log histograms
                self.log_histograms(t, e, lr)

                print()
        except KeyboardInterrupt:
            print()

        # Restore best
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(t)
        # Freeze current posterior as the prior for the next task
        self.update_prior()
    
    def update_lr(self, t, lr_mu=None, lr_sigma=None, adaptive_lr=False):
        params_dict = []
        
        # Use provided LRs or fall back to initial ones
        current_lr_mu = lr_mu if lr_mu is not None else self.lr_mu
        current_lr_sigma = lr_sigma if lr_sigma is not None else self.lr_sigma

        if t==0 or self.static:
            mu_params = []
            rho_params = []
            for name, p in self.model.named_parameters():
                if 'mu' in name:
                    mu_params.append(p)
                elif 'rho' in name:
                    rho_params.append(p)
                else:
                    mu_params.append(p)
            
            params_dict.append({'params': mu_params, 'lr': current_lr_mu})
            params_dict.append({'params': rho_params, 'lr': current_lr_sigma})
            
        else:
            # Handle hidden layers with uncertainty scaling
            for name in self.modules_names_without_cls:
                n = name.split('.')
                if len(n) == 1:
                    m = self.model._modules[n[0]]
                elif len(n) == 3:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
                elif len(n) == 4:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
                else:
                    continue

                if adaptive_lr is True:
                    params_dict.append({'params': m.weight_mu, 'lr': current_lr_mu})
                    params_dict.append({'params': m.bias_mu, 'lr': current_lr_mu})
                    params_dict.append({'params': m.weight_rho, 'lr': current_lr_sigma})
                    params_dict.append({'params': m.bias_rho, 'lr': current_lr_sigma})

                else:
                    # calculate weight uncertainty
                    w_unc = torch.log1p(torch.exp(m.weight_rho.data))
                    b_unc = torch.log1p(torch.exp(m.bias_rho.data))
                    
                    # create parameter-wise learning rates (cap at lr_mu to prevent runaway)
                    w_lr = torch.mul(w_unc.clamp(max=1.0), current_lr_mu)
                    b_lr = torch.mul(b_unc.clamp(max=1.0), current_lr_mu)

                    if hasattr(m, 'weight_mask_new') and m.weight_mask_new is not None:
                        w_lr[m.weight_mask_new] = current_lr_mu
                    if hasattr(m, 'bias_mask_new') and m.bias_mask_new is not None:
                        b_lr[m.bias_mask_new] = current_lr_mu

                    params_dict.append({'params': m.weight_mu, 'lr': w_lr})
                    params_dict.append({'params': m.bias_mu, 'lr': b_lr})
                    params_dict.append({'params': m.weight_rho, 'lr': current_lr_sigma})
                    params_dict.append({'params': m.bias_rho, 'lr': current_lr_sigma})

            # Handle classifier heads (if t > 0 and not static)
            # For simplicity, we optimize the current task classifier in task-incremental
            # or the shared head in domain-incremental.
            if hasattr(self.model, 'classifier'):
                curr_cls = None
                if isinstance(self.model.classifier, torch.nn.ModuleList):
                    # Task-incremental: only current task head
                    if t < len(self.model.classifier):
                        curr_cls = self.model.classifier[t]
                else:
                    # Domain-incremental or single head
                    curr_cls = self.model.classifier
                
                if curr_cls is not None:
                    for p_name, p in curr_cls.named_parameters():
                        if 'mu' in p_name:
                            params_dict.append({'params': p, 'lr': current_lr_mu})
                        elif 'rho' in p_name:
                            params_dict.append({'params': p, 'lr': current_lr_sigma})
                        else:
                            params_dict.append({'params': p, 'lr': current_lr_mu})

        return params_dict


    def grow(self, t, n_new=1):
        """Add n_new hidden units to each hidden layer, then fix downstream layers
        (next hidden + all classifier heads) to accept the wider input.
        Returns a fresh params_dict reflecting the new parameters."""

        if self.growth_rate == 0:
            return self.model

        # Collect hidden layers in order (everything that is a BayesianLinear but not a classifier)
        hidden_layers = []
        for name, m in self.model.named_modules():
            if isinstance(m, BayesianLinear) and not name.startswith('classifier'):
                hidden_layers.append((name, m))

        if not hidden_layers:
            return self.model

        # Calculate saturation of the most recently added generation
        saturated_params = 0
        total_params = 0

        for name, m in hidden_layers:
            if hasattr(m, 'weight_rho'):
                w_stdev = torch.log1p(torch.exp(m.weight_rho.data))
                if hasattr(m, 'weight_mask_new') and m.weight_mask_new is not None and m.weight_mask_new.any():
                    w_eval = w_stdev[m.weight_mask_new]
                else:
                    w_eval = w_stdev
                
                saturated_params += (w_eval < self.growth_threshold).sum().item()
                total_params += w_eval.numel()

            if hasattr(m, 'use_bias') and m.use_bias and hasattr(m, 'bias_rho'):
                b_stdev = torch.log1p(torch.exp(m.bias_rho.data))
                if hasattr(m, 'bias_mask_new') and m.bias_mask_new is not None and m.bias_mask_new.any():
                    b_eval = b_stdev[m.bias_mask_new]
                else:
                    b_eval = b_stdev
                
                saturated_params += (b_eval < self.growth_threshold).sum().item()
                total_params += b_eval.numel()

        saturation_fraction = saturated_params / total_params if total_params > 0 else 0.0

        if saturation_fraction <= self.growth_saturation:
            return self.model

        n_new = self.growth_rate
        for idx, (name, layer) in enumerate(hidden_layers):
            old_out = layer.out_features
            # 1. Grow the output side of this hidden layer
            layer.grow_output(n_new)
            print(f'  [grow] {name}: {old_out} -> {layer.out_features} units')

            # 2. Grow the input side of the next hidden layer (if any)
            if idx + 1 < len(hidden_layers):
                next_name, next_layer = hidden_layers[idx + 1]
                next_layer.grow_input(n_new)

        # 3. Grow the input side of every classifier head
        for cls_layer in self.model.classifier:
            cls_layer.grow_input(n_new)

        # Move model back to device
        self.model.to(self.device)

        # Refresh module name caches
        self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
        self.modules_names_without_cls = self.find_modules_names(with_classifier=False)

        wandb.log({"hidden_n": hidden_layers[0][1].out_features, "task": t})

        return self.model


    def log_histograms(self, t, e, current_lr):
        metrics = {}
        # Pull actual learning rates from the optimizer's param groups
        param_to_lr = {}
        for group in self.optimizer.param_groups:
            group_lr = group['lr']
            for p in group['params']:
                param_to_lr[p] = group_lr

        for name, m in self.model.named_modules():
            if hasattr(m, 'weight_mu') and hasattr(m, 'weight_rho'):
                mask_new = m.weight_mask_new if hasattr(m, 'weight_mask_new') else None
                mask_old = ~mask_new if mask_new is not None else None
                has_new = mask_new is not None and mask_new.any()
                
                # mu tracking
                w_mu = m.weight_mu
                lr_mu = param_to_lr.get(w_mu, 0.0)
                if has_new:
                    if mask_old.any():
                        metrics[f"dist/{name}/mu.weight_old"] = wandb.Histogram(w_mu.data[mask_old].cpu().numpy())
                        if isinstance(lr_mu, torch.Tensor):
                            metrics[f"dist/{name}/mu.lr_old"] = wandb.Histogram(lr_mu[mask_old].cpu().numpy())
                    metrics[f"dist/{name}/mu.weight_new"] = wandb.Histogram(w_mu.data[mask_new].cpu().numpy())
                    if isinstance(lr_mu, torch.Tensor):
                        metrics[f"dist/{name}/mu.lr_new"] = wandb.Histogram(lr_mu[mask_new].cpu().numpy())
                else:
                    metrics[f"dist/{name}/mu.weight"] = wandb.Histogram(w_mu.data.cpu().numpy())
                    if isinstance(lr_mu, torch.Tensor):
                        metrics[f"dist/{name}/mu.lr"] = wandb.Histogram(lr_mu.cpu().numpy())
                    else:
                        metrics[f"dist/{name}/mu.lr"] = lr_mu

                # stdev tracking
                w_rho = m.weight_rho
                w_stdev = torch.log1p(torch.exp(w_rho.data))
                lr_stdev = param_to_lr.get(w_rho, 0.0)
                
                # Scalar logs for tracking trends
                metrics[f"sigma/{name}/weight_mean"] = torch.mean(w_stdev).item()

                if has_new:
                    if mask_old.any():
                        metrics[f"dist/{name}/stdev.weight_old"] = wandb.Histogram(w_stdev[mask_old].cpu().numpy())
                        if isinstance(lr_stdev, torch.Tensor):
                            metrics[f"dist/{name}/stdev.lr_old"] = wandb.Histogram(lr_stdev[mask_old].cpu().numpy())
                    metrics[f"dist/{name}/stdev.weight_new"] = wandb.Histogram(w_stdev[mask_new].cpu().numpy())
                    if isinstance(lr_stdev, torch.Tensor):
                        metrics[f"dist/{name}/stdev.lr_new"] = wandb.Histogram(lr_stdev[mask_new].cpu().numpy())
                else:
                    metrics[f"dist/{name}/stdev.weight"] = wandb.Histogram(w_stdev.cpu().numpy())
                    if isinstance(lr_stdev, torch.Tensor):
                        metrics[f"dist/{name}/stdev.lr"] = wandb.Histogram(lr_stdev.cpu().numpy())
                    else:
                        metrics[f"dist/{name}/stdev.lr"] = lr_stdev
                
                # Scalar logs for tracking trends
                metrics[f"sigma/{name}/weight_mean"] = torch.mean(w_stdev).item()

                # Bias tracking
                if hasattr(m, 'bias_mu') and m.bias_mu is not None:
                    b_mu = m.bias_mu
                    lr_b_mu = param_to_lr.get(b_mu, 0.0)
                    
                    b_mask_new = m.bias_mask_new if hasattr(m, 'bias_mask_new') else None
                    b_mask_old = ~b_mask_new if b_mask_new is not None else None
                    b_has_new = b_mask_new is not None and b_mask_new.any()
                    
                    if b_has_new:
                        if b_mask_old.any():
                            metrics[f"dist/{name}/mu.bias_old"] = wandb.Histogram(b_mu.data[b_mask_old].cpu().numpy())
                            if isinstance(lr_b_mu, torch.Tensor):
                                metrics[f"dist/{name}/mu.bias_lr_old"] = wandb.Histogram(lr_b_mu[b_mask_old].cpu().numpy())
                        metrics[f"dist/{name}/mu.bias_new"] = wandb.Histogram(b_mu.data[b_mask_new].cpu().numpy())
                        if isinstance(lr_b_mu, torch.Tensor):
                            metrics[f"dist/{name}/mu.bias_lr_new"] = wandb.Histogram(lr_b_mu[b_mask_new].cpu().numpy())
                    else:
                        metrics[f"dist/{name}/mu.bias"] = wandb.Histogram(b_mu.data.cpu().numpy())
                        if isinstance(lr_b_mu, torch.Tensor):
                            metrics[f"dist/{name}/mu.bias_lr"] = wandb.Histogram(lr_b_mu.cpu().numpy())
                        else:
                            metrics[f"dist/{name}/mu.bias_lr"] = lr_b_mu

                    b_rho = m.bias_rho
                    b_stdev = torch.log1p(torch.exp(b_rho.data))
                    lr_b_stdev = param_to_lr.get(b_rho, 0.0)
                    
                    if b_has_new:
                        if b_mask_old.any():
                            metrics[f"dist/{name}/stdev.bias_old"] = wandb.Histogram(b_stdev[b_mask_old].cpu().numpy())
                            if isinstance(lr_b_stdev, torch.Tensor):
                                metrics[f"dist/{name}/stdev.bias_lr_old"] = wandb.Histogram(lr_b_stdev[b_mask_old].cpu().numpy())
                        
                        metrics[f"dist/{name}/stdev.bias_new"] = wandb.Histogram(b_stdev[b_mask_new].cpu().numpy())
                        if isinstance(lr_b_stdev, torch.Tensor):
                            metrics[f"dist/{name}/stdev.bias_lr_new"] = wandb.Histogram(lr_b_stdev[b_mask_new].cpu().numpy())
                    else:
                        metrics[f"dist/{name}/stdev.bias"] = wandb.Histogram(b_stdev.cpu().numpy())
                        if isinstance(lr_b_stdev, torch.Tensor):
                            metrics[f"dist/{name}/stdev.bias_lr"] = wandb.Histogram(lr_b_stdev.cpu().numpy())
                        else:
                            metrics[f"dist/{name}/stdev.bias_lr"] = lr_b_stdev

                    # Scalar log for trend tracking
                    metrics[f"sigma/{name}/bias_mean"] = b_stdev.mean().item()

        wandb.log({**metrics, "task": t, "epoch": e})


    def find_modules_names(self, with_classifier=False):
        modules_names = []
        for name, p in self.model.named_parameters():
            if with_classifier is False:
                if not name.startswith('classifier'):
                    n = name.split('.')[:-1]
                    modules_names.append('.'.join(n))
            else:
                n = name.split('.')[:-1]
                modules_names.append('.'.join(n))

        modules_names = set(modules_names)

        return modules_names

    def logs(self,t):

        lp, lvp = 0.0, 0.0
        for name in self.modules_names_without_cls:
            n = name.split('.')
            if len(n) == 1:
                m = self.model._modules[n[0]]
            elif len(n) == 3:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]

            lp += m.log_prior
            lvp += m.log_variational_posterior

        if getattr(self.model, 'cl_mode', 'task-incremental') == 'domain-incremental':
            lp += self.model.classifier[0].log_prior
            lvp += self.model.classifier[0].log_variational_posterior
        else:
            lp += self.model.classifier[t].log_prior
            lvp += self.model.classifier[t].log_variational_posterior

        return lp, lvp


    def train_epoch(self,t,x,y):

        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(self.device)

        num_batches = len(x)//self.sbatch
        j=0
        # Loop batches
        for i in tqdm(range(0,len(r),self.sbatch), desc="Training epoch"):

            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images, targets = x[b].to(self.device), y[b].to(self.device)

            # Forward
            loss=self.elbo_loss(images,targets,t,num_batches,sample=True).to(self.device)

            # Backward
            self.model.to(self.device)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.model.to(self.device)

            # Update parameters
            self.optimizer.step()
        return


    def eval(self,t,x,y,debug=False):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.as_tensor(r, device=self.device, dtype=torch.int64)

        with torch.no_grad():
            num_batches = len(x)//self.sbatch
            # Loop batches
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images, targets = x[b].to(self.device), y[b].to(self.device)

                # Forward
                outputs=self.model(images,sample=False)
                output=outputs[t]
                loss = self.elbo_loss(images, targets, t, num_batches,sample=False,debug=debug)

                _,pred=output.max(1, keepdim=True)

                total_loss += loss.detach()*len(b)
                total_acc += pred.eq(targets.view_as(pred)).sum().item() 
                total_num += len(b)           

        return total_loss/total_num, total_acc/total_num


    def set_model_(model, state_dict):
        model.model.load_state_dict(copy.deepcopy(state_dict))


    def elbo_loss(self, input, target, t, num_batches, sample,debug=False):
        if sample:
            lps, lvps, predictions = [], [], []
            for i in range(self.samples):
                predictions.append(self.model(input,sample=sample)[t])
                lp, lv = self.logs(t)
                lps.append(lp)
                lvps.append(lv)

            # hack
            w1 = 1.0#1.e-3
            w2 = 1.0#1.e-3
            w3 = 1.0#5.e-2

            outputs = torch.stack(predictions,dim=0).to(self.device)
            log_var = w1*torch.stack(lvps).mean()
            log_p = w2*torch.stack(lps).mean()
            nll = w3*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum').to(device=self.device)

            return (log_var - log_p)/num_batches + nll

        else:
            predictions = []
            for i in range(self.samples):
                pred = self.model(input,sample=False)[t]
                predictions.append(pred)


            # hack
            # w1 = 1.e-3
            # w2 = 1.e-3
            w3 = 5.e-6

            outputs = torch.stack(predictions,dim=0).to(self.device)
            nll = w3*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum').to(device=self.device)

            return nll


        # w1, w2, w3 = self.get_coefs(nll,log_var,log_p,num_batches)
        # print ("New coefficients for task {} are w1={}, w2={}, w3={}".format(t,w1,w2,w3))
        # if math.isnan(log_var) or math.isnan(log_p) or math.isnan(nll):
        #     nll = torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum')
        # # if log_var > 1e3 or log_p > 1e3 or nll>1e3:
        #     print ("BEFORE: ", (log_var/num_batches).item(), (log_p / num_batches).item(), nll.item())
        #     # while math.isnan(nll):
        #         # nll = 1e-5*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum')


    def update_prior(self):
        """
        Set each layer's prior to its current posterior (VCL prior update).
        Call this after each task finishes. The prior becomes N(mu, sigma) per weight,
        so confident weights (small sigma) get a tight prior that resists future change.
        """
        from networks.distributions import GaussianPrior
        for name, m in self.model.named_modules():
            if isinstance(m, BayesianLinear):
                mu0 = m.weight_mu.data.clone()
                sigma0 = torch.log1p(torch.exp(m.weight_rho.data))
                m.weight_prior = GaussianPrior(mu0, sigma0, self.device).to(self.device)
                if m.use_bias:
                    b_mu0 = m.bias_mu.data.clone()
                    b_sigma0 = torch.log1p(torch.exp(m.bias_rho.data))
                    m.bias_prior = GaussianPrior(b_mu0, b_sigma0, self.device).to(self.device)
        print('  [prior updated] posterior frozen as prior for next task')

    def save_model(self,t):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))



    # def get_coefs(self,nll,log_var,log_p,num_batches):
    #     def take_n(num):
    #         return torch.log10(num).item()
    #
    #     exponents = np.array([take_n(num) for num in [nll, log_p, log_var]])
    #     min_exp = exponents.min()
    #     min_exp_idx = np.argmin(exponents)
    #     if min_exp_idx == 0:
    #         w1 = (10**(3-(take_n(log_var)+min_exp)))*num_batches
    #         w2 = (10**-(3-(take_n(log_p)+min_exp)))*num_batches
    #         w3 = 10.**(3-min_exp_idx)
    #     if min_exp_idx == 1:
    #         w1 = (10**(3-(take_n(log_var)+min_exp)))*num_batches
    #         w3 = 10**(3-(take_n(nll)+min_exp))
    #         w2 = (10.**-(3-min_exp_idx))*num_batches
    #     if min_exp_idx == 2:
    #         w3 = 10**(3-(take_n(nll)+min_exp))
    #         w2 = (10**-(3-(take_n(log_p)+min_exp)))*num_batches
    #         w1 = (10.**(3-min_exp_idx))*num_batches
    #
    #     return w1, w2, w3


