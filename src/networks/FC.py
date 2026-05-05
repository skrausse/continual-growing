import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributions import VariationalPosterior, Prior



class BayesianLinear(nn.Module):
    '''
    Applies a linear Bayesian transformation to the incoming data: :math:`y = Ax + b`
    '''

    def __init__(self, in_features, out_features, args, use_bias=True):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.device = args.device
        self.rho = args.rho
        self.args = args
        self.static = getattr(args, 'static', False)

        # Variational Posterior Distributions
        # mu: Normal(0, 0.1)
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
        
        # rho initialization
        rho_base = self._get_init_rho((out_features, in_features), args)
        self.weight_rho = nn.Parameter(rho_base + torch.empty((out_features, in_features),
                                      device=self.device, dtype=torch.float32).normal_(0.0, 0.01),requires_grad=True)
        
        self.weight = VariationalPosterior(self.weight_mu, self.weight_rho, self.device)

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((out_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
            
            b_rho_base = self._get_init_rho((out_features,), args)
            self.bias_rho = nn.Parameter(b_rho_base + torch.empty((out_features,),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.01),requires_grad=True)
            self.bias = VariationalPosterior(self.bias_mu, self.bias_rho, self.device)
        else:
            self.register_parameter('bias', None)            

        # Prior Distributions
        from .distributions import UnimodalPrior, StdevMixturePrior
        reg_mode = getattr(args, 'regularization', 'bbb')
        if reg_mode == 'unimodal':
            self.weight_prior = UnimodalPrior(args)
            if self.use_bias: self.bias_prior = UnimodalPrior(args)
        elif reg_mode == 'sns':
            self.weight_prior = StdevMixturePrior(args)
            if self.use_bias: self.bias_prior = StdevMixturePrior(args)
        else: # bbb (default)
            self.weight_prior = Prior(args)
            if self.use_bias: self.bias_prior = Prior(args)

        # Initialize log prior and log posterior
        self.log_prior = torch.tensor(0.0, device=self.device)
        self.log_variational_posterior = torch.tensor(0.0, device=self.device)

        self.mask_flag = False

        # Boolean masks to track newly added parameters
        self.register_buffer('weight_mask_new', torch.zeros((out_features, in_features), dtype=torch.bool, device=self.device))
        if self.use_bias:
            self.register_buffer('bias_mask_new', torch.zeros(out_features, dtype=torch.bool, device=self.device))
        else:
            self.register_buffer('bias_mask_new', None)


    def _get_init_rho(self, shape, args):
        import math
        if getattr(args, 'rho_init_mode', 'gaussian') == 'bimodal':
            rho1 = math.log(math.expm1(args.sigma_prior1))
            rho2 = math.log(math.expm1(args.sigma_prior2))
            mask = (torch.rand(shape, device=self.device) < args.pi).float()
            return mask * rho1 + (1. - mask) * rho2
        else:
            return torch.full(shape, self.rho, device=self.device)

    def grow_output(self, n_new=1):
        """Add n_new output units with fresh initialization (new rows in weight, new bias entries)."""
        device = self.device

        # New weight rows
        new_w_mu = torch.empty((n_new, self.in_features), device=device, dtype=torch.float32).normal_(0., 0.1)
        new_w_rho_base = self._get_init_rho((n_new, self.in_features), self.args)
        new_w_rho = new_w_rho_base + torch.empty((n_new, self.in_features), device=device, dtype=torch.float32).normal_(0., 0.01)

        self.weight_mu = nn.Parameter(torch.cat([self.weight_mu.data, new_w_mu], dim=0), requires_grad=True)
        self.weight_rho = nn.Parameter(torch.cat([self.weight_rho.data, new_w_rho], dim=0), requires_grad=True)

        self.weight_mask_new.fill_(False)
        new_weight_mask = torch.ones((n_new, self.in_features), dtype=torch.bool, device=device)
        self.weight_mask_new = torch.cat([self.weight_mask_new, new_weight_mask], dim=0)

        if self.use_bias:
            new_b_mu = torch.empty((n_new,), device=device, dtype=torch.float32).normal_(0., 0.1)
            new_b_rho_base = self._get_init_rho((n_new,), self.args)
            new_b_rho = new_b_rho_base + torch.empty((n_new,), device=device, dtype=torch.float32).normal_(0., 0.01)

            self.bias_mu = nn.Parameter(torch.cat([self.bias_mu.data, new_b_mu], dim=0), requires_grad=True)
            self.bias_rho = nn.Parameter(torch.cat([self.bias_rho.data, new_b_rho], dim=0), requires_grad=True)

            self.bias_mask_new.fill_(False)
            new_bias_mask = torch.ones(n_new, dtype=torch.bool, device=device)
            self.bias_mask_new = torch.cat([self.bias_mask_new, new_bias_mask], dim=0)

        self.out_features += n_new
        self._rebuild_posteriors()


    def grow_input(self, n_new=1):
        """Add n_new input connections with fresh initialization (new columns in weight)."""
        device = self.device

        # New weight columns
        new_w_mu = torch.empty((self.out_features, n_new), device=device, dtype=torch.float32).normal_(0., 0.1)
        new_w_rho_base = self._get_init_rho((self.out_features, n_new), self.args)
        new_w_rho = new_w_rho_base + torch.empty((self.out_features, n_new), device=device, dtype=torch.float32).normal_(0., 0.01)

        self.weight_mu = nn.Parameter(torch.cat([self.weight_mu.data, new_w_mu], dim=1), requires_grad=True)
        self.weight_rho = nn.Parameter(torch.cat([self.weight_rho.data, new_w_rho], dim=1), requires_grad=True)

        self.weight_mask_new.fill_(False)
        new_weight_mask = torch.ones((self.out_features, n_new), dtype=torch.bool, device=device)
        self.weight_mask_new = torch.cat([self.weight_mask_new, new_weight_mask], dim=1)

        # Bias is unaffected by input growth, but older features are no longer "new"
        if self.use_bias and self.bias_mask_new is not None:
            self.bias_mask_new.fill_(False)

        self.in_features += n_new
        self._rebuild_posteriors()


    def _rebuild_posteriors(self):
        """Rebuild VariationalPosterior objects after parameter resize."""
        self.weight = VariationalPosterior(self.weight_mu, self.weight_rho, self.device)
        if self.use_bias:
            self.bias = VariationalPosterior(self.bias_mu, self.bias_rho, self.device)


    def prune_module(self, mask):
        self.mask_flag = True 
        self.pruned_weight_mu=self.weight_mu.data.clone().mul_(mask).to(self.device)
        self.pruned_weight_rho=self.weight_rho.data.clone().mul_(mask).to(self.device)


    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.mask_flag:
            self.weight = VariationalPosterior(self.pruned_weight_mu, self.pruned_weight_rho, self.device)
            # if self.use_bias:
            #     self.bias = VariationalPosterior(self.pruned_bias_mu, self.pruned_bias_rho)

        if (self.training or sample) and not self.static:
            weight = self.weight.sample()
            bias = self.bias.sample() if self.use_bias else None
            
        else:
            weight = self.weight.mu
            bias = self.bias.mu if self.use_bias else None
                
        if self.training or calculate_log_probs:
            reg_mode = getattr(self.args, 'regularization', 'bbb')
            
            if reg_mode in ['bbb', 'unimodal']:
                if self.use_bias:
                    self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
                    self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
                else:
                    self.log_prior = self.weight_prior.log_prob(weight)
                    self.log_variational_posterior = self.weight.log_prob(weight)
            elif reg_mode == 'sns':
                w_sigma = torch.log1p(torch.exp(self.weight_rho))
                if self.use_bias:
                    b_sigma = torch.log1p(torch.exp(self.bias_rho))
                    self.log_prior = self.weight_prior.log_prob(w_sigma) + self.bias_prior.log_prob(b_sigma)
                else:
                    self.log_prior = self.weight_prior.log_prob(w_sigma)
                self.log_variational_posterior = torch.tensor(0.0, device=self.device)
            
        else:
            self.log_prior = torch.tensor(0.0, device=self.device)
            self.log_variational_posterior = torch.tensor(0.0, device=self.device)
        
        return F.linear(input, weight, bias)

