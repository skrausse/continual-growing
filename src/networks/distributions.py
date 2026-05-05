import math
import torch


class VariationalPosterior(torch.nn.Module):
    def __init__(self, mu, rho, device):
        super(VariationalPosterior, self).__init__()
        self.mu = mu.to(device)
        self.rho = rho.to(device)
        self.device = device
        # gaussian distribution to sample epsilon from
        self.normal = torch.distributions.Normal(0, 1)
        self.sigma = torch.log1p(torch.exp(self.rho)).to(self.device)

    def sample(self):
        sigma = torch.log1p(torch.exp(self.rho))
        epsilon = self.normal.sample(self.rho.size()).to(self.device)
        return (self.mu + sigma * epsilon).to(self.device)

    def log_prob(self, input):
        sigma = torch.log1p(torch.exp(self.rho))
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.mu) ** 2) / (2 * sigma ** 2)).sum()




class Prior(torch.nn.Module):
    '''
    Scaled Gaussian Mixtures for Priors
    '''
    def __init__(self, args):
        super(Prior, self).__init__()
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = args.pi
        self.device = args.device

        self.s1 = torch.tensor([math.exp(-1. * self.sig1)], dtype=torch.float32, device=self.device)
        self.s2 = torch.tensor([math.exp(-1. * self.sig2)], dtype=torch.float32, device=self.device)

        self.gaussian1 = torch.distributions.Normal(0,self.s1)
        self.gaussian2 = torch.distributions.Normal(0,self.s2)


    def log_prob(self, input):
        input = input.to(self.device)
        log_p1 = self.gaussian1.log_prob(input)
        log_p2 = self.gaussian2.log_prob(input)
        return torch.logaddexp(math.log(self.pi) + log_p1,
                               math.log(1. - self.pi) + log_p2).sum()


class UnimodalPrior(torch.nn.Module):
    '''
    Single Gaussian Prior (Unimodal)
    '''
    def __init__(self, args):
        super(UnimodalPrior, self).__init__()
        self.device = args.device
        self.s1 = torch.tensor([math.exp(-1. * args.sigma_prior1)], dtype=torch.float32, device=self.device)
        self.gaussian = torch.distributions.Normal(0, self.s1)
        
    def log_prob(self, input):
        return self.gaussian.log_prob(input.to(self.device)).sum()


class GaussianPrior(torch.nn.Module):
    """
    Per-weight diagonal Gaussian prior N(mu0, sigma0).
    Used for VCL-style prior update: after each task, the posterior becomes
    the prior for the next task, so confident weights are strongly regularized.
    """
    def __init__(self, mu0, sigma0, device):
        super(GaussianPrior, self).__init__()
        self.device = device
        self.register_buffer('mu0', mu0.clone().detach().to(device))
        self.register_buffer('sigma0', sigma0.clone().detach().to(device))

    def log_prob(self, input):
        input = input.to(self.device)
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma0)
                - ((input - self.mu0) ** 2) / (2 * self.sigma0 ** 2)).sum()

    def extend_output(self, n_new, in_features):
        """Grow prior when grow_output adds n_new output neurons."""
        device = self.mu0.device
        self.mu0 = torch.cat([self.mu0,
                               torch.zeros(n_new, in_features, device=device)], dim=0)
        self.sigma0 = torch.cat([self.sigma0,
                                  torch.ones(n_new, in_features, device=device)], dim=0)

    def extend_input(self, out_features, n_new):
        """Grow prior when grow_input adds n_new input connections."""
        device = self.mu0.device
        self.mu0 = torch.cat([self.mu0,
                               torch.zeros(out_features, n_new, device=device)], dim=1)
        self.sigma0 = torch.cat([self.sigma0,
                                  torch.ones(out_features, n_new, device=device)], dim=1)

    def extend_bias(self, n_new):
        """Grow bias prior when grow_output adds n_new bias entries."""
        device = self.mu0.device
        self.mu0 = torch.cat([self.mu0, torch.zeros(n_new, device=device)])
        self.sigma0 = torch.cat([self.sigma0, torch.ones(n_new, device=device)])


class StdevMixturePrior(torch.nn.Module):
    '''
    Bimodal Mixture Prior directly on Standard Deviations (Spike & Slab on stdev)
    '''
    def __init__(self, args):
        super(StdevMixturePrior, self).__init__()
        self.device = args.device
        self.pi = args.pi
        
        # Target standard deviations
        self.target_sig1 = torch.tensor([math.exp(-1. * args.sigma_prior1)], dtype=torch.float32, device=self.device)
        self.target_sig2 = torch.tensor([math.exp(-1. * args.sigma_prior2)], dtype=torch.float32, device=self.device)
        
        # We use a fixed variance to create attractors at the desired sigmas
        self.gaussian1 = torch.distributions.Normal(self.target_sig1, 0.01)
        self.gaussian2 = torch.distributions.Normal(self.target_sig2, 0.01)
        
    def log_prob(self, sigma):
        sigma = sigma.to(self.device)
        prob1 = torch.exp(self.gaussian1.log_prob(sigma))
        prob2 = torch.exp(self.gaussian2.log_prob(sigma))
        return (torch.log(self.pi * prob1 + (1. - self.pi) * prob2 + 1e-8)).sum()
