from abc import ABC, abstractmethod
import torch
import numpy as np
import math


class BaseForwardProcess(ABC):
    """
    A forward process should define the following attributes:
        - R_b (torch.tensor): The base rate matrix, shape (S, S)
    The following methods: 
        - get_Q_t(batch_t): Transition probabilty at from time 0 to t
            shape (S, S)
        - get_R_t(batch_t): Transition rate matrix at time t
            shape (S, S)
    """
    @property
    @abstractmethod
    def R_b(self):
        pass

    @abstractmethod
    def get_Q_t(self, batch_t):
        pass

    @abstractmethod
    def get_R_t(self, batch_t):
        pass


class GaussianTargetRateForwardProcess(BaseForwardProcess):
    """
    Guassian target rate forward process used for the CIFAR10 example in Campbell
    """
    def __init__(self, cfg, device=None):
        """
        Args:
            cfg (ConfigDict): Follows the format in Campbell's code
                where attribute associated with the forward process is under `model`
        """
        S = cfg.data.S
        self.S = S
        self.rate_sigma = cfg.model.rate_sigma
        self.Q_sigma = cfg.model.Q_sigma
        self.time_exponential = cfg.model.time_exponential
        self.time_base = cfg.model.time_base
        if device:
            self.device = device
        else:
            self.device = cfg.device

        rate = np.zeros((S,S))

        vals = np.exp(-np.arange(0, S)**2/(self.rate_sigma**2))
        for i in range(S):
            for j in range(S):
                if i < S//2:
                    if j > i and j < S-i:
                        rate[i, j] = vals[j-i-1]
                elif i > S//2:
                    if j < i and j > -i+S-1:
                        rate[i, j] = vals[i-j-1]
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0.0:
                    rate[i, j] = rate[j, i] * np.exp(- ( (j+1)**2 - (i+1)**2 + S*(i+1) - S*(j+1) ) / (2 * self.Q_sigma**2)  )

        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))
        self._rate = rate

        eigvals, eigvecs = np.linalg.eig(rate)
        inv_eigvecs = np.linalg.inv(eigvecs)

        self.base_rate = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)
        self.inv_eigvecs = torch.from_numpy(inv_eigvecs).float().to(self.device)

    def _integral_rate_scalar(self, t, # ["B"]
    ):
        return self.time_base * (self.time_exponential ** t) - \
            self.time_base
    
    def _rate_scalar(self, t, # ["B"]
    ):
        return self.time_base * math.log(self.time_exponential) * \
            (self.time_exponential ** t)


    @property
    def R_b(self):
        """
        Get the base rate matrix R_b
        In this forward process, the base rate matrix is assumed to be the
        same across dimensions

        Returns:
            R_b (torch.tensor): shape (S, S)
        """
        return self._rate

    def get_R_t(self, t):
        """
        Get rate matrix R_t at time t
        where R_t = beta(t) * R_b

        Args:
            t (torch.tensor): shape (B,)
        Returns:
            R_t (torch.tensor): shape (B, S, S)
        """
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        R_t = self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)
        return R_t
    
    def get_Q_t(self, t):
        """
        Get Q_t, the marginal distribution of the forward process at time t
        where Q_t[c, :] = \vec{prob}^{c}(t) is the probability vector
        that specifies the categorical distribution of component c at time t given z_0
        p(z^{c}_{t}|x=z_{t=0}) = Cat(z^{c}_{t}|\vec{prob}^{c}(t))
        the probabilities are normalized over possible values S for a component

        Args:
            t (torch.tensor): shape (B,)
        Returns:
            Q_t (torch.tensor): shape (B, S, S)
        """
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.eigvals.view(1, S)
        transitions = self.eigvecs.view(1, S, S) @ \
            torch.diag_embed(torch.exp(adj_eigvals)) @ \
            self.inv_eigvecs.view(1, S, S)

        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(f"[Warning] GaussianTargetRate, large negative transition values {torch.min(transitions)}")

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions 