import torch
import torch.nn.functional as F
import tqdm
import numpy as np

import applications.cifar10.utils as utils
from applications.cifar10.managers.base_diffusion_manager import BaseDiffusionManager


class DiscreteDiffusionManager(BaseDiffusionManager):
    """
    Define a manager to handle continuous time diffusion in discrete spaces.
    """

    def __init__(self, *args, **kwargs):
        # Initialize base class
        super().__init__(*args, **kwargs)

        if self.state == 'eval':
            self.sampler_cfg = self.cfg.sampler
            self.sampler = self.sampler_cfg.name

            # Define the default sampling temperature used for property-guided-sampling
            self._guide_temp = 1

        # True if data is categorical, False if ordinal
        self.categorical = self.cfg.data.categorical

        # Get attributes from the forward process that are important for 
        # training and inference
        if hasattr(self.forward_process, 'padding_idx'):
            self.padding_idx = self.forward_process.padding_idx
        else:
            self.padding_idx = None
        self.S = self.forward_process.S

    def set_guide_temp(self, guide_temp):
        """ 
        Set the value of the class attribute corresponding to the sampling 
        temperature for guided-sampling with the property(-guide) distribution.

        Args:
            guide_temp (int or float): To be set new sampling temperature that
                must be a positive number.

        """
        err_msg_preamble = "The input 'guide_temp' must be zero or a positive number (int or float), got "

        if not isinstance(guide_temp, (int, float)):
            err_msg = err_msg_preamble+f"type '{type(guide_temp)}' instead."
            raise TypeError(err_msg)
        
        if guide_temp<0:
            err_msg = err_msg_preamble+f"value '{guide_temp}' instead."
            raise ValueError(err_msg)

        self._guide_temp = guide_temp

  
    def diffusion_batch_loss(self, batch_data, intensive_loss=True):
        """
        Return the batch loss for diffusion as defined in 
        'A Continuous Time Framework for Discrete Denoising Models'
        (https://arxiv.org/pdf/2205.14987.pdf) by Campbell et al.
        
        Args:
            batch_data (torch.utils.data.Dataset): Torch dataset object
                holding the data of the current batch that can be passed
                to the denoising model.
            intensive_loss (bool): Should this loss be normalized per point 
                in the batch (and thus not scale with the number of points 
                in the batch, i.e. 'intensive loss') or not ('extensive loss').
                (Default: True)

        Return:
            (torch.tensor, int): Scalar loss of the batch for the passed batch data
                as first and the batch size as second return variable.
        
        """
        # batch_x: shape should be (B, D)
        batch_x = batch_data['x']
        # Flatten if input is image
        if len(batch_x.shape) == 4:
            B, C, H, W = batch_x.shape
            batch_x = batch_x.view(B, C*H*W)
        B, D = batch_x.shape
        S = self.S

        if self.padding_idx:
            # Shape (B, D), true for paddings
            # Note we just need to get the padding_mask using batch_x
            # since for any t, batch_z_t should have the same padding mask as batch_x
            # provided in the forward process, the transition and rate matrix
            # are set appropriately for padding_idx 
            padding_mask = (batch_x == self.padding_idx)
        else:
            padding_mask = None

        # Sample the times (i.i.d.) for each of point in the batch
        # Remark: Thus, the times should be different for each batch point
        batch_t = self._sample_times_over_batch(B)  # Shape (B,)

        # Determine Q_t, R_t for each of the points in the batch
        # Shape (B, S, S)
        batch_Q_t = self.forward_process.get_Q_t(batch_t)
        batch_R_t = self.forward_process.get_R_t(batch_t)

        # Sample latent states z_t~p(z_t|x, t) for each point in the batch
        # Shape (B, D)
        batch_z_t = self.forward_sample_z_t(batch_x, batch_Q_t)

        # Determine (forward-in-time) R_t(z_t, :), shape (B, D, S)
        batch_R_t_z_t_vec = self._get_R_t_z_start_t_vec(batch_z_t, batch_R_t)  

        # Sample jump state z^jump_t
        # Remark: Do not use property-guidance to adjust/update jump probabilities here, thus pass batch_y=None.
        # Shape (B, D)
        batch_z_jump_t = self._sample_z_jump_t(batch_z_t, batch_t, batch_R_t_z_t_vec, batch_y=None) 

        # Predict the 'backward-in-time rate' \hat{R}^{theta}_t(z^{jump}_t, :) from z^{jump}_t to any
        # state reachable from z^{jump}_t in a transition (e.g. single component change here)
        # Shape (B, D, S)
        # This corresond to `inner_log_sig`
        # except they zero out the diagonal entries with `x_tilde_mask`
        # whereas we adjusted them to be the negative row sum 
        # so \hat{R}^{theta}_t is a proper rate matrix
        predict_hat_R_t_out = self._predict_hat_R_t_z_start_t_vec(
            batch_z_jump_t, 
            batch_t, 
            train_or_eval='train', 
            batch_y=None,
            padding_mask=padding_mask
        )
        batch_hat_R_t_z_jump_t_vec = predict_hat_R_t_out['hat_R_vec'] 
        batch_E_theta_vec_z_start_t = predict_hat_R_t_out['E_theta_vec']
        batch_sum_pqq_t = predict_hat_R_t_out['sum_pqq']

        # Compute the first term in the ELBO 
        # Determine the (batched) backward-in-time normalizing constant \hat{Z}_t(z^jump_t) from \hat{R}^{theta}_t(z^jump_t, :)
        # Shape (B,)
        # This corresponds to `reg_term` in Campbell
        batch_hat_Z_t_z_jump_t = self._get_Z_t_z_t(batch_z_jump_t, batch_hat_R_t_z_jump_t_vec)

        # Compute the second term in the ELBO
        # This corresponds to `outer_rate_sig` and `rate_sig_norm`
        # Shape (B, D, S)
        batch_R_t_z_jump_t_vec = self._get_R_t_z_end_t_vec(batch_z_jump_t, batch_R_t)
        # This corresponds to `outer_qt0_numer_sig` and `qt0_sig_norm_numer`
        # Shape (B, D, S)
        batch_Q_t_vec = self._get_Q_t_vec(batch_x, batch_Q_t)
        # This corresponds to `outer_qt0_denom_sig` and `qt0_sig_norm_denom`
        batch_Q_t_z_jump_t = batch_Q_t[
            torch.arange(B, device=self.device).repeat_interleave(D),
            batch_x.long().flatten(),
            batch_z_jump_t.long().flatten()            
        ].view(B, D, 1) + self._eps
        # This corresponds to `x_tilde_mask`
        z_jump_t_mask = torch.ones((B, D, S), device=self.device)
        z_jump_t_mask[
            torch.arange(B)[:, None].expand(B, D).to(self.device),
            torch.arange(D)[None, :].expand(B, D).to(self.device),
            batch_z_jump_t
        ] = 0.0

        # This corresponds to `Z_sig_norm`
        # Note for padding, if the rate matrix is set appropriately, we have:
        #   rate_row_sums[:, padding_idx] = 0
        #   base_Z_tmp[padding_mask] = 0
        rate_row_sums = - batch_R_t[
            torch.arange(B, device=self.device).repeat_interleave(S),
            torch.arange(S, device=self.device).repeat(B),
            torch.arange(S, device=self.device).repeat(B)
        ].view(B, S)
        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=self.device).repeat_interleave(D),
            batch_z_jump_t.long().flatten()
        ].view(B, D)
        base_Z = torch.sum(base_Z_tmp, dim=1)
        Z_subtraction = base_Z_tmp  # (B, D)
        Z_addition = rate_row_sums  # (B, S)
        # Shape (B, D, S)
        batch_Z_t_z_jump_t = base_Z.view(B, 1, 1) - \
            Z_subtraction.view(B, D, 1) + \
            Z_addition.view(B, 1, S)

        # This corresponds to the numerator of the second term in the ELBO: `outer_sum_sig`
        # Technically, the implementation should be Q_t_norm * log(batch_hat_R_t_z_jump_t_vec)
        # But this implementation implicitly dropped the log(batch_R_t_z_jump_t_vec) term
        # inside the computation for batch_hat_R_t_z_jump_t_vec
        # since it doesn't depend on theta
        Q_t_norm = utils.expsublog(batch_Q_t_vec, batch_Q_t_z_jump_t, self._eps)
        prod_R_t_Q_t = utils.expaddlog(batch_R_t_z_jump_t_vec, Q_t_norm, self._eps)
        # For paddings, the entries in prod_R_t_Q_t and batch_sum_pqq_t will be 0
        # so they are masked out in the sum
        outer_sum_sig = torch.sum(
            z_jump_t_mask * prod_R_t_Q_t * torch.log(batch_sum_pqq_t + self._eps),
            dim=(1, 2)
        )

        # This corresponds to the denominator of the second term in the ELBO: `sig_norm`
        sig_norm = torch.sum(
            z_jump_t_mask * utils.expsublog(prod_R_t_Q_t, batch_Z_t_z_jump_t, self._eps),
            dim=(1, 2)
        )

        # Calculate the loss function \hat{Z}_t(z^jump_t) - Z_t(z_t)*log[\hat{R}^theta_t(z^jump_t, z_t)] 
        # for each point in the batch.
        # Shape (B,)
        reg_term = batch_hat_Z_t_z_jump_t
        sig_term = - outer_sum_sig / sig_norm
        batch_loss_t = reg_term + sig_term

        # Get the time-dependent weight for the loss function and apply it
        # TODO: In Campbell, noise schedule is determined by definition of
        # forward process, and they don't seem to use a time weighted loss
        
        # batch_time_weight_t   = self.noise_scheduler.time_weight(batch_t)  # Shape (B,)
        batch_time_weight_t = 1
        # Determine the sum of the loss over the batch
        loss_t = torch.sum(batch_time_weight_t * batch_loss_t)  # Shape (,)

        # Direct model supervision using model predicted energy
        # Shape (B, D)
        nll = F.cross_entropy(
            batch_E_theta_vec_z_start_t.permute(0, 2, 1), 
            batch_x.long(),
            reduction='none'
        )

        # Set padded positions to 0 in the loss
        if padding_mask is not None:
            nll *= (~padding_mask)
            nll = nll.sum() / (~padding_mask).sum() * B
        else:
            nll = nll.sum() / D

        loss_t += self.nll_weight * nll

        if self._debug:
            print(f'reg: {reg_term.mean():.4f}')
            print(f'sig: {sig_term.mean():.4f}')
            print(f'outer_sum_sig: {outer_sum_sig.mean():.4f}')
            print(f'1 / sig_norm: {1 / sig_norm.mean():.4f}')
            print(f'nll: {nll.mean():.4f}')

        # If the loss should be 'intensive', divide the 'extensive' loss
        # determined above by the number of points in the batch
        if intensive_loss:
            loss_t = loss_t / B

        return dict(
            loss=loss_t,
            batch_size=B,
            nll=(nll / B).detach(),
            reg=reg_term.mean().detach(),
            sig=sig_term.mean().detach() 
        )
    
    def _get_jump_prob_vec(self, batch_z_start_t, batch_R_t_z_start_t_vec):
        """
        Return the jump probabilities from a state z^{start}_t to any other transition
        state reachable from z^{start}_t at time t, z^{transit}_t,
        p^{jump}_t[z^{start}_t](z^{transit}_t) =  [1-delta_{z^{start}_t, z^{transit}_t}]
                                                * R_t(z^{start}_t, z^{transit}_t)/Z_t(z^{start}_t)
        with
        Z_t(z_t) = \sum_{z!=z^{start}_t} R_t(z^{start}_t, z) 
                 = \sum_{z!=z^{start}_t} [1-delta_{z^{start}_t, z}]R_t(z^{start}_t, z)
        as defined in 'A Continuous Time Framework for Discrete Denoising Models'
        (https://arxiv.org/pdf/2205.14987.pdf) by Campbell et al.

        Remarks: 
        (i)
        This method works both for forward-in-time or backward-in-time rates.

        (ii)
        Due to [1-delta_{z^{start}_t, z^{transit}_t}], the identity-transition states have
        zero probability and thus only the jump-transition states have finite probabilities
        (i.e. the state has to jump and can not stay the same).

        (ii)
        The probabilities are over the encoded x-space, i.e. they specify what 
        the probability is to change to a new specific entry in encoded(z_t).
        (see illustration below)

        Args:
            batch_z_start_t (torch.tensor): The batched start state to jump 
                shape (B, D).
            batch_R_t_z_start_t_vec (torch.tensor): The transition rates
                R_t(z^{start}_t, z) at time t from z^start_t to any state z
                represented as vector R_t(z^{start}_t, :) for each dimension,
                each point in the batch, shape (B, D, S).
        
        Return:
            (torch.tensor): The jump probabilities from state z^{start}_t at time t
                shape (B, D*S), normalized over second dimension

        Implementation details:
        (i)
        delta_{z_t, :} corresponds (in tensorial form) to encoded(z_t) because encoded(z_t) will be of shape
        (batch_size, self.x_encoding_space_dim) with a 1 in each row at the position corresponding to z_t.

        (ii)
        We can rewrite
        p^{jump}_t[z^{start}_t](z^{transit}_t) = w^{jump}_t[z^{start}_t](z^{transit}_t)/Z_t(z^{start}_t)
                                               = w^{jump}_t[z^{start}_t](z^{transit}_t)/( \sum_{z} w^{jump}_t[z^{start}_t](z) )
        with weights
        w^{jump}_t[z^{start}_t](z^{transit}_t) = [1-delta_{z^{start}_t, z^{transit}_t}] R_t(z^{start}_t, z^{transit}_t)
        and where we have used
        Z_t(z^{start}_t) = \sum_{z}[1-delta_{z^{start}_t, z}]R_t(z^{start}_t, z) = \sum_{z}w^{jump}_t[z^{start}_t](z).
        
        (iii)
        Add a tiny value to R_t(z^{start}_t, z) when calculating the weights for numerical stabiliy 
        (i.e. to avoid division by zero when dividing by the sum over the weights).

        Illustration:
            Consider a 2D discrete state space with cardinalities [2, 3].
            Assume z_t=[0, 1] s.t. encoded(z_t)=(1, 0, 0, 1, 0)^T.
            Because a jump-transition must occur by construction and only happens 
            in a single component of z_t, the probabilities to change the entries that 
            are already 1, are zero,
            i.e. prob_vec^jump[z_t] = [0, p1, p2, 0, p4] while p2+p3+p5=1.
            Imagine we sample w.r.t. this probability vector and obtain a 'jump index' equal to '2' 
            (i.e. that was sampled with probability p2) corresponding to (0, 0, 1, 0, 0) [one-hot encoded].
            This means that the state encoded(z_t)=(1, 0, 0, 1, 0)^T will jump to the new
            state encoded(z^{jump}_t)=(1, 0, 1, 0, 0)^T.
        
        """
        B, D = batch_z_start_t.shape
        batch_indices = torch.arange(B, device=self.device)[:, None].expand(B, D)
        dim_indices = torch.arange(D, device=self.device)[None, :].expand(B, D)

        # Probability of jumping to another state
        # This is R_t(z^{start}_t, z) but set all entries with z = z^{start}_t to 0
        # Originally referred to as weight_vec_jump_t_z_start_t
        # Shape (B, D, S)
        batch_R_t_z_start_t_vec[
            batch_indices,
            dim_indices,
            batch_z_start_t  # find index where z = z^{start}_t
        ] = 0.0
        # Shape (B, D*S)
        batch_R_t_z_start_t_vec = batch_R_t_z_start_t_vec.view(B, -1)

        # Sum over all **dimensions and states** to renormalize, shape (B, 1)
        Z_t_z_start_t = torch.sum(batch_R_t_z_start_t_vec, dim=1).unsqueeze(-1)
        # Probability is now normalized over all states to jump to, shape (B, D*S)
        prob_vec_jump_t_z_start_t = batch_R_t_z_start_t_vec / Z_t_z_start_t

        return prob_vec_jump_t_z_start_t
    
    def _transition_to_new_state(self, batch_z_start, batch_transit_index):
        """
        Perform a transition from initial state z^start to a final state 
        z^end specified by the transition index.
        
        Args:
            batch_z_start (torch.tensor): (Batched) initial state of the transition
                shape (B, D).
            batch_transition_index (torch.tensor): (Batched) transition indices
                shape (B,) with values in [0, D*S) 
        
        Return:
            (torch.tensor): (Batched) final state z^end of the transition
                shape (B, D).
        """
        B = batch_z_start.shape[0]
        S = self.S

        batch_z_transition = batch_z_start.clone()
        # For each sample in a batch, the dimension where the transition occurs
        # Shape (B,)
        transition_dims = batch_transit_index // S
        # For each sample in a batch, the new state that we transition into
        # Shape (B,)
        transition_states = batch_transit_index % S
    
        # The new state after transition, shape (B, D)
        batch_z_transition[
            torch.arange(B, device=self.device),
            transition_dims  # Index the transitioned dimension
        ] = transition_states  # Assign the new state

        return batch_z_transition
    
    def _get_Z_t_z_t(self, batch_z_t, batch_R_t_z_t_vec):
        """
        Return the total rate to transition out from a (start) state z_t (i.e. through 
        any jump-transition) to any other state z!=z_t at time t, that is defined as
        Z_t(z_t) = \sum_{z!=z_t} R_t(z_t, z)
        and is (following the definition of R_t) equivalent to 
        Z_t(z_t) = -R_t(z_t, z_t).

        Remarks: 
        (i)
        This method works both for forward-in-time or backward-in-time rates!

        (ii)
        The time t is not explicitly used within this method but implicitly contained within
        the inputs z_t and R_t(z_t, :).

        Args:
            batch_z_t (torch.tensor): The batched state from which to jump 
                shape (B, D).
            batch_R_t_z_t_vec (torch.tensor): The transition rates R_t(z_t, z) 
                at time t from z_t to any state z represented as vector 
                R_t(z_t, :) for each dimension, each sample in the batch, shape (B, D, S).

        Return:
            (torch.tensor): Shape (B,), the total jump rate to transition out of state 'z_t' for 
                each poin in batch

        """
        B, D = batch_z_t.shape
        # R_t(z_t, z_t) = R_t(z_t, :)@encoded(z_t)
        # Get the rows of R_t(z_t, :) corresponding to self transition: R_t(z_t, z_t)
        # Shape (B, D)
        batch_indices = torch.arange(B, device=self.device)[:, None].expand(B, D)
        dim_indices = torch.arange(D, device=self.device)[None, :].expand(B, D)
        batch_R_t_z_t_z_t = batch_R_t_z_t_vec[
            batch_indices, 
            dim_indices, 
            batch_z_t
        ]
        # Z_t(z_t) = -R_t(z_t, z_t), shape (B, D)
        batch_Z_t_z_t = -batch_R_t_z_t_z_t 

        # Sum over the dimensions
        return batch_Z_t_z_t.sum(dim=1)
    
    def _get_R_t_z_start_t_vec(self, batch_z_start_t, batch_R_t):
        """
        Return the forward-in-time rate vector R_t(z^{start}_t, :) that contains the elements 
        [R_t(z^{start}_t, :)]_{z^{end}_t}=R_t(z^{start}_t, z^{end}_t) 
        corresponding to the rates to transition from z^{start}_t to state z^{end}_t 
        at time t.

        Remark: z^{start}_t and z^{end}_t are both states at the same time 't' that are separated
                by a 'forward-in-time transition' from z^{start}_t to z^{end}_t.
                The equivalent 'backward-in-time transition' would be from z^{end}_t to z^{start}_t.
        
        Args:
            batch_z_start_t (torch.tensor): Start state to jump from, shape (B, D)
            batch_R_t (torch.tensor): Forward transition rate matrix, shape (B, S, S)
        
        Return:
            (torch.tensor): The batched forward-in-time transition rate vector R_t(z_t, :)
                shape (B, D, S)
        """
        B, D = batch_z_start_t.shape
        S = self.S
        
        # R_t(z_t, :) = encoded(z_t)@R_t
        # Shape (B*D, S)
        batch_R_t_z_start_t_vec = batch_R_t[
            torch.arange(B, device=self.device).repeat_interleave(D),  # Index the batch dimension
            batch_z_start_t.flatten(),  # Index z_{t=t}, the state at t = t (state to transition from)
            :  # Index z_{t=t~}, the state at t = t~ (state to transition to), get probability to all possible states
        ]
        return batch_R_t_z_start_t_vec.view(B, D, S)

    def _get_Q_t_vec(self, batch_x, batch_Q_t):
        """
        Return the forward-in-time rate vector Q_t(x_0, :) that contains the elements 
        [Q_t(x_0, :)]_{z^{end}_t}=Q_t(x_0, z^{end}_t) 
        corresponding to the probability of transition from x_0 to state z^{end}_t 
        at time t.

        
        Args:
            batch_x (torch.tensor): Data points, shape (B, D)
            batch_R_t (torch.tensor): Forward transition rate matrix, shape (B, S, S)
        
        Return:
            (torch.tensor): The batched forward-in-time transition probability vector 
                Q_t(x_0, :), shape (B, D, S)
        """
        B, D = batch_x.shape
        S = self.S

        batch_Q_t_vec = batch_Q_t[
            torch.arange(B, device=self.device).repeat_interleave(D),  # Index the batch dimension
            batch_x.flatten().long(),  # Index x_{t=0}, the state at t = 0, which is the data
            :  # Index z_{t=t}, the state at t = t, get probability to all possible states
        ].view(B, D, S)
        return batch_Q_t_vec

    def _get_R_t_z_end_t_vec(self, batch_z_end_t, batch_R_t):
        """
        Return the forward-in-time rate vector R_t(:, z^{end}_t) that contains the elements 
        [R_t(:, z^{end}_t)]_{z^{start}_t}=R_t(z^{start}_t, z^{end}_t) 
        corresponding to the rates to transition from z^{start}_t to state z^{end}_t 
        at time t.
        
        Args:
            batch_z_end_t (torch.tensor): Start state to jump from, shape (B, D)
            batch_R_t (torch.tensor): Forward transition rate matrix, shape (B, S, S)
        
        Return:
            (torch.tensor): The batched forward-in-time transition rate vector R_t(:, z_t)
                shape (B, D, S)
        """
        B, D = batch_z_end_t.shape
        S = self.S
        
        # R_t(z_t, :) = encoded(z_t)@R_t
        # Shape (B*D, S)
        batch_R_t_z_end_t_vec = batch_R_t[
            torch.arange(B, device=self.device).repeat_interleave(D),  # Index the batch dimension
            :,  # Index z_{t=t}, the state at t = t (state to transition from), get probability to all possible states
            batch_z_end_t.flatten()  # Index z_{t=t~}, the state at t = t~ (state to transition to)
        ]
        return batch_R_t_z_end_t_vec.view(B, D, S)

    def _get_Q_t_z_end_t_vec(self, batch_z_end_t, batch_Q_t):
        """
        Return the forward-in-time transition probability vector Q_t(:, z^{end}_t) that contains the elements 
        [Q_t(:, z^{end}_t)]_{z^{start}_t}=Q_t(z^{start}_t, z^{end}_t) 
        corresponding to the rates to transition from z^{start}_t to state z^{end}_t 
        at time t.

        Remark: z^{start}_t and z^{end}_t are both states at the same time 't' that are separated
                by a 'forward-in-time transition' from z^{start}_t to z^{end}_t.
                The equivalent 'backward-in-time transition' would be from z^{end}_t to z^{start}_t.
        
        Args:
            batch_z_end_t (torch.tensor): End state to jump to, shape (B, D)
            batch_Q_t (torch.tensor): Forward transition probability matrix, shape (B, S, S)
        
        Return:
            (torch.tensor): The batched forward-in-time transition rate vector Q_t(:, z_t)
                shape (B, D, S)
        """
        B, D = batch_z_end_t.shape
        S = self.S

        # Determine Q_t(batch_z_start_t, :)=[Q_t]_{:, batch_z_start_t}=Q_t@encoded(z_start_t) 
        # and R_t(batch_z_start_t, :)=[R_t]_{:, batch_z_start_t}=R_t@encoded(z_start_t)
        # for each point in the batch.
        # Shape (B, D, S), S is for x_0
        # Correspond to qt0_denom_reg in Campbell
        batch_Q_t_z_end_t_vec = batch_Q_t[
            torch.arange(B, device=self.device).repeat_interleave(D),  # Index the batch and dimension
            :,  # Get [Q_t]_{z, z^{start}_t} for all z
            batch_z_end_t.flatten()  # Fix the state at time t to z_start_t
        ].view(B, D, S)
        return batch_Q_t_z_end_t_vec

    def _predict_hat_R_t_z_start_t_vec(
            self, batch_z_start_t, batch_t, train_or_eval,
            batch_y=None, adjustment='row_sum', padding_mask=None):
        """
        Predict the backward-in-time rate vector \hat{R}_t(z^{start}_t, z^{end}_t) that contains 
        the rates to transition from z^{start}_t to state z^{end}_t at time t, where
        the entries of \hat{R}_t(z^{start}_t, z^{end}_t) correspond to each of the z^{end}_t,
        for one fixed starting state z^{start}_t.

        Remarks: 
        z^{start}_t and z^{end}_t are both states at the same time 't' that are separated        
        by a 'backward-in-time transition' from z^{start}_t to z^{end}_t.
        The equivalent 'forward-in-time transition' would be from z^{end}_t to z^{start}_t.
        
        Args:
            batch_z_start_t (torch.tensor): Shape (B, D)
                The batched start state to jump from
            batch_t (torch.tensor): Shape (B,)
                The batched time at which the jump occurs
                Note: batch_t is technically the time where z^{end}_t is at 
                (if we consider the rate matrix as operating on a small delta_t)
            train_or_eval (str): The 'train' or 'eval' mode to be used for the denoising model
                when predicting the backward-in-time rate vector.
            batch_y (None or torch.tensor): If not None, the property y as 2D torch tensor 
                of shape (batch_size, #y-features).
                (Default: None)
            adjustment (str): Indicates which adjustment to make to the reverse rate vector
                If `=row_sum`, adjust to \hat{R}^{theta}_t(z^{start}_t, z^{start}_t) = -\sum_{z!=z^{start}_t} \hat{R}^{theta}_t(z^{start}_t, z})
                If `=zero`, adjust to \hat{R}^{theta}_t(z^{start}_t, z^{start}_t) = 0
        
        Return: Dictionary containing:
            hat_R_vec (torch.tensor): Shape (B, D, S) 
                The batched backward-in-time transition rate vector \hat{R}_t(z^{start}_t, :)
            E_theta_vec (torch.tensor): Shape (B, D, S)
                The model predicted logits of the data (normalized over S)
                used for direct model supervision
            sum_pqq (torch.tensor): Shape (B, D, S)

        Implementation details:
            Determine 'backward-in-time rate' (see implementation details in docstring for )
            \hat{R}^{theta}_t(z^{start}_t, z^{end}_t) = R_t(z^{end}_t, z^{start}_t)
                                                        \times \sum_{z} [p^{\theta}_{0|t}(z|z^{start}_t)/q_{t|0}(z^{start}_t|z)]*q_{t|0}(z^{end}_t|z),
            where z runs over all the transition states reachable from z^{start}_t - as defined in 'A Continuous Time Framework for Discrete Denoising 
            Models' (https://arxiv.org/pdf/2205.14987.pdf) by Campbell et al. - in four steps:

            Step 1: Determine
                    fraction_pq(z) = p^{\theta}_{0|t}(z|z^{start}_t)/q_{t|0}(z^{start}_t|z)
                                   = p^{\theta}_{0|t}(z|z^{start}_t)/[Q_t]_{z, z^{start}_t}
                    for each point in the batch using element-wise division (over batch points and transition states z).

            Step 2: Determine
                    sum_pqq(z^{end}_t)
                    = \sum_{z} [p^{\theta}_{0|t}(z|z^{start}_t)/q_{t|0}(z^{start}_t|z)]*q_{t|0}(z^{end}_t|z)
                    = \sum_{z} fraction_pq(z)*q_{t|0}(z^{end}_t|z)
                    = \sum_{z} fraction_pq(z)*Q_t[z, z_end_t]
                    = fraction_pq(:)@Q_t[:, z_end_t]
                    = fraction_pq(:)@Q_t(z_end_t, :)
                    vectorized for all z_end_t and for each point in the batch.

            Step 3: Determine \hat{R}^{theta}_t(z^{start}_t, z)=R_t(z, z^{start}_t)*sum_pqq(z)
                    for each point in the batch using element-wise multiplication (over batch points and transition states z).

            Step 4: The obtained \hat{R}^{theta}_t(z^{start}_t, z^{end}_t) is only correct for states z^{end}_t!=z^{start}_t.
                    However, for each component c, the 'backward-in-time rate' must fulfill
                    \hat{R}^{theta}_t(z^{start,c}_t, z^{start,c}_t) = -\sum_{z^{c}!=z^{start,c}_t} \hat{R}^{theta}_t(z^{start,c}_t, z^{c})
                    where z^{c} is the c-th component of all states reachable reachable from z^{start}_t in transitions within component c. 
                    To enforece this, we adjusted \hat{R}_t(z_start_t, :) and then return this adjusted 'backward-in-time rate'.

        """
        # Determine p_theta(x_0|z_t) for all transition states z.
        # Shape (B, D, S), S is for x_0
        batch_E_theta_vec_z_start_t = self._get_E_theta_vec_z_t(
            batch_z_start_t, batch_t, train_or_eval, batch_y=batch_y
        ) 
        # Normalizes over S to get probabilities
        # Shape (B, D, S)
        batch_p_theta_vec_z_start_t = F.softmax(batch_E_theta_vec_z_start_t, 2)
        # # Set predicted probabilities to zero for the padded positions

        # Determine Q_t and R_t for each point in the batch
        # Shape (B, S, S)
        batch_Q_t = self.forward_process.get_Q_t(batch_t) 
        batch_R_t = self.forward_process.get_R_t(batch_t)
        # Correspond to rate_vals_reg in Campbell
        batch_R_t_vec_z_start_t = self._get_R_t_z_end_t_vec(batch_z_start_t, batch_R_t)  
        # Determine Q_t(batch_z_start_t, :)=[Q_t]_{:, batch_z_start_t}=Q_t@encoded(z_start_t) 
        # and R_t(batch_z_start_t, :)=[R_t]_{:, batch_z_start_t}=R_t@encoded(z_start_t)
        # for each point in the batch.
        # Shape (B, D, S), S is for x_0
        # We fixed the end state of the transition to `batch_z_start_t`
        # which is where the backward-in-time transition starts from
        # Correspond to qt0_denom_reg in Campbell
        batch_Q_t_vec_z_start_t = self._get_Q_t_z_end_t_vec(batch_z_start_t, batch_Q_t)

        # Determine 'backward-in-time rate' (see implementation details in docstring for )
        # Step 1: Determine fraction_pq(z)=p^{\theta}_{0|t}(z|z^{start}_t)/[Q_t]_{z, z^{start}_t} 
        #         for each z and each point in the batch.
        #         Remark: 'a/b=expsublog(a,b)'
        # Shape (B, D, S), S is for x_0
        # Correspond to (p0t_reg / qt0_denom_reg) in Campbell

        # Note: how padding is addressed:
        #   Set the values back to zero in batch_fraction_pq_vec before 
        #   the matrix multiplication
        #   As a result, batch_sum_pqq_t is unaffected by paddings
        #   batch_hat_R_t_z_start_t_vec should have padded states set to 0
        #   which means the row sum of batch_hat_R_t_z_start_t_vec is not affected
        batch_fraction_pq_vec = utils.expsublog(
            batch_p_theta_vec_z_start_t, 
            batch_Q_t_vec_z_start_t,
            eps=self._eps
        )
        # if padding_mask is not None:
        #     # padding_mask_expand: shape (B, D, S)
        #     padding_mask_expand = padding_mask.unsqueeze(-1).expand_as(batch_fraction_pq_vec)
        #     # Set padded position to 0
        #     batch_fraction_pq_vec *= (~padding_mask_expand)

        # Step 2: Determine sum_pqq(z^{end}_t)=fraction_pq(:)@[Q_t]_{:, z^{end}_t}
        #         vectorized for each z^{end}_t and each point in the batch.
        # Shape (B, D, S) @ (B, S, S) --> (B, D, S) where we sum over the S for x_0
        # Correspond to (p0t_reg / qt0_denom_reg) @ qt0_numer_reg in Campbell
        batch_sum_pqq_t = batch_fraction_pq_vec @ batch_Q_t

        # Step 3: Determine \hat{R}^{theta}_t(z^{start}_t, z)=R_t(z, z^{start}_t)*sum_pqq(z)
        #         for each point in the batch using element-wise multiplication (over batch points and transition states z).
        #         Remark: 'a*b=expaddlog(a,b)'
        # Shape (B, D, S)
        # Correspond to (p0t_reg / qt0_denom_reg) * (rate_vals_reg @ qt0_numer_reg.transpose(1,2)) in Campbell
        batch_hat_R_t_z_start_t_vec = utils.expaddlog(batch_R_t_vec_z_start_t, batch_sum_pqq_t, eps=self._eps)

        # Step 4: Adjust \hat{R}^{theta}_t(z^{start}_t, z)
        # This corresponds to mask_reg in Campbell
        # Shape (B, D, S)
        batch_hat_R_t_z_start_t_vec = self._adjust_hat_R_t_z_start_t_vec(
            batch_z_start_t, batch_hat_R_t_z_start_t_vec, adjustment=adjustment)

        return dict(
            hat_R_vec=batch_hat_R_t_z_start_t_vec, 
            E_theta_vec=batch_E_theta_vec_z_start_t,
            sum_pqq=batch_sum_pqq_t
        )
    
    def _adjust_hat_R_t_z_start_t_vec(
            self, batch_z_start_t, batch_hat_R_t_z_start_t_vec, adjustment='row_sum'):
        """
        Adjust the entries in \hat{R}^{theta}_t(z^{start}_t, z^{start}_t)

        Args:
            batch_z_start_t (torch.tensor): Shape (B, D)
                The batched start state to jump from.
            batch_hat_R_t_z_start_t_vec (torch.tensor): Shape (B, D, S)
                \hat{R}^{theta}_t(z^{start}_t, :).
            adjustment (str): Indicates which adjustment to make to the diagonal element
                If `=row_sum`, adjust to \hat{R}^{theta}_t(z^{start}_t, z^{start}_t) = -\sum_{z!=z^{start}_t} \hat{R}^{theta}_t(z^{start}_t, z})
                If `=zero`, adjust to \hat{R}^{theta}_t(z^{start}_t, z^{start}_t) = 0
        Return:
            (torch.tensor): Shape (B, D, S) 
                The adjusted \hat{R}^{theta}_t(z^{start}_t, :).            
        """
        B, D = batch_z_start_t.shape
        batch_indices = torch.arange(B, device=self.device)[:, None].expand(B, D)
        dim_indices = torch.arange(D, device=self.device)[None, :].expand(B, D)

        # First set \hat{R}^{theta}_t(z^{start,c}_t, z^{start,c}_t) = 0
        # Shape (B, D, S)
        batch_hat_R_t_z_start_t_vec[
            batch_indices,
            dim_indices,
            batch_z_start_t  # Get \hat{R}^{theta}_t(z^{start}_t, z^{start}_t)
        ] = 0.0

        if adjustment == 'row_sum':
            # Sum over states for each batch, dimension 
            # where z^{start,c}_t != z^{end,c}_t
            # Shape (B, D)
            batch_hat_R_t_z_start_t_sum = torch.sum(batch_hat_R_t_z_start_t_vec, dim=2)
            
            # Set \hat{R}^{theta}_t(z^{start,c}_t, z^{start,c}_t) = -\sum_{z^{c}!=z^{start,c}_t} \hat{R}^{theta}_t(z^{start,c}_t, z^{c})
            # Shape (B, D, S)
            batch_hat_R_t_z_start_t_vec[
                batch_indices,
                dim_indices,  # Index every batch/dimension
                batch_z_start_t  # Get \hat{R}^{theta}_t(z^{start}_t, z^{start}_t)
            ] = - batch_hat_R_t_z_start_t_sum
        
        elif adjustment != 'zero':
            raise ValueError(f'adjustment={adjustment} invalid')

        if self._debug:
            print(f"[adjusted] batch_hat_R_t_z_start_t_vec: {batch_hat_R_t_z_start_t_vec.shape}")

        return batch_hat_R_t_z_start_t_vec
    
    def _get_E_theta_vec_z_t(self, batch_z_t, batch_t, train_or_eval, batch_y=None):
        """
        Determine the unnormalized energies E_theta(z|z_t) for all states z that can be reached by a transition, 
        involving a change in one component (i.e. jump-transition) or no change (i.e. identity-transition 
        s.t. z=z_t) in vectorized form.

        Args:
            batch_z_t (torch.tensor): (Batched) encoded 'conditional' state z_t,
                shape (B, D).
            batch_t (torch.tensor): (Batched) time, shape (B,).
            train_or_eval (str): The 'train' or 'eval' mode to be used for the denoising model
                when determining p_theta(z|z_t).
            batch_y (torch.tensor or None): If not None; (batched) property of shape (batch_size, #y-features)
                that is used as input in case the denoising model is conditional on the property, i.e. p_theta(z|z_t, y).
                (Default: None)

        Return:
            (torch.tensor): Shape (B, D, S). Batched[p_theta(z|z_t)]
                where the first axis is batch
                the second axis is dimension
                the third axis describes the different encoded states z 
                that can be transitioned into from z_t.
                Note: Since we also use this in direct model supervision 
                we return unnormalized logits for ease of loss computation        
        """
        # Set the denoising model into 'train mode' or 'eval mode'
        # Note: we only call train() or eval() if the model is not in the 
        # desired mode. 
        # This makes the code compatible with EMA
        # where calling train() or eval() on the same model twice
        # may cause the EMA params to overwrite the original model params
        if train_or_eval=='train':
            if not self.model_dict['denoising'].training:
                self.model_dict['denoising'].train()
        elif train_or_eval=='eval':
            if self.model_dict['denoising'].training:
                self.model_dict['denoising'].eval()
        else:
            err_msg = f"The input 'train_or_eval' must be either 'train' or 'eval'."
            raise ValueError(err_msg)
        
        # Construct the model input for the current batch and predict energies for it
        if batch_y is not None:
            batch_model_input = {'x': batch_z_t, 'y': batch_y}
        else:
            batch_model_input = {'x': batch_z_t}

        # Get model prediction for x_0 based on z_t in logits space
        # Shape (B, D, S)
        batch_E_theta_vec_z_t = self.model_dict['denoising'](batch_model_input, batch_t)    

        # Address padding
        if self.padding_idx is not None:
            B, D = batch_z_t.shape
            # Shape (B, D, S), True for all states in the padded dimension
            padding_pos_mask = (batch_z_t == self.padding_idx).unsqueeze(-1).expand((B, D, self.S))
            # Shape (B, D, S), True for all dimensions where state is padding_idx
            padding_state_mask = torch.full_like(padding_pos_mask, False)
            padding_state_mask[..., self.padding_idx] = True
            # Set to -inf all states for the padded dimension
            # and all dimensions for the padding state
            batch_E_theta_vec_z_t.masked_fill_(
                torch.logical_or(padding_pos_mask, padding_state_mask), float('-inf')
            )
            # Set back to inf the padding state of the padded dimension
            # inf is invalid for softmax, so set to a large value
            batch_E_theta_vec_z_t.masked_fill_(
                torch.logical_and(padding_pos_mask, padding_state_mask), 1e8
            )

        return batch_E_theta_vec_z_t

    def forward_sample_z_t(self, batch_x, batch_Q_t):
        """
        Sample z_t in forward-in-time diffusion (i.e. noised state at time t).
        
        Args:
            batch_x (torch.tensor): Original (i.e. unnoised) x-data as 2D 
                tensor of shape (B, D).
            batch_Q_t (torch.tensor): Forward transition probability matrix
                shape (B, S, S)

        Return:
            z_t (torch.tensor): Sampled z_t for time 't' and original x-data 'x'
                shape (B, D).                        
        """
        B, D = batch_x.shape

        # Determine \vec{prob}(t)=encoded(x)*Q_t
        # Shape (B*D, S), probability of all states at time t based on the starting state at t = 0
        # all dimensions of a sample are contiguous in the first dimension
        batch_Q_t_vec = self._get_Q_t_vec(batch_x, batch_Q_t)
        batch_prob_vec_t = batch_Q_t_vec.view(B*D, -1)

        # import pdb; pdb.set_trace()

        # Sampled states at time t, shape (B, D)
        sampled_z_t =  self._sample_z_t(batch_prob_vec_t, B)
        return sampled_z_t

    def _sample_z_jump_t(self, batch_z_t, batch_t, batch_R_t_z_t_vec, batch_y=None):
        """
        Sample a state that is reachable from z_t in a jump-transition (i.e. a change in a single discrete component).

        Remark:
        This method works for both a forward-in-time or (predicted) backward-in-time rate.

        Args:
            batch_z_t (torch.tensor): (Batched) starting state z_t, shape (B, D) 
            batch_t (torch.tensor): (Batched) times at which a jump-transition should happen
                shape (B, 1)
            batch_R_t_z_t_vec (torch.tensor): (Batched) rate R_t(z_t, :),
                shape (B, D, S).
            batch_y (None or torch.tensor): None or the property y as 2D torch tensor of shape (batch_size, #y-features).
                If batch_y is None:     Use property guidance to adjust/update jump-probabilities.
                If batch_y is not None: Do not use property guidance to adjust/update jump-probabilities.
                (Default: None)

        Return:
            (torch.tensor): Sampled jump-state as 2D torch tensor of shape (batch_size, self.x_space_dim).
        
        """
        # Determine the (batched) probability vector to jump from z_t to any other state
        # that differs from z_t in only a single component at time t. 
        # Remark: The probabilities are over the encoded x-space, i.e. they specify what 
        #         the probability is to change to a new specific entry in encoded(z_t). 
        # Shape (B, D*S)
        prob_vec_jump_t_z_t = self._get_jump_prob_vec(batch_z_t, batch_R_t_z_t_vec)

        # If the property y is not None (i.e. passed), adjust/update the jump probabilities using 
        # property guidance with this sought for property y
        if batch_y is not None:
            # The property model must be defined for property guidance
            if self.model_dict['property'] is None:
                err_msg = f"Property guidance can only be used when a property model has been defined, but self.model_dict['property'] is None."
                raise KeyError(err_msg)
            
            prob_vec_jump_t_z_t = self._get_guided_prob_vec_jump(batch_z_t, batch_t, batch_y, prob_vec_jump_t_z_t)

        # Sample the transition in index (in [0, #transitions-1] where #transitions=self.x_encoding_space_dim) of a jump-transition
        # (i.e. a transition that changes z_t in only a single discrete component) - from z_t to a jump state z^jump_t - from a 
        # categorical distribution over all transition indices of jump-states reachable from z_t defined by the determined jump probabilities.
        # Note: in Campbell, they first normalize and sample a dimension D'
        # then normalize again and sample a state S' for dimension = D'
        # Here we sample an index in D*S, then find the appropriate D' and S' 
        q_jump_t_index_z_t = torch.distributions.categorical.Categorical(prob_vec_jump_t_z_t) 
        # Shape (B,), value in [0, D*S) 
        batch_jump_index_t = q_jump_t_index_z_t.sample() 
        
        # Determine the transition state (i.e. jump state here) corresponding to the sampled transition index (i.e. jump index here).
        # Shape (B, D)
        batch_z_jump_t = self._transition_to_new_state(batch_z_t, batch_jump_index_t)

        # Sanity check that each batch should only have exactly one component that changes
        # by definition of a jump-transition under a factorized forward process
        if not ((batch_z_jump_t != batch_z_t).sum(dim=1) == 1).all():
            err_msg = f"The transition from\nz_start={batch_z_t}\nto\nz_end={batch_z_jump_t}\n is not a jump-transition for all points in the batc."
            raise ValueError(err_msg)
        return batch_z_jump_t

    def _sample_z_1(self, batch_size, num_dimensions):
        """
        I.i.d. drawn z_1 for each point in a batch.

        Args:
            num_dimensions (int): E.g. length of a sequence

        Return:
            (torch.tensor): 1D torch tensor corresponding to a z_1 sample that
                will have shape (B,D).
        
        """
        # Determine the initial distribution to sample from based on the
        # name of the forward process
        initial_dist = type(self.forward_process).__name__
        S = self.S
        if initial_dist == 'GaussianTargetRateForwardProcess':
            initial_dist_std = self.forward_process.Q_sigma
            target = np.exp(
                - ((np.arange(1, S+1) - S//2)**2) / (2 * initial_dist_std**2)
            )
            target = target / np.sum(target)

            cat = torch.distributions.categorical.Categorical(
                torch.from_numpy(target)
            )
            z_1 = cat.sample((batch_size * num_dimensions,)).view(batch_size, num_dimensions)
            z_1 = z_1.to(self.device)
        else:
            raise NotImplementedError('Unrecognized initial dist ' + initial_dist)
        
        return z_1
    
    def _sample_z_t(self, prob_vec_t, batch_size):
        """
        Sample z_t based on prob_vec_t
        
        Args:
            prob_vec_t (torch.tensor): Shape (B*D, S)
                The probability vector is normalized over the second dimension
                by construction of the transition matrix Q_t
            batch_size (int): Used to reshape the sampled z_t

        Return:
            z_t (torch.tensor): Shape (B, D)
                Sampled z_t for each sample in a batch

        """
        # Sample z_t from a categorical distribution 
        # with probability vector prob_vec_t
        q_z_t = torch.distributions.categorical.Categorical(prob_vec_t)
        z_t = q_z_t.sample()  # Shape (B*D, 1)
        z_t = z_t.view(batch_size, -1)
        return z_t

    @torch.no_grad()
    def generate(
            self, num_samples, num_intermediate_samples=0, 
            batch_y=None, classifier_guidance=False, grad_approx=False, 
            seed=None, guide_temp=None
        ):

        if seed is not None:
            utils.set_random_seed(seed)

        if guide_temp is not None:
            self.set_guide_temp(guide_temp)
    
        # Choose sampler to use
        if self.sampler == 'tau_leaping':
            outputs = self.tau_leaping(
                num_samples, num_intermediate_samples, batch_y, 
                classifier_guidance, grad_approx=grad_approx
            )
        elif self.sampler == 'pc_tau_leaping':
            outputs = self.pc_tau_leaping(
                num_samples, num_intermediate_samples, batch_y, 
                classifier_guidance, grad_approx=grad_approx
            )
        else:
            raise NotImplementedError(f'sampler={self.sampler} not implemented')

        return outputs 

    @torch.no_grad()
    def tau_leaping(
            self, num_samples, num_intermediate_samples=0, 
            batch_y=None, classifier_guidance=False, 
            grad_approx=False,
        ):
        """
        Use tau-leaping to simulate the reverse process

        Args:
            num_samples (int): Number of samples to generate
            batch_y (None or torch.tensor): None or the property y as 2D torch tensor of shape (batch_size, #y-features)
                (Default: None)
            classifier_guidance (bool): If y is provided, whether to do
                1) classifier guidance (True) or 2) classifier-free guidance (False)
                In 1), the denoising model doesn't take y as input
                In 2), the denoising model takes y as input
            guide_temp (int or float): Temperature used for guided sampling that must
                be a positive number.
                (Default: 1) 
        """

        data_shape = self.cfg.data.shape
        if isinstance(data_shape, int):
            D = data_shape
        elif len(data_shape) == 3:
            C, H, W = data_shape
            D = C * H * W            
        else:
            raise ValueError('Only cfg.data.shape with length 3 (for images) or length 1 (for other data) is supported')
        S = self.S
        num_steps = self.sampler_cfg.num_steps
        min_t = self.sampler_cfg.min_t
        
        batch_z_t_last = self._sample_z_1(num_samples, D)
        # Reverse time points to step through
        ts = np.concatenate([np.linspace(1.0, min_t, num_steps), np.array([0])])
        # Time points where we save intermediate samples

        save_ts = ts[np.linspace(0, len(ts) - 2, num_intermediate_samples, dtype=int)]
        # The input y for the denoising model
        if classifier_guidance:
            batch_y_denoising_model = None
        else:
            batch_y_denoising_model = batch_y

        batch_z_t_hist, batch_x_0_hist = [], []
        prob_ratio_hist = []
        prob_ratio_approx_hist = []
        prob_ratio_approx_sec_hist = []
        # Excluding t = 0, which we account for in the end
        for idx, t_last in enumerate(tqdm.tqdm(ts[0:-1])):
            tau = ts[idx] - ts[idx+1]

            # Predict reverse rate \hat{R}_t_last(z_t_last, :)
            batch_t_last = torch.full((num_samples,), t_last).to(self.device)
            # Set the diagonals to zero so it's valid for Poisson
            predict_hat_R_t_out = self._predict_hat_R_t_z_start_t_vec(
                batch_z_t_last, batch_t_last, 
                train_or_eval='eval', 
                adjustment='zero',
                batch_y=batch_y_denoising_model
            )
            hat_R_t_last_z_t_last_vec = predict_hat_R_t_out['hat_R_vec']

            # Apply classifier guidance
            # Update the reverse rate by the log probability of the property model
            if classifier_guidance:
                guidance_output = self._get_guided_reverse_rate(
                    batch_z_t_last, batch_t_last,
                    batch_y, hat_R_t_last_z_t_last_vec,
                    grad_approx=grad_approx,
                )
                hat_R_t_last_z_t_last_vec = guidance_output['hat_R_t_z_t']

            # P_{ds} ~ Poisson(\tau * \hat{R}_t_last(z_t_last, :))
            # Shape (B, D, S)
            poisson_dist = torch.distributions.poisson.Poisson(hat_R_t_last_z_t_last_vec * tau)
            # s - x^d_t, shape (B, D, S)
            diffs = torch.arange(S, device=self.device).view(1, 1, S) - batch_z_t_last.unsqueeze(-1)
            # Number of jumps in each dimension of each sample 
            # Shape (B, D, S)
            num_jumps = poisson_dist.sample()
            adj_diffs = num_jumps * diffs
            # Update vector, shape (B, D)
            overall_jump = torch.sum(adj_diffs, dim=2)

            # Clamp to valid state values, shape (B, D)
            batch_z_t_last = torch.clamp(batch_z_t_last + overall_jump, min=0, max=S-1).long()                

            if t_last in save_ts:
                # Save intermediate z_t
                batch_z_t_hist.append(batch_z_t_last.clone().detach().cpu().numpy())
                # Save predicted x_0 from intermediate z_t
                batch_E_theta_vec_z_t_last = self._get_E_theta_vec_z_t(
                    batch_z_t_last,
                    t_last * torch.ones((num_samples,), device=self.device),
                    train_or_eval='eval', 
                    batch_y=batch_y_denoising_model
                )
                batch_p_theta_vec_z_t_last = F.softmax(batch_E_theta_vec_z_t_last, 2)
                batch_x_0_t = torch.max(batch_p_theta_vec_z_t_last, dim=2)[1]
                batch_x_0_hist.append(batch_x_0_t.clone().detach().cpu().numpy())
                if classifier_guidance and idx >= 0:
                    # Save prob ratio for intermediate z_t
                    prob_ratio_hist.append(guidance_output['prob_ratio'].detach().cpu().numpy())
                    prob_ratio_approx_hist.append(guidance_output['prob_ratio'].detach().cpu().numpy())
                    prob_ratio_approx_sec_hist.append(guidance_output['prob_ratio'].detach().cpu().numpy())

        # Shape (# intermediates, B, D)
        batch_z_t_hist = np.array(batch_z_t_hist).astype(int)
        batch_x_0_hist = np.array(batch_x_0_hist).astype(int)
        prob_ratio_hist = np.array(prob_ratio_hist)
        prob_ratio_approx_hist = np.array(prob_ratio_approx_hist)
        prob_ratio_approx_sec_hist = np.array(prob_ratio_approx_sec_hist)

        # Predict x_0 from z_{min_t}
        # Determine p_theta(x_0|z_t) for all transition states z.
        # Shape (B, D, S), S is for x_0
        batch_E_theta_vec_z_t_last = self._get_E_theta_vec_z_t(
            batch_z_t_last, 
            min_t * torch.ones((num_samples,), device=self.device),
            train_or_eval='eval', 
            batch_y=batch_y_denoising_model)
        # batch_E_theta_vec_z_t_last[:, :, self.padding_idx] = float('-inf')
        batch_p_theta_vec_z_t_last = F.softmax(batch_E_theta_vec_z_t_last, 2)
        batch_x_0 = torch.max(batch_p_theta_vec_z_t_last, dim=2)[1]

        return dict(
            x_0=batch_x_0.detach().cpu().numpy().astype(int),
            x_hist=batch_z_t_hist,
            x_0_hist=batch_x_0_hist,
            prob_ratio_hist=prob_ratio_hist,
            prob_ratio_approx_hist=prob_ratio_approx_hist,
            prob_ratio_approx_sec_hist=prob_ratio_approx_sec_hist
        )

    @torch.no_grad()
    def pc_tau_leaping(
            self, num_samples, num_intermediate_samples=0, 
            batch_y=None, classifier_guidance=False, 
            grad_approx=False,
        ):
        """
        Tau-leaping with predictor corrector sampling

        Args:
            num_samples (int): Number of samples to generate
            batch_y (None or torch.tensor): None or the property y as 2D torch tensor of shape (batch_size, #y-features)
                (Default: None)
            classifier_guidance (bool): If y is provided, whether to do
                1) classifier guidance (True) or 2) classifier-free guidance (False)
                In 1), the denoising model doesn't take y as input
                In 2), the denoising model takes y as input
            guide_temp (int or float): Temperature used for guided sampling that must
                be a positive number.
                (Default: 1) 
        """

        data_shape = self.cfg.data.shape
        if isinstance(data_shape, int):
            D = data_shape
        elif len(data_shape) == 3:
            C, H, W = data_shape
            D = C * H * W            
        else:
            raise ValueError('Only cfg.data.shape with length 3 (for images) or length 1 (for other data) is supported')
        S = self.S
        num_steps = self.sampler_cfg.num_steps
        min_t = self.sampler_cfg.min_t
        num_corrector_steps = self.sampler_cfg.num_corrector_steps
        corrector_step_size_multiplier = self.sampler_cfg.corrector_step_size_multiplier
        corrector_entry_time = self.sampler_cfg.corrector_entry_time
        B = num_samples

        def _take_poisson_steps(in_x, in_reverse_rates, in_h):
            # P_{ds} ~ Poisson(\tau * \hat{R}_t_last(z_t_last, :))
            # Shape (B, D, S)
            poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
            # s - x^d_t, shape (B, D, S)
            diffs = torch.arange(S, device=self.device).view(1, 1, S) - in_x.unsqueeze(-1)
            # Number of jumps in each dimension of each sample 
            # Shape (B, D, S)
            num_jumps = poisson_dist.sample()
            adj_diffs = num_jumps * diffs
            # Update vector, shape (B, D)
            overall_jump = torch.sum(adj_diffs, dim=2)
            # Clamp to valid state values, shape (B, D)
            new_x = torch.clamp(in_x + overall_jump, min=0, max=S-1).long()  
            return new_x

        def _get_rates(in_x, batch_t):    
            # Predict reverse rate \hat{R}_t_last(z_t_last, :)
            # Set the diagonals to zero so it's valid for Poisson
            predict_hat_R_t_out = self._predict_hat_R_t_z_start_t_vec(
                in_x, batch_t, train_or_eval='eval', adjustment='zero',
                batch_y=batch_y_denoising_model
            )
            # The reverse rates
            reverse_rates = predict_hat_R_t_out['hat_R_vec']
            # Apply classifier guidance
            # Update the reverse rate by the log probability of the property model
            if classifier_guidance:
                guidance_output = self._get_guided_reverse_rate(
                    in_x, batch_t,
                    batch_y, reverse_rates,
                    grad_approx=grad_approx,
                )
                reverse_rates = guidance_output['hat_R_t_z_t']  
            return reverse_rates

        batch_z_t_last = self._sample_z_1(B, D)
        # Reverse time points to step through
        h = 1.0 / num_steps # approximately 
        ts = np.linspace(1.0, min_t + h, num_steps)
        # Time points where we save intermediate samples
        save_ts = ts[np.linspace(0, len(ts) - 2, num_intermediate_samples, dtype=int)]
        
        # The input y for the denoising model
        if classifier_guidance:
            batch_y_denoising_model = None
        else:
            batch_y_denoising_model = batch_y

        batch_z_t_hist, batch_x_0_hist = [], []

        # Excluding t = 0, which we account for in the end
        for idx, t_last in enumerate(tqdm.tqdm(ts[0:-1])):
            h = ts[idx] - ts[idx+1]

            # Predictor step
            batch_t_last = torch.full((B,), t_last).to(self.device)
            hat_R_t_last_z_t_last_vec = _get_rates(batch_z_t_last, batch_t_last)
            batch_z_t_last = _take_poisson_steps(batch_z_t_last, hat_R_t_last_z_t_last_vec, h)

            if t_last in save_ts:
                # Save intermediate z_t
                batch_z_t_hist.append(batch_z_t_last.clone().detach().cpu().numpy())
                # Save predicted x_0 from intermediate z_t
                batch_E_theta_vec_z_t_last = self._get_E_theta_vec_z_t(
                    batch_z_t_last, batch_t_last, train_or_eval='eval', 
                    batch_y=batch_y_denoising_model
                )
                batch_p_theta_vec_z_t_last = F.softmax(batch_E_theta_vec_z_t_last, 2)
                batch_x_0_t = torch.max(batch_p_theta_vec_z_t_last, dim=2)[1]
                batch_x_0_hist.append(batch_x_0_t.clone().detach().cpu().numpy())
            
            # Corrector steps
            if t_last <= corrector_entry_time:
                for _ in range(num_corrector_steps):
                    batch_t_corrector = torch.full((B,), t_last - h).to(self.device)
                    # Reverse rates at t - tau
                    reverse_rates = _get_rates(batch_z_t_last, batch_t_corrector)
                    # Forward rates at time t - tau
                    batch_R_t = self.forward_process.get_R_t(batch_t_corrector)
                    forward_rates = self._get_R_t_z_start_t_vec(batch_z_t_last, batch_R_t)
                    corrector_rates = forward_rates + reverse_rates
                    # Need to normalize again after adding the forward rates
                    corrector_rates[
                        torch.arange(B, device=self.device).repeat_interleave(D),
                        torch.arange(D, device=self.device).repeat(B),
                        batch_z_t_last.long().flatten()
                    ] = 0.0
                    # Take correct step
                    batch_z_t_last = _take_poisson_steps(
                        batch_z_t_last, corrector_rates, corrector_step_size_multiplier * h
                    )

        # Shape (# intermediates, B, D)
        batch_z_t_hist = np.array(batch_z_t_hist).astype(int)
        batch_x_0_hist = np.array(batch_x_0_hist).astype(int)

        # Predict x_0 from z_{min_t}
        # Determine p_theta(x_0|z_t) for all transition states z.
        # Shape (B, D, S), S is for x_0
        batch_E_theta_vec_z_t_last = self._get_E_theta_vec_z_t(
            batch_z_t_last, 
            min_t * torch.ones((B,), device=self.device),
            train_or_eval='eval', 
            batch_y=batch_y_denoising_model)
        # batch_E_theta_vec_z_t_last[:, :, self.padding_idx] = float('-inf')
        batch_p_theta_vec_z_t_last = F.softmax(batch_E_theta_vec_z_t_last, 2)
        batch_x_0 = torch.max(batch_p_theta_vec_z_t_last, dim=2)[1]

        return dict(
            x_0=batch_x_0.detach().cpu().numpy().astype(int),
            x_hist=batch_z_t_hist,
            x_0_hist=batch_x_0_hist,
        )

    def _get_property_model_log_prob(
            self, batch_z_t, batch_t, batch_y, chunk_size=128, return_logits=False):
        """
        Return the log probability of the property model for property guidance.
        As property guidance is only used for generation, the property model
        should be used in evaluation-mode!
        
        Args:
            batch_z_t (torch.tensor): (Batched) state z_t as 2D torch tensor
                of shape (B, D).
            batch_t (torch.tensor): (Batched) time as 2D torch tensor
                of shape (B,).
            batch_y (torch.tensor): (Batched) property values as 2D torch tensor
                of shape (B, #y-features).
            chunk_size (int): If B > chunk_size, 
                split the batch dimension of each tensor 
                (batch_z_t, batch_t, batch_y) into size of chunk_size

        Return:
            (torch.tensor): Log probability of the property model for the inputs
                as 1D torch tensor of shape (B,).
        
        """
        self.model_dict['property'].eval()  # Ensure that the model is in evaluation mode

        B = batch_z_t.shape[0]
        if B <= chunk_size:
            property_model_out = self.model_dict['property'].log_prob(
                batch_z_t,
                batch_t,
                batch_y,
                return_logits=return_logits
            )
            if not return_logits:
                property_log_probs = property_model_out
            else:
                property_log_probs, property_logits = property_model_out
        else:
            if return_logits: raise NotImplementedError
            batch_z_t_chunks = torch.split(batch_z_t, chunk_size)
            batch_t_chunks = torch.split(batch_t, chunk_size)
            batch_y_chunks = torch.split(batch_y, chunk_size)
            property_log_probs = []
            for i in range(len(batch_z_t_chunks)):
                batch_z_t_chunk = batch_z_t_chunks[i]
                property_log_probs.append(
                    self.model_dict['property'].log_prob(
                        batch_z_t_chunk,
                        batch_t_chunks[i],
                        batch_y_chunks[i]
                    )
                )
            property_log_probs = torch.cat(property_log_probs, dim=0)
        if not return_logits:
            return property_log_probs
        else:
            return property_log_probs, property_logits

    def _get_all_jump_transitions(self, batch_z_t):
        """
        Get all D*S possible states reachable from from the current state z_t
        via a jump transition (technically D(S-1), since we already account for no change) 
        This information is stored in a shape (B, D*S, D) tensor, where the 
        second dimension indexes the D*S possible transition,
        and the third dimension is the corresponding new state values given that jump transition

        Returns:
            batch_z_jump_t_all (torch.tensor): All possible states z_jump_t that are
                reachable from z_t via a jump transition, shape (B*D*S, D)
        """
        B, D = batch_z_t.shape
        S = self.S
    
        # Hijack the first dimension (usually the batch dimension) 
        # to encode all possible transition
        # Shape (B, D*S, D) -> (B*D*S, D)
        batch_z_t_all = batch_z_t.unsqueeze(1).repeat(1, D*S, 1)
        batch_z_t_all = batch_z_t_all.view(-1, D)
        # Shape (B, D*S) -> (B*D*S,)
        transition_indices = torch.arange(D*S).to(self.device)
        transition_indices = transition_indices.repeat(B, 1).flatten()
        # Shape (B*D*S, D)
        batch_z_jump_t_all = self._transition_to_new_state(batch_z_t_all, transition_indices)
        return batch_z_jump_t_all

    def _get_guided_reverse_rate(
            self, batch_z_t, batch_t, batch_y, batch_hat_R_t_z_t,
            grad_approx=False):
        """
        Adjust the reverse rate over the space of all jump-transitions (reachable from z_t) at time
        t, using the log-probabilities of the property model prediction over all these jump-transisitions at time
        t using the guide-probability temperature T_{guide} for sampling,

        Remark:
            Step 1: Given z_t, get all the z_t_next it can transition to via a jump transition 
            Get log(p^{jump,guided}(z)) for all z_t_next

        Args:
            batch_z_t (torch.tensor): (Batched) state z_t as 2D torch tensor
                of shape (B, D).
            batch_t (torch.tensor): (Batched) time as 2D torch tensor
                of shape (B,).
            batch_y (torch.tensor): (Batched) property values as 
                either 1D torch tensor of shape (B,) (e.g. with class indices)
                or 2D tensor of shape (B, #y-features)
            batch_hat_R_t_z_t (torch.tensor): The predicted reverse rate going from
                z_t, shape (B, D, S)
        """
        B, D = batch_z_t.shape
        S = self.S
        if not grad_approx:
           # If not using approximation, then get log prob of property model 
            # for all jump transition states, requires D*S forward passes

            # Get the log p(y|x=z_t), shape (B,)
            property_log_prob_z_t = self._get_property_model_log_prob(
                batch_z_t, batch_t, batch_y
            )

            # Expand batch_t and batch_y match the shape of batch_z_jump_t_all
            # Shape (B*D*S,)
            batch_t = batch_t.repeat(1, D*S).flatten()
            # Shape (B*D*S, 1)
            if len(batch_y.shape) == 1:
                # Shape (B*D*S, 1)
                batch_y = batch_y.repeat(1, D*S).flatten()
            else:
                # Shape (B*D*S, #y-features)
                num_y_features = batch_y.shape[1]
                batch_y = batch_y.unsqueeze(1).repeat(1, D*S, 1).view(B*D*S, num_y_features)
            batch_y = batch_y.to(self.device)

            # Get all jump transitions, shape (B*D*S, D)
            batch_z_jump_t_all = self._get_all_jump_transitions(batch_z_t)

            # property_log_prob, shape (B*D*S) -> (B, D, S)
            property_log_prob = self._get_property_model_log_prob(
                batch_z_jump_t_all, batch_t, batch_y
            )
            property_log_prob = property_log_prob.view(B, D, S)

            # Get log p(y|z~) - log p(y|z_t) for all z~
            # Shape (B, D, S)
            property_log_prob_ratio = property_log_prob - property_log_prob_z_t.view(B, 1, 1)
        
        else:
            if not self.categorical:
                batch_z_t = batch_z_t.float()
                # Get \grad_{x}{log p(y|x)}(z_t), shape (B, D)
                with torch.enable_grad():
                    batch_z_t.requires_grad_(True)
                    # Get the log p(y|x=z_t), shape (B,)
                    property_log_prob_z_t = self._get_property_model_log_prob(
                        batch_z_t, batch_t, batch_y
                    )
                    property_log_prob_z_t.sum().backward()
                    # Shape (B, D)
                    grad_log_prob_z_t = batch_z_t.grad
                all_singles = batch_z_t.unsqueeze(-1).expand(B, D, S)
                all_diffs = torch.arange(0, S, device=batch_z_t.device).unsqueeze(0).unsqueeze(1).expand(B, D, S)
                # 1st order Taylor approximation of the log difference
                # Shape (B, D, S)
                property_log_prob_ratio = (all_diffs - all_singles) * grad_log_prob_z_t.unsqueeze(-1)

            else:
                # Else, data is categorical
                # One-hot encode categorical data, shape (B, D, S)
                batch_z_t = F.one_hot(batch_z_t.long(), num_classes=S).to(torch.float)
                
                # Get \grad_{x}{log p(y|x)}(z_t), shape (B, D, S)
                with torch.enable_grad():
                    batch_z_t.requires_grad_(True)
                    # Get the log p(y|x=z_t), shape (B,)
                    property_log_prob_z_t = self._get_property_model_log_prob(
                        batch_z_t, batch_t, batch_y
                    )
                    property_log_prob_z_t.sum().backward()
                    # Shape (B, D, S)
                    grad_log_prob_z_t = batch_z_t.grad
                # 1st order Taylor approximation of the log difference
                # Shape (B, D, S)
                property_log_prob_ratio = grad_log_prob_z_t - (batch_z_t * grad_log_prob_z_t).sum(dim=-1, keepdim=True)
        
        # Scale log prob ratio by temperature
        property_log_prob_ratio /= self._guide_temp
        # Sometimes after scaling by temperature the log ratio is too large
        # property_log_prob_ratio = torch.clamp(property_log_prob_ratio, max=70)

        # Exponentiate to get p(y|z~) / p(y|z_t)
        # Shape (B, D, S)
        property_prob_ratio = torch.exp(property_log_prob_ratio)

        # Multiply the reverse rate elementwise with the property density ratio
        # p(y|x=z~) / p(y|x=z_t)
        # Note those entries in batch_hat_R_t_z_t corresponding to self transition
        # should already be set to 0
        batch_hat_R_t_z_t = batch_hat_R_t_z_t * property_prob_ratio

        return dict(hat_R_t_z_t=batch_hat_R_t_z_t, prob_ratio=property_prob_ratio)