import torch
import numpy as np


class BaseDiffusionManager(object):
    """
    Define the base class for any type of 'diffusion'.
    """

    def __init__(
        self,
        cfg,
        forward_process,
        denoising_model=None,
        denoising_optimizer=None,
        property_model=None,
        property_optimizer=None,
        time_encoder=None,
        debug=False,
    ):
        self.cfg = cfg

        # Model definition
        self.forward_process = forward_process
        self.device = cfg.device
        # state: train or eval
        # used to determine which config fields are necessary
        self.state = cfg.state
        if time_encoder is not None:
            self.time_encoder = time_encoder.to(self.device)
        self.model_dict = {
            "denoising": (
                None if (denoising_model is None) else denoising_model.to(self.device)
            ),
            "property": (
                None if (property_model is None) else property_model.to(self.device)
            ),
        }
        self.optim_dict = {
            "denoising": denoising_optimizer,
            "property": property_optimizer,
        }
        self._debug = debug
        # Define a tiny number to be used for numerical stability (i.e. when dividing or taking logarithms)
        self._eps = cfg.eps_ratio

        if self.state == "train":
            # Configs for loss definition
            self.min_time = cfg.loss.min_time

            # Training configs
            self.clip_grad = cfg.training.clip_grad
            self.warmup = cfg.training.warmup
            self.lr = cfg.optimizer.lr

            # Denoising model specific traininig config
            if denoising_model is not None:
                self.nll_weight = cfg.loss.nll_weight

    def train(self):
        """Set all models into train-mode."""
        for model_name in self.model_dict:
            # Remark: If a model is not definded it is None
            if self.model_dict[model_name] is not None:
                self.model_dict[model_name].train()

    def eval(self):
        """Set all models into evaluation-mode."""
        for model_name in self.model_dict:
            # Remark: If a model is not definded it is None
            if self.model_dict[model_name] is not None:
                self.model_dict[model_name].eval()

    def encode_t(self, t):
        """Encode the input time t using the time encoder."""
        return self.time_encoder(t)

    def train_on_batch(
        self, batch_data, which_model, training_state, loader_batch_size=None
    ):
        """
        Train the model on the batch (i.e. update the model parameters) and
        return the loss of the batch data.

        Args:
            batch_data (torch.utils.data.Dataset): Torch dataset object
                holding the data of the current batch that can be passed
                to the model specified by the input 'which_model'.
            which_model (str): For which model should we calculate the
                batch loss? Expected is 'denoising' or 'property'.
            training_state (dict): Stores current state in training
                e.g. number of iterations passed
            loader_batch_size (int or None): Batch size specified for the data
                loader as integer or None.
                (Default None)

        Return:
            (float): Loss value.

        """
        # Check that the model specified by 'which_model' is defined.
        # Thus, check if 'which_model' is a dictionary-key of self.model_dict
        # and the corresponding dictionary-value (i.e. the model) is not None.
        if (which_model not in self.model_dict) or (
            self.model_dict[which_model] is None
        ):
            err_msg = f"No model with name '{which_model}' has been defined."
            raise ValueError(err_msg)

        # Set the model into training-mode
        self.model_dict[which_model].train()

        # Zero the gradients
        self.optim_dict[which_model].zero_grad()

        # Differ cases for 'which_model'
        if which_model == "denoising":
            batch_output = self.diffusion_batch_loss(batch_data, intensive_loss=True)

        elif which_model == "property":
            batch_output = self.property_batch_loss(batch_data, intensive_loss=True)
        else:
            err_msg = f"The batch loss is only defined for 'which_model' corresponding to 'denoising' or 'property', got 'which_model={which_model}' instead."
            raise ValueError(err_msg)
        batch_loss = batch_output["loss"]
        batch_size = batch_output["batch_size"]

        # Using an 'intensive' batch loss (i.e. per point in the batch) allows us to decouple
        # the batch size of the data loader (='loader_batch_size') from the learning rate,
        # which otherwise would both be coupled hyperparameters.
        # However, this leads to an overweighting of the last batch in case this batch has a
        # size smaller than the others (that have a batch size equal to 'loader_batch_size').
        # Rescaling the loss by the factor <batch_size>/loader_batch_size counteracts this
        # potential overweighting of the last batch.
        batch_loss = batch_loss * batch_size / loader_batch_size
        if batch_loss.isnan().any() or batch_loss.isinf().any():
            raise ValueError("Loss is nan")

        # Backpropagate and thus set the gradients of the model parameters
        # w.r.t. the loss
        batch_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.model_dict[which_model].parameters(), 1.0
            )

        if self.warmup > 0:
            for g in self.optim_dict[which_model].param_groups:
                g["lr"] = self.lr * np.minimum(
                    training_state["n_iter"] / self.warmup, 1.0
                )

        # Perform model parameter updates
        self.optim_dict[which_model].step()

        # Return the updated training state
        training_state["n_iter"] += 1
        for k, v in batch_output.items():
            if k == "batch_size":
                continue
            else:
                training_state[k] = v.item()

        return training_state

    @torch.no_grad()  # Don't calculate gradients for evaluation
    def eval_loss_on_batch(
        self, batch_data, which_model, loader_batch_size=None, return_logits=False
    ):
        """
        Evaluate the loss over the batch and return it WITHOUT training the model.

        Args:
            batch_data (torch.utils.data.Dataset): Torch dataset object
                holding the data of the current batch that can be passed
                to the model specified by the input 'which_model'.
            which_model (str): For which model should we calculate the
                batch loss? Expected is 'denoising' or 'property'.
            loader_batch_size (int or None): Batch size specified for the data
                loader as integer or None.
                (Default None)

        Return:
            (float): Loss value.

        """
        # Check that the model specified by 'which_model' is defined.
        # Thus, check if 'which_model' is a dictionary-key of self.model_dict
        # and the corresponding dictionary-value (i.e. the model) is not None.
        if (which_model not in self.model_dict) or (
            self.model_dict[which_model] is None
        ):
            err_msg = f"No model with name '{which_model}' has been defined."
            raise ValueError(err_msg)

        # Set the model into evaluation-mode
        self.model_dict[which_model].eval()

        # Differ cases for 'which_model'
        if which_model == "denoising":
            batch_output = self.diffusion_batch_loss(batch_data, intensive_loss=True)
        elif which_model == "property":
            batch_output = self.property_batch_loss(
                batch_data, intensive_loss=True, return_logits=return_logits
            )
        else:
            err_msg = f"The batch loss is only defined for 'which_model' corresponding to 'denoising' or 'property', got 'which_model={which_model}' instead."
            raise ValueError(err_msg)
        batch_loss = batch_output["loss"]
        batch_size = batch_output["batch_size"]

        # Using an 'intensive' batch loss (i.e. per point in the batch) allows us to decouple
        # the batch size of the data loader (='loader_batch_size') from the learning rate,
        # which otherwise would both be coupled hyperparameters.
        # However, this leads to an overweighting of the last batch in case this batch has a
        # size smaller than the others (that have a batch size equal to 'loader_batch_size').
        # Rescaling the loss by the factor <batch_size>/loader_batch_size counteracts this
        # potential overweighting of the last batch.
        batch_loss = batch_loss * batch_size / loader_batch_size

        # Return the loss value
        eval_output = dict()
        for k, v in batch_output.items():
            if k == "batch_size":
                continue
            elif k == "logits":
                eval_output[k] = v
            else:
                eval_output[k] = v.item()
        return eval_output

    def property_batch_loss(self, batch_data, intensive_loss=True, return_logits=False):
        """
        Return the batch loss for the property model.

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
        # Access the batch x-data
        batch_x = batch_data["x"]
        # Flatten if input is image
        if len(batch_x.shape) == 4:
            B, C, H, W = batch_x.shape
            batch_x = batch_x.view(B, C * H * W)

        # Access the y-data
        if "y" not in batch_data:
            err_msg = f"To train the property model, the batch data must include 'y' values, but didn't."
            raise KeyError(err_msg)

        batch_y = batch_data["y"]

        # Determine the batch size from batch_x that has shape
        # (batch_size, #x-features)
        batch_size = batch_x.shape[0]

        # Sample a time t for each point in the batch
        batch_t = self._sample_times_over_batch(batch_size)
        # Determine Q_t for each of the points in the batch
        # Shape (B, S, S)
        batch_Q_t = self.forward_process.get_Q_t(batch_t)

        # Sample latent state of each datapoint in the batch for their
        # corresponding time t (in forward diffusion process)
        # Remark: 'self.forward_sample_z_t' returns the state and the noise 'epsilon'
        batch_z_t = self.forward_sample_z_t(batch_x, batch_Q_t)

        # Get the log-probability of the property model as 1D torch tensor of shape (batch_size,)
        property_model_output = self._get_property_model_log_prob(
            batch_z_t, batch_t, batch_y, return_logits=return_logits
        )
        if not return_logits:
            log_prob = property_model_output
        else:
            log_prob, logits = property_model_output

        # The loss is given by the negative log-probability summed over the
        # entire batch
        loss = -torch.sum(log_prob)

        # If the loss should be 'intensive', divide the 'extensive' loss
        # determined above by the number of points in the batch
        if intensive_loss:
            loss = loss / batch_size

        outputs = dict(loss=loss, batch_size=batch_size)
        if return_logits:
            outputs["logits"] = logits
        return outputs

    def _denoise_z_t(self, batch_z_t, batch_t):
        """
        Pass the noisy input at time t through the denoising model
        """
        if self.model_dict["denoising"] is None:
            raise ValueError("Denoising model is None when using it to denoise input")
        # Shape (B, D, S)
        self.model_dict["denoising"].eval()
        batch_x_0_logits = self.model_dict["denoising"](dict(x=batch_z_t), batch_t)
        # Shape (B, D)
        batch_x_0 = torch.max(batch_x_0_logits, dim=2)[1]
        return batch_x_0

    def _sample_times_over_batch(self, batch_size):
        """
        Sample a time t in [0, 1] for each point in the batch.

        Args:
            batch_size (int): Batch size, i.e. number of points
                for which a time should be sampled for)

        Return:
            (torch.tensor): Sampled times as 1D torch tensor of shape
                (batch_size,) containing times in [0, 1].
        """
        # I.i.d. draw batch_size many samples from [0, 1]
        batch_t = torch.rand(batch_size).to(self.device)  # shape: (batch_size,)

        # Set a minimum time s.t. t ~ Unif(epsilon, 1) to avoid training instability
        # when t --> 0 (where \hat{R_t}) becomes ill-conditioned
        batch_t *= (1.0 - self.min_time) + self.min_time
        return batch_t

    def _parse_time(self, t, batch_size):
        """
        Parse the input time 't' using the input 'x' as 'template'.

        Args:
            t (torch.tensor, float or int): Time to be parsed either as number,
                as 2D torch tensor of shape (batch_size, 1), or as 1D torch
                tensor of shape (batch_size, )
            batch_size (int): Batch size.

        Returns:
            (torch.tensor): The time as 1D torch tensor of shape (batch_size,).
        """
        # Differ cases for type of input t
        if torch.is_tensor(t):
            # Differ cases for dimension of time t
            if t.dim() == 1:
                # Check that time t has the correct shape
                if t.shape[0] != batch_size:
                    err_msg = f"If the time is passed as 1D torch tensor it must have shape (batch_size,), got '{t.shape}' instead."
                    raise ValueError(err_msg)

                parsed_t = t

            elif t.dim() == 2:
                # Check that time t has the correct shape
                if not (t.shape[0] == batch_size and t.shape[1] == 1):
                    err_msg = f"If the time is passed as 2D torch tensor it must have shape (batch_size, 1), got '{t.shape}' instead."
                    raise ValueError(err_msg)

                # If the checks above were fine, assign time to the parsed time
                parsed_t = t.squeeze()

            else:
                err_msg = f"If the time is passed as torch tensor it must be either a 2D tensor of shape (batch_size, 1) or a 1D tensor of shape (batch_size,), got dimension '{t.dim()}' instead."
                raise ValueError(err_msg)

        elif isinstance(t, (int, float)):
            # Parse the time as a torch tensor
            parsed_t = t * torch.ones(batch_size)
        else:
            err_msg = f"The input time 't' must be either a number (float or int) or a torch.tensor of shape (batch_size, ) where the expected batch size is '{batch_size}'."
            raise TypeError(err_msg)

        return parsed_t

    ##############################################################################################################
    ### SPECIFY METHODS THAT HAVE TO BE IMPLEMENTED BY DERIVED CLASSES
    ##############################################################################################################

    def diffusion_batch_loss(self, *args, **kwargs):
        """
        Return the batch loss for diffusion.
        """
        raise NotImplementedError(
            "The method 'diffusion_batch_loss' must be implemented by any derived class."
        )

    def forward_sample_z_t(self, *args, **kwargs):
        """
        Sample z_t in forward diffusion (i.e. noised state at time t).
        """
        raise NotImplementedError(
            "The method 'forward_sample_z_t' must be implemented by any derived class."
        )

    def generate(self, *args, **kwargs):
        """
        Generate 'novel' points by sampling from p(z_1) and then using ancestral
        sampling (backwards in time) to obtain 'novel' \hat{z}_0=\hat{x}.
        """
        raise NotImplementedError(
            "The method 'generate' must be implemented by any derived class."
        )

    def _backward_sample_z_t_next(self, *args, **kwargs):
        """
        Sample next state backward in time from last (i.e. current) state.
        """
        raise NotImplementedError(
            "The method '_backward_sample_z_t_next' must be implemented by any derived class."
        )

    def _sample_z_1(self, *args, **kwargs):
        """
        I.i.d. drawn z_1 for each point in a batch.
        """
        raise NotImplementedError(
            "The method '_sample_z_1' must be implemented by any derived class."
        )

    def _get_property_model_log_prob(self, batch_z_t, batch_t, batch_y):
        """
        Return the log probability of the property log model.
        """
        raise NotImplementedError(
            "The method '_get_property_model_log_prob' must be implemented by any derived class."
        )
