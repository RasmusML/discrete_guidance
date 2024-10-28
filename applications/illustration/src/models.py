# Import public modules
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List

# Import custom modules
from . import utils

# Define the denoising model
class DenoisingModel(torch.nn.Module):
    def __init__(self, 
                 D:int, 
                 S:int, 
                 hidden_dims:List[int]=[100], 
                 p_dropout:float=0.2, 
                 eps:float=1e-10) -> None:
        """
        Args:
            D (int): Dimensionality of the discrete state space.
            S (int): Cardinality of each discrete state space dimension.
                (i.e., number of single-dimension states in each dimension)
            hidden_dims (list of ints): List of hidden-layer dimensions.
                E.g., [100] will be a neural network with a single hidden
                layer of dimension 100.
                (Default: [100])
            p_dropout (float): Dropout probability.
                (Default: 0.2)
            eps (float): Epsilon used for numerical stability.
                (Default: 1e-10)

        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.D           = D
        self.S           = S
        self.hidden_dims = hidden_dims # Number of hidden layers
        self.p_dropout   = p_dropout
        self.eps         = eps
        
        # Stack time as scalar to spatial-input
        self.input_dim = self.D*self.S + 1
        
        # Define the output dimension
        self.output_dim = self.D*self.S 

        # Define the linear parts of the model
        linear_list       = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        linear_list      += [torch.nn.Linear(self.hidden_dims[layer_id-1], self.hidden_dims[layer_id]) for layer_id in range(1, len(self.hidden_dims))]
        self.linear_list  = torch.nn.ModuleList(linear_list)
        self.linear_last  = torch.nn.Linear(self.hidden_dims[-1], self.output_dim)

        # Define an activation function
        self.activation_fn = torch.nn.ReLU()

        # Define (global) dropout function
        self.dropout_fn = torch.nn.Dropout(p=self.p_dropout, inplace=False)

    def encode_x(self, x:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)

        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()

    def forward(self, 
                xt:torch.tensor, 
                t:torch.tensor) -> torch.tensor:
        """"
        Define forward pass of the model.
        
        Args:
            xt (torch.tensor): Shape (B, D).
            t (torch.tensor): Shape (B,).

        Return:
            (torch.tensor): Logits of shape (B, D, S).
        
        """
        # Encode space and flatten from (B, D, S) to (B, D*S)
        xt_enc = self.encode_x(xt) # (B, D, S)
        xt_enc = xt_enc.view(-1, self.D*self.S) # (B, D*S)

        # Encoding t corresponds to reshaping it from (B,) to (B, 1) here
        t_enc = t.view(-1, 1)

        # Stack scalar (per batch-point) time to encoded x
        h = torch.cat([xt_enc, t_enc], dim=-1) 

        # Perform pass through the network
        for layer_id in range(len(self.linear_list)):
            h = self.dropout_fn(h)
            h = self.linear_list[layer_id](h)
            h = self.activation_fn(h)

        # Shape (B, #classes)
        h = self.dropout_fn(h)
        h = self.linear_last(h)
    
        # Bring logits in correct shape
        logits = h.view(-1, self.D, self.S) # (B, D, S)
    
        return logits
    
class PredictorModel(torch.nn.Module):
    output_type = 'continuous_value'

    def __init__(self, 
                 y_guide_name: str, 
                 D:int, 
                 S:int, 
                 hidden_dims:List[int]=[100], 
                 p_dropout:float=0.2, 
                 sigma_noised:float=1.0, 
                 sigma_unnoised:float=1.0, 
                 eps:float=1e-10) -> None:
        """
        Args:
            y_guide_name (str): Name of property that should be predicted.
            D (int): Dimensionality of the discrete state space.
            S (int): Cardinality of each discrete state space dimension.
                (i.e., number of single-dimension states in each dimension)
            hidden_dims (list of ints): List of hidden-layer dimensions.
                E.g., [100] will be a neural network with a single hidden
                layer of dimension 100.
                (Default: [100])
            p_dropout (float): Dropout probability.
                (Default: 0.2)
            sigma_noised (float): Sigma of the properties 
                (for which a predictor is setup here) of 
                the fully noised samples (at t=0).
                (Default: 1.0)
            sigma_unnoised (float): Sigma of the properties 
                (for which a predictor is setup here) of 
                the unnoised samples (at t=1).
                (Default: 1.0)
            eps (float): Epsilon used for numerical stability.
                (Default: 1e-10)

        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.D                = D
        self.S                = S
        self.y_guide_name     = y_guide_name
        self.hidden_dims      = hidden_dims
        self.p_dropout        = p_dropout
        self.eps              = eps
        self.output_layer_dim = 1
        self.sigma_noised     = sigma_noised
        self._sigma_unnoised  = sigma_unnoised

        # Stack time as scalar to spatial-input
        self.input_dim = self.D*self.S + 1

        # Define the model parts
        linear_list       = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        linear_list      += [torch.nn.Linear(self.hidden_dims[layer_id-1], self.hidden_dims[layer_id]) for layer_id in range(1, len(self.hidden_dims))]
        self.linear_list  = torch.nn.ModuleList(linear_list)
        self.linear_last  = torch.nn.Linear(self.hidden_dims[-1], self.output_layer_dim) # Output layer

        # Define an activation function
        self.activation_fn = torch.nn.ReLU()

        # Define (global) dropout function
        self.dropout_fn = torch.nn.Dropout(p=self.p_dropout, inplace=False)
    
    def encode_x(self, x:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)

        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()

    def forward(self, 
                batch_data_t:torch.tensor, 
                t:torch.tensor, 
                is_x_onehot:bool=False) -> torch.tensor:
        """
        Define forward pass of the model.
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): Probabilities for each component of the encoded property 'y'
                as 2D torch tensor of shape (batch_size, dim[encoded(y)])
        
        """
        # Get xt
        xt = batch_data_t['x']

        # Differ the cases where xt is already encoded or not
        if is_x_onehot:
            # xt is already encoded
            xt_enc = xt # (B, D, S)
        else:
            # xt has to be encoded
            xt_enc = self.encode_x(xt) # (B, D, S)

        # Flatten features
        xt_enc = xt_enc.view(-1, self.D*self.S) # (B, D*S)

        # Encoding t corresponds to reshaping it from (B,) to (B, 1) here
        t_enc = t.view(-1, 1)

        # Stack scalar (per batch-point) time to encoded x
        h = torch.cat([xt_enc, t_enc], dim=-1) 

        # Perform pass through the network
        for layer_id in range(len(self.linear_list)):
            h = self.dropout_fn(h)
            h = self.linear_list[layer_id](h)
            h = self.activation_fn(h)
        
        # Perform pass through last layer
        h = self.dropout_fn(h)
        h = self.linear_last(h) # (B, #classes)

        return h.squeeze()
    
    
    def log_prob(self, 
                 batch_data_t:torch.tensor, 
                 t:torch.tensor, 
                 is_x_onehot:bool=False) -> torch.tensor:
        """
        Return the log probability given the data. 
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): (Batched) log-probability for each point in the batch as
                1D torch tensor of shape (batch_size,).
        
        """
        # Get the y-data
        y_data = batch_data_t[self.y_guide_name]

        # Determine the class-probabilities
        y_pred = self.forward(batch_data_t, t, is_x_onehot=is_x_onehot) # Shape (B, #classes)

        # Determine sigma(t) and log(sigma(t))
        sigma_t     = self.get_sigma_t(t).squeeze()+self.eps
        log_sigma_t = torch.log(sigma_t)

        # Calculate the log_prob per point
        square_diff = (y_data.squeeze()-y_pred.squeeze())**2/(2*sigma_t**2)
        log_prob = -square_diff-log_sigma_t-np.sqrt(2*np.pi)

        return log_prob
    
    @property
    def sigma_unnoised(self) -> torch.tensor:
        """
        Return the unnoised sigma based on the model parameter 'self.log_sigma_unnoised'.

        Return:
            (torch.tensor): Unnoised sigma.

        """
        return self._sigma_unnoised

    def get_sigma_t(self, t:torch.tensor) -> torch.tensor:
        """
        Return sigma(t) as an interpolation between the
        noised sigma at t=0 and the unnoised sigma at t=1.
        
        Remark: 
        At t=1 we have the data distribution (unnoised)
        and at t=0 we have the noised distribution.
        Thus, interpolate the predictor sigma in a similar way.

        Args:
            t (torch.tensor): Times.

        Return:
            (torch.tensor): Determined sigma(t).
        
        """
        # Remarks: (1) Interploate the log(sigma) not sigma here.
        #          (2) Use t^(1/4) instead of t as interpolation
        #              variable here.
        log_sigma_t = t**(1/4)*math.log(self.sigma_unnoised)+(1-t**(1/4))*math.log(self.sigma_noised)
        return torch.exp(log_sigma_t)

    def plot_sigma_t(self, device:object) -> object:
        """
        Plot sigma(t) vs. t and return the resulting figure.

        Args:
            device (object): The device the model parameters 
                are on.
        
        Return:
            (object): Matplotlib figure object.
        
        """
        fig = plt.figure()
        t = torch.linspace(0, 1, 1000).to(device)
        sigma_t = self.get_sigma_t(t)
        plt.plot(utils.to_numpy(t), utils.to_numpy(sigma_t), 'b-')
        plt.xlabel('t')
        plt.ylabel('sigma(t)')
        plt.xlim([0, 1])
        plt.ylim([0, max(utils.to_numpy(sigma_t))*1.05])
        plt.show()
        return fig