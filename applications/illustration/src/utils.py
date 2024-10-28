# Import public modules
import copy
import os
import random
import scipy
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from numbers import Number
from numpy.typing import ArrayLike
from pathlib import Path
from torch.distributions.categorical import Categorical
from typing import Optional, List, Tuple

# Import custom modules
from src import fm_utils

def set_random_seed(random_seed:int) -> None:
    """ 
    Set random seed(s) for reproducibility. 
    
    Args:
        random_seed (int): Random seed to be used as basis 
            for the seeds of 'random', 'numpy', and 'torch'
            modules.
    
    """
    # Set random seeds for any modules that potentially use randomness
    random.seed(random_seed)
    np.random.seed(random_seed+1)
    torch.random.manual_seed(random_seed+2)

def to_numpy(x:ArrayLike) -> np.ndarray:
    """
    Map input x to numpy array.
    
    Args:
        x (np.ndarray or torch.tensor): Input to be mapped to numpy array.
    
    Return:
        (np.ndarray): Input x casted to a numpy array.
        
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        err_msg = f"The input must be either a numpy array or a torch tensor, got type {type(x)} instead."
        raise TypeError(err_msg)

def generate_toy_dataset(num_pnts:int, 
                         p_data_list:List[ArrayLike], 
                         seed:int=100) -> Tuple[ArrayLike, ArrayLike]:
    """
    Generate a discrete space toy dataset.

    Args:
        num_nts (int): Number of points the dataset 
            should have.
        p_data_list (list): List of discrete state space 
            distributions for each dimension.
            For example, a 1D state space will have a
            p_data_list containing a single array-like
            element that corresponds to the probability
            distribution of these 1D states.
        seed (int): Random seed for random state used when
            sampling states.
            (Default: 100)

    Return:
        (ArrayLike): States 'x' of all points.
        (ArrayLike): Property values 'y' of all points
    
    """
    # Set the random seed
    set_random_seed(seed)

    # Determine the dimensionality of the state space as 
    # the number of elements of p_data list.
    D = len(p_data_list)

    # Sample discrete states in each dimension for each point
    # and stack the resulting single-dimensional states to a 
    # multi-dimensional state for each point
    x_d_list = list()
    for d, p_data in enumerate(p_data_list):
        x_enc_d = scipy.stats.multinomial(n=1, p=p_data).rvs(num_pnts) # (B, S)
        x_d     = np.argmax(x_enc_d, axis=1) # (B,)
        x_d_list.append(x_d.reshape(-1, 1))

    x_data = np.stack(x_d_list, axis=1).squeeze() # (B, D) if D>1 and (B,) for D=1

    # If D is 1, squeezing will make x_data a 1D array of shape (B,), reshape it to (B, 1)
    if D==1:
        x_data = x_data.reshape(-1, 1)

    # Map each state to a unique y label (i.e. enumerate space with state-indies corresponding to y)
    y_data = np.array([get_state_label(x, D) for x in x_data])

    return x_data, y_data

def get_state_label(x:torch.tensor, 
                    D:int) -> torch.tensor:
    """
    Return the state 'label' (i.e., the label/index 
    of the discrete space) of the input state 'x'.
    Remark: This function is only implemented for 1D (D=1) states,
            where 'x' is equivalent to its label/index.

    Args:
        x (torch.tensor): Input state.
        D (int): Discrete state space dimension.

    Return:
        (torch.tensor): State label/index.
            
    """
    if D!=1:
        err_msg = f"'get_state_label' is only implemented for D=1."
        raise ValueError(err_msg)
    
    # For D=1, the state label/index corresponds to the state itself.
    return x

def generate_p_data_cat_probs(mu_1:float, 
                              mu_2:float, 
                              sigma_1:float, 
                              sigma_2:float, 
                              S:int) -> List[ArrayLike]:
    """
    Generate the probability distribution of the 1D discrete 
    (i.e., categorical) states in the state space of 
    cardinality S. 
    Use a discretized Gaussian mixture model with two 
    Gaussians where the discretization is such that
    each state {0, ..., S-1} is linked to a discretization 
    bin of size 1.
    For example, state j occupies the bin [j-1/2, j+1/2).
    
    Args:
        mu_1 (float): Mean of the first Gaussian.
        mu_2 (float): Mean of the second Gaussian.
        sigma_1 (float): Sigma of the first Gaussian.
        sigma_2 (float): Sigma of the second Gaussian.
        S (int): Cardinality of the discrete space.
    
    Return:
        (list): List containing a single array-like element 
            containing the probabilities of each discrete state.
            Remark: Here, we only deal with probability distributions
                    over 1D state spaces (thus only a single element),
                    but this could be generalized to higher dimensions
                    by saving the marginal probability distributions
                    in each dimension as individual elements.
                    (I.e., the list length reflects the discrete state
                    space dimensionality here.)

    """
    # Create the bin edges
    x_bin_edges = np.linspace(-0.5, S-0.5, S+1)

    # Define the two normal (i.e., Gaussian) distributions
    norm_1 = scipy.stats.norm(loc=mu_1, scale=sigma_1)
    norm_2 = scipy.stats.norm(loc=mu_2, scale=sigma_2)

    # Discretize these normal distributions obtaining 'weights'
    # for each bin stemming from each of the two distributions.
    cdfs_1    = norm_1.cdf(x_bin_edges)
    cdfs_2    = norm_2.cdf(x_bin_edges)
    weights_1 = cdfs_1[1:]-cdfs_1[:-1]
    weights_2 = cdfs_2[1:]-cdfs_2[:-1]

    # Add the weights and normalize them over all bins to
    # obtain the probabilities of each bin (i.e., state)
    # Remark: Do not confuse 'weights' with the Gaussian mixture
    #         weights, which are 1/2 for each Gaussian here.
    weights = weights_1 + weights_2
    probs   = weights/np.sum(weights).squeeze()
    probs   = probs.squeeze()

    return [probs]

def train_model(model_name:str, 
                models_dict:dict, 
                optims_dict:dict, 
                dataloader:object, 
                S:int, 
                num_epochs:int=100) -> None:
    """
    Train a specific model.

    Adapted from example code in:
    A. Campbell et al., Generative Flows on Discrete State Spaces (2024).

    Args:
        model_name (str): Name of the model to be trained. 
            Should be a key of 'models_dict'.
        models_dict (dict): Dictionary containing the model names as
            dictionary-keys and the model objects as dictionary-values.
        optims_dict (dict): Dictionary containing the model names as
            dictionary-keys and the optimizer objects for the models as 
            corresponding dictionary-values.
        dataloader (object): (torch) dataloader object of the train set.
        S (int): Cardinality of each discrete state space dimension.
        num_epochs (int): Number of epochs.
            (Default: 100)
    
    """
    # Variables, B, D, S for batch size, number of dimensions and state space size respectively
    # Assume we have a model that takes as input xt of shape (B, D) and time of shape (B,)
    # and outputs x1 prediction logits of shape (B, D, S).
    losses = list()
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_losses = list()
        for batch_data in dataloader:
            # Extract quantities from the batch
            x_1 = batch_data['x']

            # x_1 has shape (B, D)
            B = x_1.shape[0]
            D = x_1.shape[1]
            device = x_1.device

            # Set the model into train mode
            models_dict[model_name].train()

            # Zero-out gradients
            optims_dict[model_name].zero_grad()

            # Sample time from Uniform(0, 1)
            t = torch.rand((B,)).to(device )

            # Clone x_1 into to be sampled x_1
            x_t = x_1.clone()

            # Sample uniform noise, construct a corruption mask, and 
            # apply both to x_t thereby 'sample' x_t.
            uniform_noise = torch.randint(0, S, (B, D)).to(device )
            corrupt_mask  = torch.rand((B, D)).to(device ) < (1 - t[:, None]).to(device )
            x_t[corrupt_mask] = uniform_noise[corrupt_mask]

            # Differ cases depending on the mode
            if model_name=='denoising_model':
                # Determine the logits of the model and compute the cross-entropy
                # Remark: The denoising model returns logits corresponding to state-space probabilities
                logits = models_dict[model_name](x_t, t) # (B, D, S)
                loss = F.cross_entropy(logits.transpose(1, 2), x_1, reduction='mean')
            elif model_name=='predictor_model':
                # The loss is given by the mean negative log-likelihood
                batch_data_t      = copy.deepcopy(batch_data)
                batch_data_t['x'] = x_t
                loss = -torch.mean(models_dict[model_name].log_prob(batch_data_t, t))
            else:
                err_msg = f"Model name '{model_name}' is not allowed, use 'denoising' or 'predictor'."
                raise ValueError(err_msg)
                                    
            # Back-propagate gradients and update model parameters
            loss.backward()
            optims_dict[model_name].step()

            epoch_losses.append(loss.item())
        
        losses.append(np.mean(epoch_losses))

    plt.figure()
    plt.title(f"Loss-curve: {model_name}")
    plt.plot(losses, 'b-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim([0, np.max(losses)*1.01])
    plt.show()

def generate(models_dict:dict, 
             B:int, 
             D:int, 
             S:int, 
             seed:int=43, 
             dt:float=0.001, 
             noise:float=0.0, 
             x1_temp:float=1.0,
             device:str='cpu',
             y_guide:Optional[Number]=None,
             guide_temp:float=1.0) -> Tuple[list, list]:
    """
    Generate x-samples.

    Adapted from example code in:
    A. Campbell et al., Generative Flows on Discrete State Spaces (2024).

    Args:
        models_dict (dict): Dictionary containing the model names as
            dictionary-keys and the model objects as dictionary-values.
        B (int): Batch size (i.e., number of samples/particles here)
        D (int): Dimensionality of the discrete state space.
        S (int): Cardinality of each discrete state space dimension.
        seed (int): Seed for random state of generation.
            (Default: 42)
        dt (float): Time step for Euler method.
            (Default: 0.001)
        stochasticity (float): Stochasticity value used for continuous-time
            discrete-space flow models (i.e., '\eta').
            (Default: 0.0)
        x1_temp (float): Temperature of the unconditional model.
            (Default: 1.0)
        device (str): Device to use as string.
            (Default: 'cpu')
        y_guide (None or number): Property value to guide generation to 
            using property-guidance with the property predictor model.
            If None, do not guide (i.e., unconditional generation).
            (Default: None)
        guide_temp (float): Guidance temperature.
            (Default: 1.0)

        Return:
            (list) List of time steps.
            (list) List of stats of all particles for each time step.

    """
    # This function is only implemented for exact guidance; i.e. not for Taylor-approximated guidance (TAG)
    use_tag = False # Use exact sampling

    # Variables, B, D, S for batch size, number of dimensions and state space size respectively
    # Assume we have a model that takes as input xt of shape (B, D) and time of shape (B,)
    # and outputs x1 prediction logits of shape (B, D, S).
    set_random_seed(seed)
    xt_list = list()
    t_list  = list()
    t       = 0.0 # Initial time
    x0      = torch.randint(0, S, (B, D)).to(device) # Sample x_0 and assign it to initial x_t
    xt      = x0.clone()
    xt_list.append(xt)

    # Remark: The last time should be '1-dt' (to avoid t=1)
    t_list = list(np.arange(0, 1, dt))
    for t in tqdm.tqdm(t_list):
        ts = t * torch.ones((B,)).to(device)
        logits = models_dict['denoising_model'](xt, ts) # (B, D, S)
        logits /= x1_temp
        x1_probs = F.softmax(logits, dim=-1) # (B, D, S)
        x1_probs_at_xt = torch.gather(x1_probs, -1, xt[:, :, None]) # (B, D, 1)

        # Don’t add noise on the final step
        if t + dt < 1.0:
            N = noise
        else:  
            N=0

        # Get the rates:
        R_t = ((1 + N + N * (S - 1) * t ) / (1-t)) * x1_probs + N * x1_probs_at_xt

        # Adjut the rates
        if y_guide is not None:
            predictor_model = models_dict['predictor_model']
            y_guide_name    = predictor_model.y_guide_name

            predictor_log_prob = lambda xt, t: predictor_model.log_prob({
                'x': xt, 
                y_guide_name: y_guide*torch.ones((xt.shape[0],), dtype=torch.long, device=device)}, 
                t, 
                is_x_onehot=False
            )

            # Get the guided rates
            R_t = fm_utils.get_guided_rates(
                predictor_log_prob, 
                xt, 
                t, 
                R_t, 
                S,
                use_tag=use_tag, 
                guide_temp=guide_temp,
            )

        # Set the diagonal of the rates to negative row sum
        R_t.scatter_(-1, xt[:, :, None], 0.0)
        R_t.scatter_(-1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))

        # Obtain probabilities from the rates
        step_probs = (R_t * dt).clamp(min=0.0, max=1.0)
        step_probs.scatter_(-1, xt[:, :, None], 0.0)
        step_probs.scatter_(-1, xt[:, :, None], (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0))
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        # Sample x_t and append it            
        xt = Categorical(step_probs).sample() # (B, D) 38
        xt_list.append(xt)

    xt_list = [to_numpy(xt).squeeze() for xt in xt_list]

    return xt_list, t_list

class DictDataset(torch.utils.data.Dataset):
    """
    Define a custom dataset that returns a dictionary with dictionary-keys 
    'x' if sliced where the dictionary-values will correspond to the 
    sliced x data values.
    The same can be done for additional key-value pairs within kwargs.
    (E.g., one could pass 'y=<torch.tensor>' that would add also a 'y' entry).
    """
    def __init__(self, 
                 x:torch.tensor, 
                 device:Optional[object]=None, 
                 **kwargs) -> None:
        """
        Args:
            x (torch.tensor): 2D torch tensor of shape (#datapoints, #x-features).
            device (None or object): Device the data should be mapped to.
                If None no device is specified.
                (Default: None)
            **kwargs: Additional entries as key-value pairs.
        
        """
        if device is None:
            # Get the device
            device = torch.device('cpu')

        # Assign x and y to the corresponding class attributes
        self.device    = device
        self.x         = x.to(self.device)
        self.vars_dict = {key: value.to(self.device) for key, value in kwargs.items()}

    def to(self, device:object) -> None:
        """
        Map the data to the specified device.
        
        Args:
            device (object): The device the data
                should be mapped to.

        """
        # Update the device class attribute
        self.device = device

        # Map everything to the wished device
        self.x         = self.x.to(self.device)
        self.vars_dict = {key: value.to(self.device) for key, value in self.vars_dict.items()}

    def __len__(self) -> int:
        """ Return the number of datapoints (as integer). """
        # Remark: self.x should have shape (#datapoints, #x-features)
        return self.x.shape[0]

    def __getitem__(self, ix:int) -> dict:
        """
        Implement slicing. 
        
        Args:
            ix (int): Datapoint index.

        Return:
            (dict): Item correspondig to 
                datapoint index.
            
        """
        # Cast ix to a list if it is a tensor
        if torch.is_tensor(ix):
            ix = ix.tolist()        

        # Return a dictionary containing the data slices for ix
        ix_data_dict = {'x': self.x[ix]}
        ix_data_dict.update({var_name: var_values[ix] for var_name, var_values in self.vars_dict.items()})
        return ix_data_dict

def create_folder_if_inexistent(folder_path) -> None:
    """
    Create a folder (and its parents) if it does not exist yet. 
    
    Args:
        folder_path (str or Path): Path to the to be created (if it doesn't exist) folder.
    
    """
    if not os.path.exists(folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        print(f"Created the following inexistent folder (and any inexistent parents): {folder_path}")
