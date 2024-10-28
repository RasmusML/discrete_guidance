# Import public modules
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

def plot_x_distribution(x_values:ArrayLike,
                        S:int, 
                        title:Optional[str]=None) -> None:
    """
    Plot the distribution of categorical 1D (in a state space 
    of cardinality S) states of different datapoints.

    Args:
        x_values (array-like): 1D states of different datapoints
            of shape (#datapoints,) or (#datapoints, 1). 
        S (int): Cardinality of categorical 1D state space.
        title (None or str): Optional title.
            If None, do not display any title.
            (Default: None)
    
    """
    plt.figure()
    if title is not None:
        plt.title(title)
    bin_edges   = np.linspace(-0.5, S-0.5, S)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    plt.hist(x_values, bin_edges, density=True)
    if S<10:
        plt.xticks(bin_centers)
    plt.xlabel('')
    plt.yticks([])
    plt.ylabel('')
    plt.show()

def make_time_evolution_plot_predictor(generation_dict:dict, 
                                       plot_specs:dict, 
                                       figpath:Optional[str]=None) -> object:
    """
    Make the time-evolution plot for the predictor model.

    Args:
        generation_dict (dict): Dictionary containing the results
            of a generation run.
        plot_specs (dict): Plot specifications.
        figpath (None or str): Figure save path.
            If None, do not save the figure
            (Default: None)
    
    Return:
        (object): Matplotlib figure object.
    
    """
    # Define labels
    p_0_label = r'$p(y|x_{0})$'
    p_t_label = r'$p(y|x_{t})$'
    p_1_label = r'$p(y|x_{1})$'

    # Extract from generation dict
    Z                   = generation_dict['predictor_probs_matrix']
    postprocessing_dict = generation_dict['postprocessing_dict']

    # Extract from postprocessing dictionary
    x_bin_edges    = postprocessing_dict['x_bin_edges']
    t_bin_edges    = postprocessing_dict['t_bin_edges']

    # Determine the limits
    x_lim = [x_bin_edges.min(), x_bin_edges.max()]

    # Make subplots
    fig, axs = plt.subplots(1, 3, figsize=plot_specs['figsize'], width_ratios=plot_specs['sp_width_ratios'])

    # Middel pannel: Plot p(x_t)
    ax = axs[1]
    # Get information from the 2D histogram
    extent = [t_bin_edges.min(), t_bin_edges.max(), x_bin_edges.min(), x_bin_edges.max()]
    ax.imshow(Z.T, extent=extent, cmap=plot_specs['cmap']['predictor'], aspect='auto', origin='lower')

    # x-axis is time here
    ax.set_title(p_t_label, fontsize=plot_specs['title_fs'])
    ax.set_xlabel('Time', fontsize=plot_specs['axis_fs'], labelpad=plot_specs['axis_label_pad'])
    ax.set_xlim([t_bin_edges.min(), t_bin_edges.max()])
    ax.set_xticks([0, 1])
    ax.tick_params(axis='both', labelsize=plot_specs['tick_fs'])
    
    # y-axis is state space here
    ax.set_yticks([])
    ax.set_ylim([x_bin_edges.min(), x_bin_edges.max()])

    # Left pannel: Plot p(x_0)
    ax = axs[0]
    bar_counts  = Z[0, :]
    bar_centers = np.linspace(0, bar_counts.shape[0], bar_counts.shape[0])

    # Show the bar plot
    cmap_bar(bar_centers, bar_counts, cmap=plot_specs['cmap']['predictor'], ax=ax, orientation='horizontal', max_counts=np.max(Z), color_dampening=plot_specs['color_dampening'], max_scaling=plot_specs['max_scaling']['predictor'])

    # Set a contrast background color
    ax.set_facecolor(plot_specs['hist_background_color'])

    # x-axis is histogram-density here
    ax.xaxis.set_inverted(True) # Make the histogram-density be zero on the right side (default would be on the left side)
    ax.set_title(p_0_label, fontsize=plot_specs['title_fs'])
    ax.set_xticks([])
    # y-axis is state space here
    ax.set_yticks([])
    ax.set_ylabel('Discrete-states', fontsize=plot_specs['axis_fs'])
    ax.set_ylim(x_lim)

    # Right pannel: Plot p(x_1)
    ax = axs[2]
    bar_counts  = Z[-1, :]
    bar_centers = np.linspace(0, bar_counts.shape[0], bar_counts.shape[0])
    # Show the bar plot
    cmap_bar(bar_centers, bar_counts, cmap=plot_specs['cmap']['predictor'], ax=ax, orientation='horizontal', max_counts=np.max(Z), color_dampening=plot_specs['color_dampening'], max_scaling=plot_specs['max_scaling']['predictor'])

    # Set a contrast background color
    ax.set_facecolor(plot_specs['hist_background_color'])

    # x-axis is histogram-density here
    ax.set_title(p_1_label, fontsize=plot_specs['title_fs'])
    ax.set_xticks([])
    # y-axis is state space here
    ax.set_yticks([])
    ax.set_ylim(x_lim)

    # Adjust subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.05, 
                        hspace=0.4)
    plt.show()

    if figpath is not None:
        fig.savefig(figpath)

    return fig

def make_time_evolution_plot(generation_dict:dict, 
                             plot_specs:dict, 
                             conditional:bool=False, 
                             figpath:Optional[str]=None) -> object:
    """
    Make the time-evolution plot (for conditonal or unconditional generation).

    Args:
        generation_dict (dict): Dictionary containing the results
            of a generation run.
        plot_specs (dict): Plot specifications.
        conditional (bool): Was generation conditonal (conditional=True) 
            or unconditonal (conditional=False).
            (Default: False)
        figpath (None or str): Figure save path.
            If None, do not save the figure
            (Default: None)
    
    Return:
        (object): Matplotlib figure object.
    
    """
    if conditional:
        p_0_label = r'$p(x_{0}|y)$'
        p_t_label = r'$p(x_{t}|y)$'
        p_1_label = r'$p(x_{1}|y)$'
    else:
        p_0_label = r'$p(x_{0})$'
        p_t_label = r'$p(x_{t})$'
        p_1_label = r'$p(x_{1})$'

    # Extract from generation dict
    t_list              = generation_dict['t_list']
    x_t_list            = generation_dict['x_t_list']
    postprocessing_dict = generation_dict['postprocessing_dict']


    # Extract from postprocessing dictionary
    x_bin_edges    = postprocessing_dict['x_bin_edges']
    t_bin_edges    = postprocessing_dict['t_bin_edges']
    Z              = postprocessing_dict['Z']
    num_jumps_dict = postprocessing_dict['num_jumps_dict'] # Dictionary mapping number of jumps to list of trajectories with this number of jumps

    # Determine the limits
    x_lim = [x_bin_edges.min(), x_bin_edges.max()]

    # Make subplots
    fig, axs = plt.subplots(1, 3, figsize=plot_specs['figsize'], width_ratios=plot_specs['sp_width_ratios'])

    # Middel pannel: Plot p(x_t)
    ax = axs[1]

    # Get information from the 2D histogram
    extent = [t_bin_edges.min(), t_bin_edges.max(), x_bin_edges.min(), x_bin_edges.max()]
    ax.imshow(Z.T, extent=extent, cmap=plot_specs['cmap']['generative'], aspect='auto', origin='lower')

    # Plot some trajectories
    x_t_trajectory_counter = 0
    seen_states = list()
    for num_jumps in num_jumps_dict:
        example_x_t_trajectory = num_jumps_dict[num_jumps][0]
        set_diff = set(seen_states)-set(example_x_t_trajectory)
        if set_diff==0:
            continue
        
        ax.plot(t_list+[1], example_x_t_trajectory, '-', color=plot_specs['trajectories_color'], zorder=1, lw=plot_specs['trajectories_lw'])
        x_t_trajectory_counter += 1
        seen_states = list(set(seen_states+list(example_x_t_trajectory)))
        if plot_specs['num_trajectories']<=x_t_trajectory_counter:
            break

    # x-axis is time here
    ax.set_xlabel('Time', fontsize=plot_specs['axis_fs'], labelpad=plot_specs['axis_label_pad'])
    ax.set_title(p_t_label, fontsize=plot_specs['title_fs'])
    ax.set_xlim([t_bin_edges.min(), t_bin_edges.max()])
    ax.set_xticks([0, 1])
    ax.tick_params(axis='both', labelsize=plot_specs['tick_fs'])

    # y-axis is state space here
    ax.set_yticks([])
    ax.set_ylim([x_bin_edges.min(), x_bin_edges.max()])

    # Left pannel: Plot p(x_0)
    ax = axs[0]
    x_0 = x_t_list[0]

    # Show the histogram
    cmap_hist(x_0, x_bin_edges, cmap=plot_specs['cmap']['generative'], ax=ax, orientation='horizontal', max_counts=np.max(Z), color_dampening=plot_specs['color_dampening'], max_scaling=plot_specs['max_scaling']['generative'])

    # Set a contrast background color
    ax.set_facecolor(plot_specs['hist_background_color'])

    # x-axis is histogram-density here
    ax.xaxis.set_inverted(True) # Make the histogram-density be zero on the right side (default would be on the left side)
    ax.set_title(p_0_label, fontsize=plot_specs['title_fs'])
    ax.set_xticks([])

    # y-axis is state space here
    ax.set_yticks([])
    ax.set_ylabel('Discrete-states', fontsize=plot_specs['axis_fs'])
    ax.set_ylim(x_lim)

    # Right pannel: Plot p(x_1)
    ax = axs[2]
    x_1 = x_t_list[-1]

    # Show the histogram
    cmap_hist(x_1, x_bin_edges, cmap=plot_specs['cmap']['generative'], ax=ax, orientation='horizontal', max_counts=np.max(Z), color_dampening=plot_specs['color_dampening'], max_scaling=plot_specs['max_scaling']['generative'])

    # Set a contrast background color
    ax.set_facecolor(plot_specs['hist_background_color'])

    # x-axis is histogram-density here
    ax.set_title(p_1_label, fontsize=plot_specs['title_fs'])
    ax.set_xticks([])

    # y-axis is state space here
    ax.set_yticks([])
    ax.set_ylim(x_lim)

    # Adjust subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.05, 
                        hspace=0.4)
    plt.show()

    if figpath is not None:
        fig.savefig(figpath)

    return fig

def generation_postprocessing(t_list:list, 
                              x_t_list:list, 
                              S:int) -> dict:
    """
    Postprocess trajectories etc. after obtaining trajectories x_t_list
    in generation.

    Args:
        t_list (list): List of timesteps.
        x_t_list (list): List of 1D states (of different particles) at
            each of the time steps in t_list.
        S (int): Caradinality of the 1D state space.

    Return: Dictionary with the postprocessing results in the form:
            }
                'Z': <2D-histogram-of-trajectories>,
                't_bin_edges': <time-bin-edges>,
                'x_bin_edges': <x-bin-edges>,
                'num_jumps_dict': {<number-of-jumps>: <list-of-trajectories>},
            }

    """
    # Generate a matrix by stacking the 1D states of different particles for each
    # time point.
    # From this matrix, we can get the trajectory of a particle as:
    # x_t_trajectory_particle = x_t_matrix[:, <particle-index>]
    x_t_matrix = np.stack(x_t_list)

    # Determine the number of jumps for each trajectory
    num_jumps_dict = collections.defaultdict(list)
    for particle_index in range(x_t_matrix.shape[1]):
        x_t_trajectory = x_t_matrix[:, particle_index]
        num_jumps = len(np.unique(x_t_trajectory))-1
        num_jumps_dict[num_jumps].append(x_t_trajectory)

    # Determine the bins (i.e., edges)
    t_array     = np.array(t_list)
    t_diffs     = np.diff(t_array)
    t_bin_edges = list(t_array[:-1]+t_diffs/2)
    t_bin_edges = [t_array[0]-t_diffs[0]/2] + t_bin_edges + [t_array[-1]+t_diffs[-1]/2]
    t_bin_edges = np.array(t_bin_edges)
    x_bin_edges = np.linspace(-0.5, S-0.5, S+1)

    x_t_values_list = list()
    t_values_list   = list()
    for t, x_t_vals in zip(t_list, x_t_list):
        ts = t*np.ones_like(x_t_vals)
        t_values_list.append(ts)
        x_t_values_list.append(x_t_vals.squeeze())
    x_t_values_array = np.hstack(x_t_values_list)
    t_values_array   = np.hstack(t_values_list)

    _hist = np.histogram2d(t_values_array, x_t_values_array, bins=[t_bin_edges, x_bin_edges])

    postprocessing_dict = {
        'Z': _hist[0],
        't_bin_edges': t_bin_edges,
        'x_bin_edges': x_bin_edges,
        'num_jumps_dict': num_jumps_dict,
    }

    return postprocessing_dict

def cmap_hist(data:ArrayLike, 
              bin_edges:ArrayLike, 
              max_counts:float,
              max_scaling:float=6.0,
              ax:Optional[object]=None,  
              color_dampening:float=1.0,  
              cmap:str='Greys', 
              orientation:str='vertical', 
              **kwargs) -> object:   
    """ 
    Plot histogram with where bars follow a color-map.

    Args:
        data (ArrayLike): Distribution used to construct the histogram. 
        bin_edges (ArrayLike): Bin edges for the histogram.
        max_counts (float): Maximum counts value by which the bar count
            values will be divided by when determining their colors.
            Remark: max_counts should be smaller than max({counts}) for
                    the counts of all bars.
        max_scaling (float): Scaling factor for the upper y-axis limit.
            Remark: The upper y-axis limit is calculated as 
                    mean({histogram-bar-heights})*max_scaling
            (Default: 6.0)
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)
        color_dampening (float): Factory by how much the colors should
            be dampened of a histogram bar.
            Remark: The color of a bar is determined based on the bar
                    height relative to all other bar heights in the
                    histogram. Without dampening, the maximal bar will
                    have the last color of the colorbar.
            (Default: 1.0)
        cmap (str): Colormap.
            (Default: 'Greys') 
        orientation (str): Vertical histogram ('vertical') with vertical bars,
            or horizontal histogram ('horizontal') with horizontal bars.
            (Default:'vertical')
        **kwargs: Keyword-arguments that will be forwarded to either
            matplotlib.pyplot.bar or matplotlib.pyplot.bar function.

    Return:
        (object): Matplotlib axis object.

    """
    # Histogram data to obtain bars
    bar_centers = (bin_edges[:-1]+bin_edges[1:])/2
    _hist = np.histogram(data, bins=bin_edges)
    bar_counts = np.array(_hist[0], dtype=np.float64)

    # Plot the bars
    return cmap_bar(bar_centers, bar_counts, max_counts=max_counts, ax=ax, max_scaling=max_scaling, color_dampening=color_dampening, cmap=cmap, orientation=orientation, **kwargs)

def cmap_bar(bar_centers:ArrayLike, 
             bar_counts:ArrayLike, 
             max_counts:float=1.0,  
             max_scaling:float=6.0, 
             ax:Optional[object]=None,
             color_dampening:float=1.0, 
             cmap:str='Greys', 
             orientation:str='vertical', 
             **kwargs) -> object:
    """ 
    Make a bar plot where the bars are colored according
    to a colormap (where different bar heights have
    different colors).

    Args:
        bar_centers (ArrayLike): Bar centers.
        bar_counts (ArrayLike): Counts of the bars.
        max_counts (float): Maximum counts value by which the bar count
            values will be divided by when determining their colors.
            Remark: max_counts should be smaller than max({counts}) for
                    the counts of all bars.
        max_scaling (float): Scaling factor for the upper y-axis limit.
            Remark: The upper y-axis limit is calculated as 
                    mean({bar_heights})*max_scaling
            (Default: 6.0)
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)
        color_dampening (float): Factory by how much the colors should
            be dampened of a histogram bar.
            Remark: The color of a bar is determined based on the bar
                    height relative to all other bar heights in the
                    histogram. Without dampening, the maximal bar will
                    have the last color of the colorbar.
            (Default: 1.0)
        cmap (str): Colormap.
            (Default: 'Greys') 
        orientation (str): Vertical histogram ('vertical') with vertical bars,
            or horizontal histogram ('horizontal') with horizontal bars.
            (Default:'vertical')
        **kwargs: Keyword-arguments that will be forwarded to either
            matplotlib.pyplot.bar or matplotlib.pyplot.bar function.

    Return:
        (object): Matplotlib axis object.

    """
    # If the axis is not passed, get the current one
    if ax is None:
        ax = plt.gca()

    # Get the colormap object
    cmap_obj = plt.get_cmap(cmap)

    if orientation=='vertical':
        bars = ax.bar(bar_centers, bar_counts, **kwargs)
    elif orientation=='horizontal':
        bars = ax.barh(bar_centers, bar_counts, **kwargs)
    else:
        err_msg = f"Orientation must be either 'vertical' or 'horizontal', got '{orientation}' instead."
        raise ValueError(err_msg)
              
    for bar in bars:
        # Get the bar counts for the current bar
        if orientation=='vertical':
            counts = bar.get_height()
            # The max-counts are the counts assigned the max-value color,
            # i.e. the color corresponding to cmap_obj(1), thus normalize.
            color = cmap_obj(counts/max_counts*color_dampening)
        else:
            counts = bar.get_width()
            # The max-counts are the counts assigned the max-value color,
            # i.e. the color corresponding to cmap_obj(1), thus normalize.
            color = cmap_obj(counts/max_counts*color_dampening)

        bar.set_zorder(1)
        bar.set_facecolor(color)

    if orientation=='vertical':
        ax.set_ylim([0, np.mean(bar_counts)*max_scaling])
    else:
        ax.set_xlim([0, np.mean(bar_counts)*max_scaling])

    return ax

def custom_cmap(color_start:str='w', 
                color_end:str='k') -> object:
    """
    Make a custom colormap by linearly interpolating
    from a start to an end color.

    Args:
        color_start (str): Start color.
        color_end (str): End color.

    Return:
        (object): Matplotlib colorbar object.
    
    """
    return matplotlib.colors.LinearSegmentedColormap.from_list("", [color_start, color_end])
