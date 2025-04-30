import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from typing import List, Dict, Tuple, Optional, Union, Any

def plot_photon_paths(photon_paths: List[np.ndarray], max_paths: int = 20, 
                      figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot 3D trajectories of photons.
    
    Args:
        photon_paths: List of photon paths
        max_paths: Maximum number of paths to plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Limit number of paths to avoid cluttering
    num_paths = min(len(photon_paths), max_paths)
    
    # Generate colors for each path
    colors = plt.cm.jet(np.linspace(0, 1, num_paths))
    
    for i, path in enumerate(photon_paths[:num_paths]):
        # Plot photon path
        ax.plot(path[:, 0], path[:, 1], path[:, 2], '-', color=colors[i], alpha=0.6)
        # Plot start point
        ax.plot([path[0, 0]], [path[0, 1]], [path[0, 2]], 'o', color='g', markersize=6)
        # Plot end point
        ax.plot([path[-1, 0]], [path[-1, 1]], [path[-1, 2]], 's', color='r', markersize=6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Photon Trajectories (showing {num_paths} paths)')
    
    # Make axis lengths equal
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mean_x = np.mean(ax.get_xlim())
    mean_y = np.mean(ax.get_ylim())
    mean_z = np.mean(ax.get_zlim())
    
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    
    return fig

def plot_energy_deposition(energy_grid: np.ndarray, 
                           slice_dim: str = 'z', slice_index: Optional[int] = None,
                           use_log_scale: bool = True,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot a 2D slice of energy deposition.
    
    Args:
        energy_grid: 3D grid of deposited energy
        slice_dim: Dimension to slice along ('x', 'y', or 'z')
        slice_index: Index for the slice (default: middle of grid)
        use_log_scale: Whether to use logarithmic color scale
        figsize: Figure size
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get grid dimensions
    nx, ny, nz = energy_grid.shape
    
    # Set default slice index to middle of grid
    if slice_index is None:
        if slice_dim == 'x':
            slice_index = nx // 2
        elif slice_dim == 'y':
            slice_index = ny // 2
        else:  # 'z'
            slice_index = nz // 2
    
    # Extract 2D slice based on dimension
    if slice_dim == 'x':
        slice_2d = energy_grid[slice_index, :, :]
        title = f'Energy Deposition (X={slice_index})'
        xlabel, ylabel = 'Y', 'Z'
    elif slice_dim == 'y':
        slice_2d = energy_grid[:, slice_index, :]
        title = f'Energy Deposition (Y={slice_index})'
        xlabel, ylabel = 'X', 'Z'
    else:  # 'z'
        slice_2d = energy_grid[:, :, slice_index]
        title = f'Energy Deposition (Z={slice_index})'
        xlabel, ylabel = 'X', 'Y'
    
    # Create visualization with appropriate color scale
    if use_log_scale:
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        mask = slice_2d > epsilon
        if np.any(mask):
            norm = LogNorm(vmin=max(slice_2d[mask].min(), epsilon), vmax=slice_2d.max())
            im = ax.imshow(slice_2d.T, origin='lower', cmap='hot', norm=norm)
        else:
            im = ax.imshow(slice_2d.T, origin='lower', cmap='hot')
    else:
        im = ax.imshow(slice_2d.T, origin='lower', cmap='hot')
    
    # Add colorbar and labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Deposited Energy')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return fig

def plot_energy_profile(energy_grid: np.ndarray, axis: int = 2, 
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot 1D energy deposition profile along a specified axis.
    
    Args:
        energy_grid: 3D grid of deposited energy
        axis: Axis along which to sum (0=X, 1=Y, 2=Z)
        figsize: Figure size
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sum across other dimensions to get 1D profile
    if axis == 0:
        profile = np.sum(energy_grid, axis=(1, 2))
        axis_label = 'X'
    elif axis == 1:
        profile = np.sum(energy_grid, axis=(0, 2))
        axis_label = 'Y'
    else:  # axis == 2
        profile = np.sum(energy_grid, axis=(0, 1))
        axis_label = 'Z'
    
    # Create the plot
    ax.plot(profile, '-o', markersize=4)
    ax.set_xlabel(f'{axis_label} Position')
    ax.set_ylabel('Integrated Deposited Energy')
    ax.set_title(f'Energy Profile along {axis_label} Axis')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_statistics_pie(results: Dict[str, Any], figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Create a pie chart of photon statistics.
    
    Args:
        results: Simulation results dictionary
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract statistics
    stats = {
        'Absorbed': results.get('absorbed', 0),
        'Transmitted': results.get('transmitted', 0),
        'Reflected': results.get('reflected', 0),
        'Escaped': results.get('escaped', 0)
    }
    
    # Filter out zero values
    labels = []
    values = []
    for label, value in stats.items():
        if value > 0:
            labels.append(label)
            values.append(value)
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%',
        textprops={'fontsize': 12},
        shadow=True, 
        startangle=90
    )
    
    # Set equal aspect ratio to ensure the pie is drawn as a circle
    ax.axis('equal')
    ax.set_title('Photon Fate Statistics')
    
    # Enhance visibility of text
    for autotext in autotexts:
        autotext.set_weight('bold')
    
    return fig

def create_summary_plots(results: Dict[str, Any], 
                         max_paths: int = 20,
                         figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create a comprehensive figure with multiple plots summarizing results.
    
    Args:
        results: Simulation results dictionary
        max_paths: Maximum number of photon paths to display
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3)
    
    # 3D photon paths
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    paths = results.get('photon_paths', [])
    num_paths = min(len(paths), max_paths)
    colors = plt.cm.jet(np.linspace(0, 1, num_paths))
    
    for i, path in enumerate(paths[:num_paths]):
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], '-', color=colors[i], alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Photon Trajectories')
    
    # Energy deposition (XY plane)
    ax2 = fig.add_subplot(gs[0, 1])
    energy_grid = results.get('deposited_energy', np.zeros((1, 1, 1)))
    slice_index = energy_grid.shape[2] // 2  # Middle Z slice
    
    # Add small epsilon to avoid log(0) in log scale
    epsilon = 1e-10
    slice_2d = energy_grid[:, :, slice_index]
    mask = slice_2d > epsilon
    
    if np.any(mask):
        norm = LogNorm(vmin=max(slice_2d[mask].min(), epsilon), vmax=slice_2d.max())
        im2 = ax2.imshow(slice_2d.T, origin='lower', cmap='hot', norm=norm)
    else:
        im2 = ax2.imshow(slice_2d.T, origin='lower', cmap='hot')
    
    plt.colorbar(im2, ax=ax2, label='Deposited Energy')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Energy Deposition (Z={slice_index})')
    
    # Energy profile along Z
    ax3 = fig.add_subplot(gs[0, 2])
    profile = np.sum(energy_grid, axis=(0, 1))
    ax3.plot(profile, '-o', markersize=4)
    ax3.set_xlabel('Z Position')
    ax3.set_ylabel('Integrated Energy')
    ax3.set_title('Depth Profile')
    ax3.grid(True, alpha=0.3)
    
    # Statistics pie chart
    ax4 = fig.add_subplot(gs[1, 0])
    stats = {
        'Absorbed': results.get('absorbed', 0),
        'Transmitted': results.get('transmitted', 0),
        'Reflected': results.get('reflected', 0),
        'Escaped': results.get('escaped', 0)
    }
    
    # Filter out zero values
    labels = []
    values = []
    for label, value in stats.items():
        if value > 0:
            labels.append(label)
            values.append(value)
    
    if values:
        ax4.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax4.axis('equal')
        ax4.set_title('Photon Fate Statistics')
    
    # Energy deposition (XZ plane)
    ax5 = fig.add_subplot(gs[1, 1])
    slice_index = energy_grid.shape[1] // 2  # Middle Y slice
    slice_2d = energy_grid[:, slice_index, :]
    mask = slice_2d > epsilon
    
    if np.any(mask):
        norm = LogNorm(vmin=max(slice_2d[mask].min(), epsilon), vmax=slice_2d.max())
        im5 = ax5.imshow(slice_2d.T, origin='lower', cmap='hot', norm=norm)
    else:
        im5 = ax5.imshow(slice_2d.T, origin='lower', cmap='hot')
    
    plt.colorbar(im5, ax=ax5, label='Deposited Energy')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Z')
    ax5.set_title(f'Energy Deposition (Y={slice_index})')
    
    # Additional statistics or histogram
    ax6 = fig.add_subplot(gs[1, 2])
    # Example: Histogram of path lengths
    if paths:
        path_lengths = [len(path) for path in paths]
        ax6.hist(path_lengths, bins=20)
        ax6.set_xlabel('Path Length (steps)')
        ax6.set_ylabel('Count')
        ax6.set_title('Photon Path Length Distribution')
        ax6.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def save_plots(results: Dict[str, Any], output_dir: str = 'results') -> None:
    """
    Save a comprehensive set of plots from simulation results.
    
    Args:
        results: Simulation results dictionary
        output_dir: Directory to save plots
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Photon paths
    fig = plot_photon_paths(results.get('photon_paths', []))
    fig.savefig(os.path.join(output_dir, 'photon_paths.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Energy deposition slices
    energy_grid = results.get('deposited_energy', np.zeros((1, 1, 1)))
    
    # XY plane (mid Z)
    fig = plot_energy_deposition(energy_grid, slice_dim='z')
    fig.savefig(os.path.join(output_dir, 'energy_xy.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # XZ plane (mid Y)
    fig = plot_energy_deposition(energy_grid, slice_dim='y')
    fig.savefig(os.path.join(output_dir, 'energy_xz.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # YZ plane (mid X)
    fig = plot_energy_deposition(energy_grid, slice_dim='x')
    fig.savefig(os.path.join(output_dir, 'energy_yz.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Energy profiles
    for axis, name in enumerate(['x', 'y', 'z']):
        fig = plot_energy_profile(energy_grid, axis=axis)
        fig.savefig(os.path.join(output_dir, 'profile_{}.png'.format(name)), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # 4. Statistics pie chart
    fig = plot_statistics_pie(results)
    fig.savefig(os.path.join(output_dir, 'statistics.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Summary figure
    fig = create_summary_plots(results)
    fig.savefig(os.path.join(output_dir, 'summary.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)