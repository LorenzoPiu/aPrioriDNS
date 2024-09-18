#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:17:23 2024

@author: Lorenzo Piu
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import gaussian_kde

from ._styles import ParityPlot, ContourPlot, CondMeanPlot, ScatterPlot
from ._utils import extract_random_elements, reorder_arrays, process_arrays

plt.rcParams.update({
    'font.size': 18,           # Default font size
    'figure.figsize': (10, 6), # Default figure size (width, height) in inches
    # 'axes.titlesize': 18,      # Title font size
    # 'axes.labelsize': 14,      # Axis label font size
    # 'xtick.labelsize': 12,     # X-axis tick label font size
    # 'ytick.labelsize': 12,     # Y-axis tick label font size
    # 'legend.fontsize': 12,     # Legend font size
    'lines.linewidth': 2,      # Line width
    # 'lines.markersize': 6,     # Marker size
    'axes.linewidth': 1.5,     # Axes border width
    'grid.linewidth': 1,       # Grid line width
    # 'xtick.major.width': 1,    # Major tick width on x-axis
    # 'ytick.major.width': 1,    # Major tick width on y-axis
    # 'xtick.minor.width': 0.8,  # Minor tick width on x-axis
    # 'ytick.minor.width': 0.8   # Minor tick width on y-axis
})

def parity_plot(x, 
                y, 
                rel_error=0.2, 
                colormap='viridis', 
                x_name='x',
                y_name='y',
                title=None,
                density=False, 
                c=None, 
                linewidth=3,
                vmin=None,
                vmax=None,
                cbar_title=None,
                marker='.',
                s=1, 
                alpha=1,
                R2=True,
                ticks=None,
                limits=None,
                save=False,
                show=True,
                max_dim=100000,
                remove_cbar=False,
                remove_x=False,
                remove_y=False
                ):
    
    # Flatten the vectors because they will be stack afterwards
    x = x.flatten()
    y = y.flatten()
    
    # Find the limits of the domain
    min_x = np.minimum(np.min(x), np.min(y))
    max_x = np.maximum(np.max(x), np.max(y))
    
    if limits is None:
        limits=[min_x, max_x]
        
    # Compute the R2 score if required:
    if R2:
        R2 = 1 - np.sum((x-y)**2) / np.sum((x-np.average(x))**2)
    
    if density:
        if len(x) > max_dim:
            print(f"\nParity Plot:\nThe vector size is too large to compute"
                  f" the Gaussian kde. A random sample of {max_dim} elements "
                  "will be extracted from the vector for the plot. You can "
                  "change this limit with the parameter max_dim")
            x = extract_random_elements(x, max_dim, seed=42)
            y = extract_random_elements(y, max_dim, seed=42)
        # Perform KDE
        x = np.reshape(x, [1, len(x)])
        y = np.reshape(y, [1, len(y)])
        
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        z = z/max(z)
        
        # Sort the points by density, so densest points are plotted last
        x = x.flatten()
        y = y.flatten()
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        
        c = z
        cbar_title='Density'

    # PLOT
    fig = plt.figure(figsize=ParityPlot.fig_size, 
                     dpi=ParityPlot.dpi)
    plt.fill_between(limits, 
                     [min_x, max_x*(1-rel_error),], 
                     [min_x, max_x*(1+rel_error)], 
                     color='grey', alpha=0.5)
    
    if c is None:
        c='blue'
    
    plt.scatter(x, y, marker=marker, c=c, s=s, 
                cmap=colormap, alpha=alpha, vmin=vmin, vmax=vmax)
    
    if (not isinstance(c, str)) and (c is not None): # if c is a string means it was initialized as a single color, so no need for colormap
        cbar = plt.colorbar(shrink=.9, aspect=15, fraction=.1,pad=.05)
        cbar.ax.tick_params(labelsize=ParityPlot.fontsize*3//4)
        if cbar_title is not None:
            cbar.set_label(cbar_title, rotation=90, fontsize=ParityPlot.fontsize)
    plt.xlabel(x_name, fontsize=ParityPlot.fontsize)
    plt.ylabel(y_name, fontsize=ParityPlot.fontsize)
    if title is not None:
        plt.title(title)
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xticks(fontsize=ParityPlot.fontsize*3//4)
    plt.yticks(fontsize=ParityPlot.fontsize*3//4)
    if ticks is not None:
        if isinstance(ticks, int):
            xmin    = plt.gca().get_xlim()[0]
            xmax    = plt.gca().get_xlim()[1]
            ticks = np.linspace(xmin, xmax, ticks)
        plt.xticks(ticks=ticks)
        plt.yticks(ticks=ticks)
    plt.plot(limits, limits, 'k--', linewidth=linewidth)
    if R2:
        plt.text(0.02, 0.92, rf'$R^2 = {R2:.4f}$', 
                 transform=plt.gca().transAxes, 
                 fontsize=ParityPlot().fontsize//2)
        
    # Adjust borders tickness
    for spine in plt.gca().spines.values():
        spine.set_linewidth(ParityPlot.border_width )
    
    plt.gca().set_aspect('equal', adjustable='box')
    if remove_cbar:
        cbar.remove()
    figure = plt.gcf()
    if remove_x:
        plt.xlabel('')
        plt.xticks([])
    if remove_y:
        plt.xlabel('')
        plt.xticks([])
    
    if save:
        if title is None:
            title=f'Parity_plot_{x_name}_{y_name}'
        plt.savefig(title, dpi=ParityPlot.dpi, bbox_inches = "tight")
    
    if show:
        plt.show()
        
    # Close the figure to prevent it from displaying when returned
    plt.close(figure)
    
    return figure
    

def contour_plot(X, 
                 Y, 
                 Z,
                 log=False,
                 colormap='viridis', 
                 cbar_title=None,
                 cbar_shrink=0.7,
                 levels=None,
                 color='black',
                 labels=False,
                 linestyle='-',
                 linecolor='black',
                 linewidth=1,
                 x_ticks=None,
                 y_ticks=None,
                 x_lim=None,
                 y_lim=None,
                 vmin=None,
                 vmax=None,
                 transparent=True,
                 title=None,
                 x_name='x',
                 y_name='y',
                 remove_cbar=False,
                 remove_x=False,
                 remove_y=False,
                 remove_title=False,
                 transpose=False,
                 save=False,
                 show=True
                 ):
    
    Z = Z.copy() # so it won't be modified inside the funciton
    X = X.copy()
    Y = Y.copy()
    if log:
        Z = np.log10(Z)
    
    if transpose:
        temp = X
        X = Y
        Y = temp
    
    ratio = np.max(X)-np.min(X)/X.shape[1]
    figsize = [ContourPlot.fig_size, ratio*ContourPlot.fig_size]
    fontsize=ContourPlot.fontsize
    
    # Create the contour plot
    plt.figure(dpi=ContourPlot.dpi)
    # if transparent, plot the points and then cut the vectors
    if transparent:
        # plt.pcolormesh(X, Y, Z, color='white')
        Z_full = Z.copy()
        if (vmin is not None) and (vmax is not None):
            Z[(Z<vmin) | (Z>vmax)] = None
        else:
            if (vmin is not None):
                if (np.min(Z) < vmin):
                    Z[Z<vmin] = None
            if (vmax is not None):
                if (np.max(Z) > vmax):
                    Z[Z>vmax] = None
        
    # Plot filled contours
    contourf = plt.pcolormesh(X, Y, Z, cmap=colormap, vmin=vmin, vmax=vmax)  # Filled contours
    
    if levels is not None:
        # Plot iso-lines with all lines in black
        contour = plt.contour (X, Y, Z_full, 
                              levels=levels, 
                              colors=linecolor, 
                              linestyles=linestyle, 
                              linewidths=linewidth)  # All lines in black
    
    # Add a color bar to the plot
    cbar = plt.colorbar(contourf, shrink=cbar_shrink, aspect=15, fraction=.1,pad=.05)
    cbar.ax.tick_params(labelsize=ContourPlot.fontsize*3//4)
    cbar.ax.tick_params(labelsize=ContourPlot.fontsize*3//4)
    if cbar_title is not None:
        cbar.set_label(cbar_title, rotation=270, fontsize=ContourPlot.fontsize )
    
    # Add labels and title
    plt.xlabel(x_name, fontsize=fontsize)
    plt.ylabel(y_name, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    
    # x and y limits
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    
    # Handle ticks
    if x_ticks is not None:
        if isinstance(x_ticks, int):
            xmin    = plt.gca().get_xlim()[0]
            xmax    = plt.gca().get_xlim()[1]
            x_ticks = np.linspace(xmin, xmax, x_ticks)
        plt.xticks(ticks=x_ticks)
    if y_ticks is not None:
        if isinstance(y_ticks, int):
            ymin    = plt.gca().get_ylim()[0]
            ymax    = plt.gca().get_ylim()[1]
            y_ticks = np.linspace(ymin, ymax, y_ticks)
        plt.yticks(ticks=y_ticks)
    plt.xticks(fontsize=fontsize*3//4)
    plt.yticks(fontsize=fontsize*3//4)
    if labels:
        # Optionally, add labels to contour lines
        plt.clabel(contour, inline=True, fontsize=fontsize//4, fmt='%1.1f')
        
    plt.gca().set_aspect('equal', adjustable='box')
    if remove_cbar:
        cbar.remove()
    figure = plt.gcf()
    if remove_x:
        plt.xlabel('')
        plt.xticks([])
    if remove_y:
        plt.xlabel('')
        plt.xticks([])
    if remove_title:
        plt.title('')
    
    figure = plt.gcf()
    
    if save:
        if title is None:
            title=f'Contour_plot'
            if cbar_title is not None:
                title = title+'_'+cbar_title
        else:
            title = 'Scatter_plot' + title
        plt.savefig(title, dpi=ContourPlot.dpi, bbox_inches = "tight")
    
    if show:
        plt.show()
        
    # Close the figure to prevent it from displaying when returned
    plt.close(figure)
    
    return figure

def cond_mean_plot(x, y, num_bins=30, 
                   log=False, 
                   minmax=True,
                   variance=False,
                   background=False,
                   background1=False,
                   colors=None,
                   markers=None,
                   x_name=None,
                   y_name=None,
                   title=None,
                   x_ticks=None,
                   y_ticks=None,
                   remove_markers=True,
                   remove_x=False,
                   remove_y=False,
                   save=False,
                   show=True
                   ):
    """
    Plots the conditional mean of y over x using numpy.

    Parameters:
    x (array-like): Independent variable.
    y (array-like): Dependent variable.
    num_bins (int): Number of bins to divide x into.
    """
    
    # check that x and y have the same dimension.
    # if y contains more than 1 vector, extracts m, the number of vectors it contains.
    # it also makes y a list, so that it is iterable
    x, y, m = process_arrays(x, y)
    
    # style
    lines = None
    if colors is None:
        colors = generate_colors(m)
    if markers is None:
        markers = generate_markers(m)
    elif markers is False:
        markers = ['' for i in range(m)]
        lines   = generate_lines(m)
    if lines is None:
        lines = ['-' for i in range(m)]
        
    fontsize = CondMeanPlot.fontsize
    dpi      = CondMeanPlot.dpi
    fig_size = CondMeanPlot.fig_size
    
    # initialize figure
    plt.figure(figsize=fig_size, dpi=dpi)
    
    # iterate for the number of y
    for i in range(m):
        x_b,y_b,y_max,y_min,y_var = bins(x, y[i], n=num_bins, log=log)
        color = colors[i]
        # Plotting
        plt.plot(x_b, y_b, marker=markers[i], linestyle=lines[i], c=color)
        if minmax:
            plt.fill_between(x_b, y_max, y_min, alpha=0.45, color=color)
        if variance:
            plt.fill_between(x_b, y_b+y_var, y_b-y_var, alpha=0.55, color=color)
        if background:
            plt.scatter(x, y[i], marker='.', alpha=0.3, c=color)
        if background1 and i==0:
            plt.scatter(x, y[i], marker='.', alpha=0.6, c='#c2c2c2') #c='#d3d3d3')
    

    plt.grid(True, linestyle='--', alpha=0.7)
    # Add labels
    plt.xlabel(x_name, fontsize=fontsize)
    plt.ylabel(y_name, fontsize=fontsize)
    
    # Adjust borders tickness
    for spine in plt.gca().spines.values():
        spine.set_linewidth(ParityPlot.border_width )
    
    # Handle ticks
    if x_ticks is not None:
        if isinstance(x_ticks, int):
            xmin    = plt.gca().get_xlim()[0]
            xmax    = plt.gca().get_xlim()[1]
            x_ticks = np.linspace(xmin, xmax, x_ticks)
        plt.xticks(ticks=x_ticks)
    if y_ticks is not None:
        if isinstance(y_ticks, int):
            ymin    = plt.gca().get_ylim()[0]
            ymax    = plt.gca().get_ylim()[1]
            y_ticks = np.linspace(ymin, ymax, y_ticks)
        plt.yticks(ticks=y_ticks)
    plt.xticks(fontsize=fontsize*3//4)
    plt.yticks(fontsize=fontsize*3//4)

    figure = plt.gcf()
    if remove_x:
        plt.xlabel('')
        plt.xticks([])
    if remove_y:
        plt.xlabel('')
        plt.xticks([])
    
    figure = plt.gcf()
    
    if save:
        if title is None:
            title=f'Contour_plot'
        else:
            title = 'Conditional_mean_' + title
        plt.savefig(title, dpi=dpi, bbox_inches = "tight")
    
    if show:
        plt.show()
        
    # Close the figure to prevent it from displaying when returned
    plt.close(figure)
    
    return figure

def scatter(x, y,  logx=False, 
                   logy=False,
                   c=None,
                   s=1,
                   alpha=1,
                   max_dim=2000000,
                   colormap='viridis',
                   cbar_title=None,
                   vmin=None,
                   vmax=None,
                   marker='.',
                   avg=True,
                   avg_label=None,
                   num_bins=100, 
                   linestyle='--',
                   linewidth=2,
                   linecolor='b',
                   Z_st=None,
                   x_name=None,
                   y_name=None,
                   title=None,
                   x_ticks=None,
                   y_ticks=None,
                   remove_markers=True,
                   remove_x=False,
                   remove_y=False,
                   save=False,
                   show=True
                   ):
    
    # style
    fontsize = ScatterPlot.fontsize
    dpi      = ScatterPlot.dpi
    fig_size = ScatterPlot.fig_size
    
    if (c is not None) and (len(x) > max_dim):
        print(f"\nScatter Plot:\nThe vector size is too large to display"
              f" the scatter plot. A random sample of {max_dim} elements "
              "will be extracted from the vector for the plot. You can "
              "change this limit with the parameter max_dim")
        x = extract_random_elements(x, max_dim, seed=42)
        y = extract_random_elements(y, max_dim, seed=42)
        c = extract_random_elements(c, max_dim, seed=42)
    
    # initialize figure
    plt.figure(figsize=fig_size, dpi=dpi)
    
    plt.scatter(x, y, s=s, c=c, marker=marker, alpha=alpha)
    if (not isinstance(c, str)) and (c is not None): # if c is a string means it was initialized as a single color, so no need for colormap
        cbar = plt.colorbar(shrink=.9, aspect=15, fraction=.1,pad=.05)
        cbar.ax.tick_params(labelsize=fontsize*3//4)
        if cbar_title is not None:
            cbar.set_label(cbar_title, rotation=90, fontsize=ParityPlot.fontsize)

    if avg is True:
        x_b,y_b,y_max,y_min,y_var = bins(x, y, n=num_bins, log=logx)
        plt.plot(x_b, y_b, linewidth=linewidth, color=linecolor, linestyle=linestyle, label=avg_label)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels
    plt.xlabel(x_name, fontsize=fontsize)
    plt.ylabel(y_name, fontsize=fontsize)
    
    # Handle ticks
    if x_ticks is not None:
        if isinstance(x_ticks, int):
            xmin    = plt.gca().get_xlim()[0]
            xmax    = plt.gca().get_xlim()[1]
            x_ticks = np.linspace(xmin, xmax, x_ticks)
        plt.xticks(ticks=x_ticks)
    if y_ticks is not None:
        if isinstance(y_ticks, int):
            ymin    = plt.gca().get_ylim()[0]
            ymax    = plt.gca().get_ylim()[1]
            y_ticks = np.linspace(ymin, ymax, y_ticks)
        plt.yticks(ticks=y_ticks)
        
    if logx is True:
        plt.xscale('log')
    if logy is True:
        plt.yscale('log')
        
    plt.xticks(fontsize=fontsize*3//4)
    plt.yticks(fontsize=fontsize*3//4)
    
    if Z_st is not None:
        y_min, y_max = plt.gca().get_ylim()
        plt.vlines(Z_st, y_min, y_max, color='k', linestyle='-.', linewidth=2, label=r'$Z_{st}$')
        plt.ylim([y_min, y_max])
        plt.legend()
    
    # Adjust borders tickness
    for spine in plt.gca().spines.values():
        spine.set_linewidth(ParityPlot.border_width )

    figure = plt.gcf()
    if remove_x:
        plt.xlabel('')
        plt.xticks([])
    if remove_y:
        plt.xlabel('')
        plt.xticks([])
    
    figure = plt.gcf()
    
    if save:
        if title is None:
            title=f'Contour_plot'
        else:
            title = 'Conditional_mean_' + title
        plt.savefig(title, dpi=dpi, bbox_inches = "tight")
    
    if show:
        plt.show()
        
    # Close the figure to prevent it from displaying when returned
    plt.close(figure)
    
    return figure

def bins(x, y, n=40, log=False):
    """
    Bins the data in x and y into n bins and computes the average, max, and min of y for each bin.
    
    Args:
    -----
        x (array-like): 
            Independent variable.
        y (array-like): 
            Dependent variable.
        n (int, optional): 
            Number of bins. Default is 40.
        log (bool, optional): 
            If True, use logarithmic bins. Default is False.
    
    Returns:
    --------
        tuple
            - x_b (np.ndarray): Midpoints of the bins in x.
            - y_b (np.ndarray): Average values of y in each bin.
            - y_max (np.ndarray): Maximum values of y in each bin.
            - y_min (np.ndarray): Minimum values of y in each bin.
    """
    x = np.copy(np.asarray(x))
    y = np.copy(np.asarray(y))
    
    x, y = reorder_arrays(x, y)
    
    # Create bins
    if log:
        bins = np.logspace(np.min(x), np.max(x), n + 1)
    else:
        bins = np.linspace(np.min(x), np.max(x), n + 1)
        
    x_b   = []
    y_b   = []
    y_max = []
    y_min = []
    y_var = []
    for i in range(n):
        # check if we have points in the interval
        if np.any((x >= bins[i]) & (x <= bins[i+1])):
            y_bin_i = y[(x >= bins[i]) & (x <= bins[i+1])]
            x_b.append((bins[i+1]+bins[i])/2)
            y_b.append   (np.average(y_bin_i))
            y_max.append (np.max    (y_bin_i))
            y_min.append (np.min    (y_bin_i))
            y_var.append (np.var    (y_bin_i))
    
    x_b   = np.array(x_b)
    y_b   = np.array(y_b)
    y_max = np.array(y_max)
    y_min = np.array(y_min)
        
    return x_b, y_b, y_max, y_min, y_var

def generate_colors(num_colors, cmap='plasma'):
    """
    Generates a list of distinct colors.

    Args:
        num_colors (int): The number of colors to generate.

    Returns:
        list: A list of colors in hexadecimal format.
    """
    # Generate a colormap
    cmap = plt.get_cmap(cmap)
    
    # Generate colors using the colormap
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    
    # Convert RGBA to hex
    hex_colors = [matplotlib.colors.to_hex(color) for color in colors]
    
    return hex_colors

def generate_markers(n):
    all_markers = list(['o', 's', 'D', '^', '>', '<', 'd', 'p', 'H', '*'])
    return all_markers[:n]

def generate_lines(n):
    linestyle_tuple = [
     ('solid',                 '-'),
     ('dashed',                '--'),
     ('dashdotted',            '-.'),
     ('dotted',                (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    
    lines = [linestyle_tuple[i][1] for i in range(n)]
    return lines