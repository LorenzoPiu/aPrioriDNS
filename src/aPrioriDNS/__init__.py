#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:16:15 2024

@author: lorenzo piu
"""

# Import useful functions and classes from the DNS module
from .DNS import (delete_file, 
                 download, 
                 process_file, 
                 filter_gauss, 
                 filter_box, 
                 filter_3D, 
                 save_file, 
                 gradient_x, 
                 gradient_y, 
                 gradient_z,
                 Field3D,
                 Scalar3D,
                 Mesh3D
                 )

# Import plot utilities
from .plot_utilities import (contour_plot,
                             parity_plot,
                             cond_mean_plot,
                             scatter
                             )

# # Default modules to import
# import inspect
# import importlib

# # Function to dynamically import all functions and classes from a given module
# def import_module_contents(module_name):
#     module = importlib.import_module(module_name, package=__name__)
#     for name, member in inspect.getmembers(module):
#         if inspect.isfunction(member) or inspect.isclass(member):
#             globals()[name] = member
#             __all__.append(name)
            
# def import_module_classes(module_name):
#     module = importlib.import_module(module_name, package=__name__)
#     for name, member in inspect.getmembers(module):
#         if inspect.isclass(member):
#             globals()[name] = member
#             __all__.append(name)

# # Initialize __all__ list
# __all__ = []

# # Import all the classes from the module DNS
# import_module_classes('.DNS')

# # Import contents from the module plot_utilities
# import_module_contents('.plot_utilities')
