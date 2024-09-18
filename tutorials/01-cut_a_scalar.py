#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:31:52 2024

@author: lorenzo piu
"""
"""
# Cut a 3D scalar

The goal of this exercise is to explain how the class Scalar3D works, in the 
two modes available. Once understood this, using this class should simplify
operations with large 3d arrays.

Make sure to download the data folder from the GitHub repo to be able to do
this tutorial until the end :)

First of all, we need to import the module. Make sure you installed
the package with the command
pip install aPrioriDNS
"""
import aPrioriDNS
from aPrioriDNS.DNS import Scalar3D
"""
now let's define a generic array that we will use as a scalar field """
import numpy as np
shape = [30, 20, 15]
array = np.random.rand(*shape)

scalar = Scalar3D(shape=shape, value=array)
print(f'Initial shape:\n{scalar.shape}')
"""
Cut the field using the mode 'equal'. This is useful when we want to
cut the extrema after filtering operations, or in general after operations 
that have problems in the boundary treatment.
Let's say we want to cut two cells from every boundary:
"""
scalar.cut(n_cut=2, mode='equal')
print(f'Scalar\'s shape after cutting with \'equal\' mode:\n{scalar.shape}')

"""
This time we are going to use the DNS dataset, to use the Scalar3D class
in a more useful way.
If you downloaded the entire folder from github this path should work fine
"""
import os
directory = os.path.join('..','data','Lifted_H2_subdomain') # change this with your path to the data folder
T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path '{T_path}' is correct...")
if not os.path.exists(T_path):
    raise ValueError("The path '{T_path}' does not exist in your system. Check to have the correct path to your data folder in the code")

# Blastnet's data contain information about the shape of the field in the info.json file
import json
with open(os.path.join(directory,'info.json'), 'r') as file:
    info = json.load(file)
DNS_shape = info['global']['Nxyz']
    
# Now we have the shape and the path of the file, we can define the Scalar3D object:
T = Scalar3D(shape=DNS_shape, path=T_path)
# Try to access the value of the temperature field:
T_values = T._3D #this will return a 3d array with the values of Temperature
print("Temperature value in the cell 10,10,10")
print(T_values[10,10,10])

"""
The object is now defined and we can access its values. Nothing special until now.
But let's check why using the Scalar3D object works so well with DNS data.
Now we are going to define a numpy array of the same size as the temperature
field, and we'll compare how much memory it takes in your system:
"""
T_numpy = np.random.rand(*DNS_shape)
import sys
T_size = sys.getsizeof(T)
T_numpy_size = sys.getsizeof(T_numpy)

print(f"\nSize of the numpy array:         {T_numpy_size} bytes"
      f"\nSize of the Scalar3D object:     {T_size} bytes")

"""
Showed the advantage of the object, in the following section the goal is to learn
how to use the method cut to cut this field.
To cut a different amount of cells in every direction x, y and z, we can use
the mode 'xyz'. This mode allows us to cut a different number of cells from
the 3 directions, but the number of cells cut, for example, in the x direction,
will be symmetric in the two directions of the x axis. If we cut 3 cells in
the x direction, the function will remove 3 cells at the beginning and 3 at the end.
"""
T_cut = T.cut(n_cut=[30, 0, 10], mode='xyz')
print(f'Initial shape:                               {T.shape}')
print(f'Scalar\'s shape after cutting with \'xyz\' mode:{T_cut.shape}')





