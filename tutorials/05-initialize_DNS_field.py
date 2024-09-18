#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:10:20 2024

@author: lorenzo piu
"""

import os
from aPrioriDNS.DNS import Field3D
from aPrioriDNS import DNS
import json

directory = os.path.join('..','data','Lifted_H2_subdomain') # change this with your path to the data folder
T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    raise ValueError("The path '{T_path}' does not exist in your system. Check to have the correct path to your data folder in the code")

# Blastnet's data contain information about the shape of the field in the info.json file
with open(os.path.join(directory,'info.json'), 'r') as file:
    info = json.load(file)
DNS_shape = info['global']['Nxyz']
    
directory = os.path.join('..','data','Lifted_H2_subdomain') # change this with your path to the data folder
T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    raise ValueError("The path '{T_path}' does not exist in your system. Check to have the correct path to your data folder in the code")

# Blastnet's data contain information about the shape of the field in the info.json file
import json
with open(os.path.join(directory,'info.json'), 'r') as file:
    info = json.load(file)
DNS_shape = info['global']['Nxyz']
    
DNS_field = Field3D(directory)

DNS_field.T

DNS_field.U_Y._3D

filt_YO2 = DNS.filter_3D(DNS_field.YO2._3D, 8)

DNS_field.plot_z_midplane('YH2O2')


