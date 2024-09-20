#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:50:44 2024

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
    
directory = os.path.join('..','data','Lifted_H2_subdomain') # change this with your path to the data folder
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    raise ValueError("The path '{T_path}' does not exist in your system. Check to have the correct path to your data folder in the code")
    
DNS_field = Field3D(directory)

filter_size = 12

field_filt_name = DNS_field.filter_favre(filter_size, filter_type='Gauss')

DNS_field_filt_gauss = Field3D(field_filt_name)

DNS_field_filt_box = Field3D(DNS_field.filter_favre(filter_size, filter_type='Box'))
