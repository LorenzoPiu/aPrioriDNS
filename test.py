#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:08:12 2024

@author: lorenzo piu
"""

from src.aPrioriDNS.DNS import Field3D
import src.aPrioriDNS as ap

ap.DNS.add_variable('R_C', 'RC_{}.dat')

my_field = Field3D("data/Lifted_H2_subdomain")
# 

