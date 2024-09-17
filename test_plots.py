#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:08:12 2024

@author: lorenzo piu
"""

from src.aPrioriDNS.DNS import Field3D
from src.aPrioriDNS.plot_utilities import cond_mean_plot
from src.aPrioriDNS.plot_utilities import parity_plot
from src.aPrioriDNS.plot_utilities import contour_plot
import src.aPrioriDNS as ap
import numpy as np

my_field = Field3D('data/Filter16FavreGauss')
# ap.download()

n = 100000

x = np.random.randint(0, 100, n)

y_1 = np.random.randn(n)
y_2 = 2+ 0.5*np.random.randn(n)
y_3 = 5+ ((0.007*np.random.randn(n)))*x
y_list = list([y_1, y_3, y_2, y_1, y_3])

cond_mean_plot(x, y_list, num_bins=1000, markers=False, minmax=False, variance=True, x_name=r'$x$ [mm]', remove_x=True)

parity_plot(my_field.HRR_DNS.value, my_field.HRR_LFR.value)

contour_plot(my_field.mesh.X_midZ, my_field.mesh.Y_midZ, my_field.T.z_midplane)

my_field.plot_z_midplane('T')

