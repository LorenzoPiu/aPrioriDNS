#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:08:12 2024

@author: lorenzo piu
"""

from src.aPrioriDNS.DNS import Field3D
import src.aPrioriDNS as ap

# my_field = Field3D("data/Filter8FavreGauss")

small_field = Field3D("data/Filter12FavreGauss")

small_field.compute_progress_variable()
small_field.ox = 'O2'
small_field.fuel = 'H2'
small_field.compute_mixture_fraction(Y_ox_2=0.233, Y_f_1=0.65*2/(0.65*2+0.35*28), s=2)
small_field.plot_z_midplane('C')
small_field.plot_z_midplane('Z')

small_field.DNS_folder_path = 'data/Lifted_H2_subdomain'
small_field.compute_M()

small_field.compute_z_grad()

small_field.compute_mixing_timescale(mode='sub')
small_field.plot_z_midplane('Tau_m_Sub')

# ap.download()

# if __name__ == "__main__":
#     # # my_field.compute_reaction_rates()
#     # filtered_field = Field3D(my_field.filter_favre(filter_size=12))  
#     filtered_field = Field3D("data/Filter8FavreGauss")
#     # filtered_field.compute_reaction_rates()
#     # filtered_field.compute_strain_rate(save_tensor=False)
#     # filtered_field.compute_residual_kinetic_energy()
#     # filtered_field.compute_residual_dissipation_rate()
#     # filtered_field.compute_chemical_timescale()
#     # filtered_field.compute_mixing_timescale()

#     filtered_field.compute_reaction_rates_batch(parallel=True)

