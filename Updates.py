#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:44:20 2024

@author: lorenzo piu

UPDATES:
    - updated the cut function
    
    - updated the computation of the HRR in the batch reaction rates
    
    - corrected a bug in the chemical timescale computation (SFR mode)
    
    - Scalar3D class object is now more versatile, with dunder methods inherited from numpy
    
    - improved memory management in the computation of chemical source terms
    
    - added possibility to compute source terms in parallel
"""

