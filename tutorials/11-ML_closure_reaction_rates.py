#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:50:44 2024

@author: lorenzo piu
"""

import os
from aPrioriDNS.DNS import Field3D

directory = os.path.join('..', '..','data','Lifted_H2_subdomain') # change this with your path to the data folder
T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    raise ValueError("The path '{T_path}' does not exist in your system. Check to have the correct path to your data folder in the code")

# Initialize the DNS field
DNS_field = Field3D(directory)

# Compute the reaction rates
DNS_field.compute_reaction_rates()

# # Compute the strain rate module on the DNS data without saving the tensor components
# DNS_field.compute_strain_rate(save_tensor=False)


# FILTERING
filter_size = 16

# filter DNS field and initialize filtered field
filtered_field = Field3D(DNS_field.filter_favre(filter_size))

# compute the reaction rates on the filtered field
filtered_field.compute_reaction_rates()

# compute the strain rate on the filtered field
filtered_field.compute_strain_rate(save_tensor=True)

# compute the residual dissipation rate with Smagorinsky model
filtered_field.compute_residual_dissipation_rate(mode='Smag')

# compute residual kinetic energy
filtered_field.compute_residual_kinetic_energy()

# compute chemical timescale with SFR, FFR and Chomiak model 
filtered_field.compute_chemical_timescale(mode='SFR')
filtered_field.fuel = 'H2'
filtered_field.ox = 'O2'
filtered_field.compute_chemical_timescale(mode='Ch')
filtered_field.compute_mixing_timescale(mode='Kolmo')

###############################################################################
#                        Machine Learning Closure
###############################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# DATA PROCESSING
shape  = filtered_field.shape
length = shape[0]*shape[1]*shape[2]
T      = filtered_field.T.value.reshape(length,1) # extract the valeue of Temperature from the filtered field and rehape it as a column vector
S      = filtered_field.S_LES.value.reshape(length,1)
Tau_c  = filtered_field.Tau_c_SFR.value.reshape(length,1)

HRR_LFR = filtered_field.HRR_LFR.value.reshape(length,1)
HRR_DNS = filtered_field.HRR_DNS.value.reshape(length,1)

# Data scaling
T      = T-np.min(T)/(np.max(T)-np.min(T))
S      = np.log10(S)
S      = S-np.min(S)/(np.max(S) - np.min(S))
Tau_c  = np.log10(Tau_c)
Tau_c  = Tau_c-np.min(Tau_c) / (np.max(Tau_c)-np.min(Tau_c))

# Build the training vector
X = np.hstack([T, S, Tau_c])

# Divide between train and test data
X_train, X_test, HRR_LFR_train, HRR_LFR_test, HRR_DNS_train, HRR_DNS_test = train_test_split(
    X, HRR_LFR, HRR_DNS, test_size=0.9, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
HRR_LFR_train = torch.tensor(HRR_LFR_train, dtype=torch.float32)
HRR_LFR_test  = torch.tensor(HRR_LFR_test,  dtype=torch.float32)
HRR_DNS_train = torch.tensor(HRR_DNS_train, dtype=torch.float32)
HRR_DNS_test  = torch.tensor(HRR_DNS_test,  dtype=torch.float32)


# NN class definition
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList() # initialize the layers list as an empty list using nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size)) # Add the first input layer. The layer takes as input <input_size> neurons and gets as output <hidden_size> neurons
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size)) # Add hidden layers
        self.layers.append(nn.Linear(hidden_size, output_size)) # add output layer

    def forward(self, x):    # Function to perform forward propagation
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
# NN architecture
input_size = 3
num_layers = 6
hidden_size = 64
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters()) # here we are using the Adam optimizer, to optimize model.parameters, but what is there inside this attribute?

# transfer on GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # GPU available on Mac M2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using a device with cuda
model = model.to(device)
X_train = X_train.to(device)
X_test = X_test.to(device)
HRR_LFR_train = HRR_LFR_train.to(device)
HRR_DNS_train = HRR_DNS_train.to(device)
HRR_LFR_test = HRR_LFR_test.to(device)
HRR_DNS_test = HRR_DNS_test.to(device)

# Move the optimizer's state to the same device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# Lists to store training loss and testing loss
train_loss_list = []
test_loss_list = []

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    output = model(X_train)

    # compute loss on training data
    loss = criterion(output*HRR_LFR_train, HRR_DNS_train)

    # Compute loss on testing data. NOTE: we aren't gonna use the test loss for optimization!!!
    with torch.no_grad():
        output_test = model(X_test)
        loss_test = criterion(output_test*HRR_LFR_test, HRR_DNS_test)

    # Backprop and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_list.append(loss.item()) # Save the losses
    test_loss_list.append(loss_test.item())

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

# Plot training and testing loss
plt.plot(train_loss_list, label='Training Loss')
plt.plot(test_loss_list, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.yscale('log')
plt.legend()
plt.show()

# PLOTTING

with torch.no_grad():
    gamma = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
YO2 = filtered_field.YO2.value.reshape(length,1)
C = (np.max(YO2) - YO2) / (np.max(YO2) - np.min(YO2))

plt.scatter(HRR_DNS, HRR_LFR, s=0.001)
plt.plot(np.array([0,5e9]), np.array([0,5e9]), c='k')
plt.axis('equal')
plt.show

plt.scatter(HRR_DNS, gamma*HRR_LFR, s=0.001)
plt.plot(np.array([0,5e9]), np.array([0,5e9]), c='k')
plt.axis('equal')
plt.show

plt.scatter(T, gamma, s=0.001)
plt.scatter(C, gamma, s=0.001)
plt.scatter(S, gamma, s=0.001)
plt.scatter(Tau_c, gamma, s=0.001)
plt.scatter(T, HRR_DNS/HRR_LFR, s=0.001)

np.sqrt(np.sum((gamma*HRR_LFR - HRR_DNS)**2))
np.sqrt(np.sum((*HRR_LFR - HRR_DNS)**2))
np.sqrt(np.sum((HRR_LFR - HRR_DNS)**2))
