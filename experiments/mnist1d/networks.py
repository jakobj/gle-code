#!/usr/bin/env python3

import torch
import torch.nn as nn
from lib.gle.layers import GLELinear
from lib.gle.dynamics import GLEDynamics
from lib.gle.abstract_net import GLEAbstractNet
from lib.utils import get_phi_and_derivative


class GLETDNN(GLEAbstractNet, torch.nn.Module):
    def __init__(self, *, input_size=10, hidden_size=100, output_size=10,
                 tau, dt, gamma=1.0, phi=None, output_phi=None):
        super().__init__(full_forward=False)
        assert False

        self.tau = tau
        self.dt = dt

        self.phi, self.phi_prime = get_phi_and_derivative(phi)
        self.output_phi, self.output_phi_prime = get_phi_and_derivative(output_phi)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = GLELinear(self.input_size, self.hidden_size)
        self.fc2 = GLELinear(self.hidden_size, self.hidden_size)
        self.fc3 = GLELinear(self.hidden_size, self.output_size)

        self.fc1_dynamics = GLEDynamics(self.fc1, dt=self.dt, tau_m=self.tau,
                                       gamma=gamma, phi=self.phi,
                                       phi_prime=self.phi_prime)
        self.fc2_dynamics = GLEDynamics(self.fc2, dt=self.dt, tau_m=self.tau,
                                       gamma=gamma, phi=self.phi,
                                       phi_prime=self.phi_prime)
        self.fc3_dynamics = GLEDynamics(self.fc3, dt=self.dt, tau_m=self.tau,
                                       gamma=gamma, phi=self.output_phi,
                                       phi_prime=self.output_phi_prime)

        print("Initialized LE-TDNN model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])


class E2ELagMLPNet(GLEAbstractNet, torch.nn.Module):

    def __init__(self, *, dt, tau,  prospective_errors=False, n_hidden_layers=4,
                 hidden_fast_size=50, hidden_slow_size=50, gamma=0.0,
                 phi='tanh', output_phi='linear', dtype=torch.float32, tau_r_scaling=1.0):
        super().__init__(full_forward=False, full_backward=False, dtype=dtype)

        self.output_size = 10  # still MNIST with 10 classes

        self.dt = dt

        self.phi, self.phi_prime = get_phi_and_derivative(phi)
        self.output_phi, self.output_phi_prime = get_phi_and_derivative(output_phi)

        hidden_size = hidden_fast_size + hidden_slow_size

        # specify taus for LE and LI units
        tau_m = tau * torch.ones(hidden_size, dtype=dtype)
        tau_r = tau * torch.ones(hidden_size, dtype=dtype)
        tau_r[:hidden_slow_size] = tau_r_scaling * self.dt       # τ_r = [dt, dt, τ]
        tau_m[:hidden_slow_size // 2] = tau / 2  # τ_m = [τ/2, τ, τ]
        print("Using tau_m:", tau_m)
        print("Using tau_r:", tau_r)

        # clamp taus to dt and 100
        self.tau_r = tau_r.clamp(self.dt, 100)
        self.tau_m = tau_m.clamp(self.dt, 100)

        # empty lists to store layers and dynamics
        layers = []
        dyns = []

        # input layer
        layer = GLELinear(1, hidden_size, dtype=dtype)
        layers.append(layer)
        dyns.append(GLEDynamics(layer, dt=self.dt, tau_m=self.tau_m,
                                tau_r=self.tau_r,
                                prospective_errors=prospective_errors,
                                dtype=dtype))

        # half lagged, half instantaneous
        for i in range(n_hidden_layers - 1):
            layer = GLELinear(hidden_size, hidden_size, dtype=dtype)
            layers.append(layer)
            dyns.append(GLEDynamics(layer, dt=self.dt, tau_m=self.tau_m,
                                    tau_r=self.tau_r, gamma=gamma, phi=self.phi,
                                    phi_prime=self.phi_prime,
                                    prospective_errors=prospective_errors,
                                    dtype=dtype))

        # instantaneous output layer
        layer = GLELinear(hidden_size, self.output_size, dtype=dtype)
        layers.append(layer)
        dyns.append(GLEDynamics(layer, dt=self.dt,
                                tau_m=torch.ones(self.output_size, dtype=dtype) * tau,
                                tau_r=torch.ones(self.output_size, dtype=dtype) * tau,
                                gamma=gamma, phi=self.output_phi,
                                phi_prime=self.output_phi_prime,
                                prospective_errors=prospective_errors,
                                dtype=dtype))

        # turn all variables in layers into attributes of the model
        for i, layer in enumerate(layers):
            setattr(self, f'layer_{i}', layer)
        self.layers = layers

        # turn all variables in dyns into attributes of the model
        for i, dyn in enumerate(dyns):
            setattr(self, f'dyn_{i}', dyn)
        self.dyns = dyns

        self.hidden_layers = n_hidden_layers
        self.dtype = dtype

        print("Initialized {} model with {} parameters".format(self.__class__.__name__, self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])


# static lag net with membrane lags
class GLELagNet(GLEAbstractNet, torch.nn.Module):
    def __init__(self, *, dt, tau, identity_lag=True, prospective_errors=False):
        super().__init__(full_forward=False)
        assert False

        self.fc1 = GLELinear(1, 10)
        self.fc2 = GLELinear(10, 10)
        self.fc3 = GLELinear(10, 10)
        self.fc4 = GLELinear(10, 10)

        self.tau = tau
        self.dt = dt

        self.phi = lambda x: x
        self.phi_prime = lambda x: torch.ones_like(x)

        # linearly distributed time constants
        self.tau_m = torch.round(torch.linspace(self.dt, tau, 10), decimals=2)
        self.tau_r = torch.ones_like(self.tau_m) * self.dt

        self.fc1_dynamics = GLEDynamics(self.fc1, tau_m=self.tau_m,
                                       tau_r=self.tau_r, dt=self.dt,
                                       prospective_errors=prospective_errors)
        self.fc2_dynamics = GLEDynamics(self.fc2, tau_m=self.tau_m,
                                       tau_r=self.tau_r, dt=self.dt,
                                       prospective_errors=prospective_errors)
        self.fc3_dynamics = GLEDynamics(self.fc3, tau_m=self.tau_m,
                                       tau_r=self.tau_r, dt=self.dt,
                                       prospective_errors=prospective_errors)
        self.fc4_dynamics = GLEDynamics(self.fc4, tau_m=self.tau_m,
                                       tau_r=self.tau_r, dt=self.dt,
                                       prospective_errors=prospective_errors)

        # turn off learning for lag part
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False
        self.fc3.weight.requires_grad = False
        self.fc3.bias.requires_grad = False
        self.fc4.weight.requires_grad = False
        self.fc4.bias.requires_grad = False

        # set lag part to identity
        if identity_lag:
            self.identity_lag_net()

    def identity_lag_net(self):
        # set lag part to identity
        self.fc1.weight.data = torch.ones_like(self.fc1.weight.data)
        self.fc1.bias.data = torch.zeros_like(self.fc1.bias.data)
        self.fc2.weight.data = torch.eye(self.fc2.weight.data.shape[0])
        self.fc2.bias.data = torch.zeros_like(self.fc2.bias.data)
        self.fc3.weight.data = torch.eye(self.fc3.weight.data.shape[0])
        self.fc3.bias.data = torch.zeros_like(self.fc3.bias.data)
        self.fc4.weight.data = torch.eye(self.fc3.weight.data.shape[0])
        self.fc4.bias.data = torch.zeros_like(self.fc3.bias.data)

class GLELagWindow:
    def __init__(self, data, dt, tau):
        self.data = data
        self.input_idx = 0
        self.network = GLELagNet(dt=dt, tau=tau)

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        if self.input_idx >= len(self):
            raise StopIteration

        slice = self.data[:, self.input_idx: self.input_idx + 1]
        output = self.network(slice)

        self.input_idx += 1
        return output
