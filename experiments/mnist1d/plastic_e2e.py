#!/usr/bin/env python3

import argparse
import torch
import pandas as pd
import pickle

from lib.gle.abstract_net import GLEAbstractNet
from lib.gle.layers import GLELinear
from lib.gle.dynamics import GLEDynamics
from data.datasets import get_mnist1d_splits

from .mnist1d_training import mnist1d_run
from .networks import E2ELagMLPNet
from lib.utils import get_loss_and_derivative, get_phi_and_derivative


if __name__ == '__main__':
    # parse parameters from command line which often change
    parser = argparse.ArgumentParser(description='Train an GLE network E2E on the MNIST1D dataset.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--precision-parameters', type=str, default="single", help="Precision for parameters.")
    parser.add_argument('--precision-dynamics', type=str, default="single", help="Precision for dynamic variables.")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate.")
    parser.add_argument('--optimizer-step-interval', type=int, default=1, help="Optimizer step interval.")
    parser.add_argument('--tau_r-scaling', type=float, default=1.0, help="tau_r scaling.")
    parser.add_argument('--scheduler', type=str, default='plateau', help="lr scheduler.")
    args = parser.parse_args()

    params = {
        # NN parameters
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": 100,
        "batch_size_test": 1000,
        "log_interval": 10,
        "checkpoint_interval": 100,
        "input_size": 1,
        "hidden_layers": 6,  # hidden lag layers
        "hidden_fast_size": 17,  # LE units
        "hidden_slow_size": 36,  # LI units
        "phi": 'tanh',
        "output_phi": 'linear',
        "loss_fn": 'ce',
        # LE parameters
        "dt": 0.2,
        "tau": 1.2,
        "beta": 1.0,
        "gamma": 0.0,
        "n_updates": 1,
        "prospective_errors": True,
        "use_cuda": True,
        "tau_r_scaling": args.tau_r_scaling,
        "optimizer_step_interval": args.optimizer_step_interval,
        "scheduler": args.scheduler,
    }

    if args.precision_dynamics == "single":
        params["dtype_dynamics"] = torch.float32
    elif args.precision_dynamics == "half":
        params["dtype_dynamics"] = torch.float16
    else:
        raise NotImplementedError()

    if args.precision_parameters == "single":
        params["dtype_parameters"] = torch.float32
        params["eps"] = 1e-8
    elif args.precision_parameters == "half":
        params["dtype_parameters"] = torch.float16
        params["eps"] = 1e-4
    else:
        raise NotImplementedError()

    print('Using params', params)

    torch.manual_seed(params["seed"])
    print("Using seed: {}".format(params['seed']))

    # supersampling
    sample_length = 72  # original length of MNIST-1D samples in arbitrary units
    params['steps_per_sample'] = int(sample_length / params['dt']) # supersampling with factor 1/dt
    print('Using {} steps per sample.'.format(params['steps_per_sample']))

    # rescale learning rate to sample length
    params['lr'] *= sample_length / params['steps_per_sample']
    print('Using learning rate {}.'.format(params['lr']))

    loss_fn, loss_fn_deriv = get_loss_and_derivative(params['loss_fn'], params['output_phi'])
    print("Using {} loss with {} output nonlinearity.".format(params['loss_fn'], params['output_phi']))

    def wrapper_loss_fn(output, target, *args, **kwargs):
        x = loss_fn(output.to(torch.float32), target, *args, **kwargs)
        return x

    def wrapper_loss_fn_deriv(self, output, target, beta):
        x = loss_fn_deriv(self, output, target, beta)
        # if x.max() > 0.5:
        #     x *= 0.5
        return x

    E2ELagMLPNet.compute_target_error = wrapper_loss_fn_deriv

    model = E2ELagMLPNet(dt=params['dt'], tau=params['tau'],
                         prospective_errors=params['prospective_errors'],
                         n_hidden_layers=params['hidden_layers'],
                         hidden_fast_size=params['hidden_fast_size'],
                         hidden_slow_size=params['hidden_slow_size'],
                         phi=params['phi'], output_phi=params['output_phi'],
                         dtype_parameters=params["dtype_parameters"],
                         dtype_dynamics=params["dtype_dynamics"],
                         tau_r_scaling=params['tau_r_scaling'])

    print(f"Using {model.hidden_layers} hidden layers with {params['hidden_fast_size']} LE and {params['hidden_slow_size']} LI units each.")

    # check for CUDA
    if torch.cuda.is_available() and params['use_cuda']:
        DEVICE = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.cuda()
        print("Using CUDA.")
        print("Device: {}".format(DEVICE))
    else:
        params['use_cuda'] = False
        print("Not using CUDA.")

    from lib.memory import Slicer
    memory = Slicer
    memory.kwargs = {}
    print("Using memory:", memory.__name__)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], eps=params["eps"])

    if params['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)
    elif params['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=2,
                                                               factor=0.5,
                                                               eps=params["eps"])
    else:
        raise NotImplementedError()
    scheduler.min_lr = 1e-5

    torch.manual_seed(params['seed'])  # reset seed for reproducibility

    # returns metrics dictionary
    metrics = mnist1d_run(params, memory, model, wrapper_loss_fn, None,
                          *get_mnist1d_splits(final_seq_length=params['steps_per_sample'], dtype=params["dtype_dynamics"]),
                          optimizer=optimizer, lr_scheduler=scheduler,
                          use_le=True, optimizer_step_interval=params['optimizer_step_interval'])

    print(f"Finished training with final test accuracy: {metrics['test_acc'][-1]:.2f}%")

    # convert metrics dict to pandas DF and dump to pickle
    df = pd.DataFrame.from_dict(metrics)

    fname = f"./results/mnist1d/plastic_e2e_{params['seed']}_{args.precision_parameters}_{args.precision_dynamics}_{params['lr']}_{params['optimizer_step_interval']}_{params['tau_r_scaling']}_{params['scheduler']}_{scheduler.min_lr}_metrics.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
    print(f"Dumped metrics to: {fname}")
