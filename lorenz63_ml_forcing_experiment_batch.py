"""
python3

TODO: maybe include attractor-reconstruction selection criterion in network training

@author: Sebastian Scher, march 2019
"""

import os
import sys
import matplotlib
matplotlib.use('agg')
import pandas as pd
import seaborn as sns
import numpy as np

from tqdm import tqdm, trange
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras import backend as K
import tensorflow as tf
import keras



name='sigma_forcing' # for plots


def lorenz(X, t,sigma):
    """
    lorenz ode system
    :param X: [x,y,z]
    :return: dXdt
    we must have a t argument for odeint
    """
    beta = 8. / 3.
    rho = 28
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    dXdt = [dxdt, dydt, dzdt]
    return dXdt


def make_lorenz_run(y0, Nsteps, sigma_arr):
    res = np.zeros((Nsteps+1,4))
    res[0] = np.array([*y0,sigma_arr[0]])
    sol = y0
    for i in range(Nsteps):
        sigma = sigma_arr[i]
        # print(i)
        # we make only one step, but we gert a 2d array back. 9with only one element)
        # therefore, we extract this element with [0]
        sol = odeint(lorenz, sol, t=[0,tstep], args=(sigma,))[1]
        res[i] = np.array([*sol, sigma])
    return res



def make_network_climrun_with_F(network, x_init, Nsteps, F):
    # crreate a "climate run" with the network
    # to keep the code as similar as possible as the one for lorenz95, we use the name
    # "F" instead of "sigma"
    network_clim_run = np.zeros((Nsteps + 1, 3))
    network_clim_run[0] = x_init
    state = x_init
    # add empty time dimension
    state = state[np.newaxis,:]
    for i in range(Nsteps):
        # add F to input. forthis we need to make F a2darray (with empty dimensions)
        state = np.concatenate([state, np.array([[F]])], axis=1)
        state = network.predict(state)
        network_clim_run[i + 1] = state.squeeze()

    return network_clim_run


def make_network_climrun_no_F(network, x_init, Nsteps):
    # crreate a "climate run" with the network
    network_clim_run = np.zeros((Nsteps + 1, 3))
    network_clim_run[0] = x_init
    state = x_init
    # add empty time dimension
    state = state[np.newaxis,:]
    for i in range(Nsteps):
        state = network.predict(state)
        network_clim_run[i + 1] = state.squeeze()

    return network_clim_run


plotdir='plots_lorenz63_forcing'
os.system(f'mkdir {plotdir}')


# paramters for experiments

Nsteps_train = int(sys.argv[1])
Nsteps_test = 10000

# F stands for sigma here
F_start = int(sys.argv[2])
F_end = int(sys.argv[3])
F_test_points = [4, 5, 6, 7, 8, 8.5, 9, 10, 12, 14]

N_training = 10 # how often ot repeat the experiment

# fixed parameters
tstep=0.01
t_arr = np.arange(0, Nsteps_train) * tstep
n_epoch = 30
lead_time = 1  # in steps
hidden_size=128
n_hidden = 2
lr = 0.001
N = 3 # number of variables (alwayys 3 vor lorenz63)
Nsteps_clim = Nsteps_test // lead_time

param_string = '_'.join([str(e) for e in (Nsteps_train, Nsteps_test, n_epoch, lead_time,
                                          F_start, F_end,  name, N_training
                                          )])

F_arr = F_start * np.ones(len(t_arr)) + (F_end - F_start) / Nsteps_train / tstep * t_arr


# we make two runs, started with slightly different initial conditions
# one will be the training and one the test run
x_init1 = [8, 1, 1]
x_init1 = [5, 1, 1]

modelrun_train = make_lorenz_run(x_init1, Nsteps_train, F_arr)
# this now also conatins the varying sigma param as 4th variable
norm_mean = modelrun_train.mean(axis=0)
norm_std = modelrun_train.std(axis=0)

F_norm_mean = norm_mean[3]
F_norm_std = norm_std[3]
modelrun_train = (modelrun_train  - norm_mean) / norm_std




def train_network(X_train, y_train):
    """
        for cimplicity, we train only a single network optimized on short-term performance
    """


    # build and train network
    layers = []
    for _ in range(n_hidden):
        layers.append(keras.layers.Dense(hidden_size, input_shape=(4,), activation='relu'))

    layers.append(keras.layers.Dense(3, activation='linear'))
    network = keras.Sequential(layers)

    optimizer = keras.optimizers.Adam(lr=lr)
    network.compile(optimizer=optimizer, loss='mean_squared_error')

    network.fit(X_train, y_train, epochs=n_epoch, verbose=0, validation_split=0.1,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         min_delta=0,
                                                         patience=4,

                                                         verbose=0, mode='auto')]
                )
    return network


def train_network_noforcing(X_train, y_train):
    """
        for cimplicity, we train only a single network optimized on short-term performance
    """


    # build and train network
    layers = []
    for _ in range(n_hidden):
        layers.append(keras.layers.Dense(hidden_size, input_shape=(3,), activation='relu'))

    layers.append(keras.layers.Dense(3, activation='linear'))
    network = keras.Sequential(layers)

    optimizer = keras.optimizers.Adam(lr=lr)
    network.compile(optimizer=optimizer, loss='mean_squared_error')

    network.fit(X_train, y_train, epochs=n_epoch, verbose=0, validation_split=0.1,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         min_delta=0,
                                                         patience=4,

                                                         verbose=0, mode='auto')]
                )
    return network


# X and y data are the same, but shifted by lead-time,
# we call this _main, because it is the main run with varying F
X_train_main = modelrun_train[:-lead_time]
y_train_main = modelrun_train[lead_time:,:3]  # 4th variable is sigma whcih we dont need in the y data

# this is now the model run for training. For testing, we make additional runs with different fixed F values

runs_fixedF_train = []
runs_fixedF_test = []
for F in F_test_points:
    run_train = make_lorenz_run(x_init1, Nsteps_train, sigma_arr= np.ones(Nsteps_train)*F)
    run_test = make_lorenz_run(x_init1, Nsteps_test, sigma_arr= np.ones(Nsteps_test)*F)
    # normalize with same weights as used for training
    run_train_norm = (run_train - norm_mean) / norm_std
    run_test_norm = (run_test - norm_mean) / norm_std
    X_train = run_train_norm[:-lead_time]
    y_train = run_train_norm[lead_time:, :3]
    X_test = run_test_norm[:-lead_time]
    y_test = run_test_norm[lead_time:,:3]


    runs_fixedF_train.append((X_train, y_train))
    runs_fixedF_test.append((X_test, y_test))





# now loop over relizations (we do the same training a couple of times)
# in each loop, we train the main network (on the run with changing F), and then we test in
# on the runs with fixed F. additionally, for every fixed F, we train a network without F as input, and
# test it on the fixed F run
res_training_loop = []
res_clim = []
for i_training in trange(N_training):

    # train networks
    # X_train_main contains F as second channel. For the network with F we need this:
    main_network_with_F = train_network(X_train_main, y_train_main)
    # for the network without F, we dont need F, so only pass in the first 3 vars
    main_network_no_F = train_network_noforcing(X_train_main[:,:3], y_train_main)

    # loop over test sets for different F and evaluate the network
    for i_F in trange(len(F_test_points)):
        X_test, y_test = runs_fixedF_test[i_F]
        X_train, y_train = runs_fixedF_train[i_F]
        F = F_test_points[i_F]
        # for the neural networks, we need normalized F
        F_normed = (F - F_norm_mean) / F_norm_std

        # make predictions with main network, and compute MAE
        preds_with_F = main_network_with_F.predict(X_test).squeeze()
        err_with_F = np.abs(preds_with_F - y_test).mean()
        preds_no_F = main_network_no_F.predict(X_test[:,:3]).squeeze()
        err_no_F = np.abs(preds_no_F - y_test).mean()

        # now train a network on the fixed F run
        network_fixed = train_network_noforcing(X_train[:,:3], y_train)
        preds_fixed = network_fixed.predict(X_test[:,:3]).squeeze()
        err_fixed = np.abs(preds_fixed - y_test).mean()

        # store in dataframe
        res_training_loop.append(pd.DataFrame({'mae':err_with_F, 'mae_noF':err_no_F,'mae_ref':err_fixed,'F':F,
                                               'n_training':i_training}, index=[0]))

        # make network climate runs and compute climate statistics
        # we start the climate runs with a state from X_test, without the f channel
        network_with_F_clim = make_network_climrun_with_F(main_network_with_F,X_test[100,:3],Nsteps_clim, F_normed )
        network_no_F_clim = make_network_climrun_no_F(main_network_no_F,X_test[100,:3],Nsteps_clim)
        network_ref_clim = make_network_climrun_no_F(network_fixed,X_test[100,:3],Nsteps_clim)
        # the network climate has a timestep of lead_time, so we have to reduce the test run as well
        # to get the same time frequency
        model_reduced = X_test[::lead_time]
        # remove channel with F
        model_reduced = model_reduced[:,:3]

        # now compute std and mean of the clims and store in dataframe.

        df = dict(model_mean = model_reduced.mean(),
                  model_std = model_reduced.std(axis=0).mean(),
                  main_net_with_F_mean = network_with_F_clim.mean(),
                  main_net_with_F_std = network_with_F_clim.std(axis=0).mean(),
                  main_net_no_F_mean = network_no_F_clim.mean(),
                  main_net_no_F_std = network_no_F_clim.std(axis=0).mean(),
                  ref_net_mean = network_ref_clim.mean(),
                  ref_net_std = network_ref_clim.std(axis=0).mean(),
                  n_training=i_training,
                  F=F)

        res_clim.append(pd.DataFrame(df, index=[0]))


# combine into single dataframe and save
res_training_loop = pd.concat(res_training_loop)
res_training_loop.to_pickle(f'res_experiment2_{param_string}.pkl')
res_clim = pd.concat(res_clim)
res_clim.to_pickle(f'res_clim_experiment2_{param_string}.pkl')

# ----------------------- plots------------------------

# rename variables to get a legend that is easier to comprehend
res_training_loop = res_training_loop.rename(columns={'mae':'main net with F',
            'mae_noF':'main net no F',
           'mae_ref':'ref net'})

sns.set_context('notebook', font_scale=1.1)
sns.set_palette('colorblind')


plt.figure(figsize=(8,4))
sns.lineplot('F','MAE', hue='type', estimator=None, units='n_training',
             data=res_training_loop.melt(id_vars=('F', 'n_training'), value_name='MAE', var_name='type'))
sns.despine()
# mark the forcing regoin used in the main training
plt.fill_between([F_start, F_end], y1=[0, 0], y2=[plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]],
                 alpha=0.3, color='grey', zorder=10)
plt.ylim(ymin=0)
plt.title(f'N_train:{Nsteps_train:.0e}')
plt.xlabel('$\sigma$')
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz63_exp2_{param_string}_shortterm.pdf')

plt.figure(figsize=(8,4))
sns.boxplot('F','MAE', hue='type',
             data=res_training_loop.melt(id_vars=('F', 'n_training'), value_name='MAE', var_name='type'))
sns.despine()
plt.ylim(ymin=0)
plt.title(f'N_train:{Nsteps_train:.0e}')
plt.xlabel('$\sigma$')

plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz63_exp2_{param_string}_shortterm_boxplot.pdf')



# plots with clim performance

# climate std
plt.figure(figsize=(8,4))
sub = res_clim[['main_net_with_F_std', 'main_net_no_F_std', 'ref_net_std','model_std', 'F', 'n_training']]
sns.lineplot('F','clim std', hue='type', estimator=None, units='n_training',
             data=sub.melt(id_vars=('F', 'n_training'), value_name='clim std', var_name='type'))

sns.despine()
plt.fill_between([F_start, F_end], y1=[plt.gca().get_ylim()[0], plt.gca().get_ylim()[0]], y2=[plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]],
                 alpha=0.3, color='grey', zorder=10)
plt.title(f'N_train:{Nsteps_train:.0e}')
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz63_exp2_{param_string}_clim_std.pdf')

# climate mean
plt.figure(figsize=(8,4))
sub = res_clim[['main_net_with_F_mean', 'main_net_no_F_mean','ref_net_mean','model_mean', 'F', 'n_training']]
sns.lineplot('F','clim mean', hue='type', estimator=None, units='n_training',
             data=sub.melt(id_vars=('F', 'n_training'), value_name='clim mean', var_name='type'))

sns.despine()
plt.fill_between([F_start, F_end], y1=[plt.gca().get_ylim()[0], plt.gca().get_ylim()[0]], y2=[plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]],
                 alpha=0.3, color='grey', zorder=10)
plt.title(f'N_train:{Nsteps_train:.0e}')
plt.xlabel('$\sigma$')
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz63_exp2_{param_string}_clim_mean.pdf')


# plot both mean and std in same plot
plt.figure(figsize=(8,4))
sub = res_clim[['main_net_with_F_std', 'main_net_no_F_std', 'ref_net_std', 'model_std', 'F', 'n_training']]
# rename variables to get a legend that is easier to comprehend
sub = sub.rename(columns={'main_net_with_F_std':'main net with F',
            'main_net_no_F_std':'main net no F',
           'ref_net_std':'ref net'
           , 'model_std':'model'})
sns.lineplot('F','clim std', hue='type', estimator=None, units='n_training', alpha=0.8,
             data=sub.melt(id_vars=('F', 'n_training'), value_name='clim std', var_name='type'))
sub = res_clim[['main_net_with_F_mean', 'main_net_no_F_mean', 'ref_net_mean','model_mean', 'F', 'n_training']]
ax = sns.lineplot('F','clim mean', hue='type', estimator=None, units='n_training', alpha=0.8,
             data=sub.melt(id_vars=('F', 'n_training'), value_name='clim mean', var_name='type'),
             legend=False)
# getting dasehd lines is a bit tricky, but this works (https://stackoverflow.com/questions/51963725/how-to-plot-a-dashed-line-on-seaborn-lineplot)
# the bias lines are the last half of all lines on the ax (becase the two lineplots are on the same ax)
for line in ax.lines[len(ax.lines)//2:]:
    line.set_linestyle("--")

sns.despine()
plt.ylabel('climate mean (dashed) \n and std (solid)')
plt.fill_between([F_start, F_end], y1=[plt.gca().get_ylim()[0], plt.gca().get_ylim()[0]], y2=[plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]],
                 alpha=0.3, color='grey', zorder=10)
plt.title(f'N_train:{Nsteps_train:.0e}')
plt.xlabel('$\sigma$')
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz63_exp2_{param_string}_clim_mean_std.pdf')
