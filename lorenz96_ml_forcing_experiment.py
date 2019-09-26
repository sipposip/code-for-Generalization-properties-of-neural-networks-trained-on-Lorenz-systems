"""
python3

This script does a forcing experiment with the Lorenz95 networks.
For this experiment, a main run with the lorenz95 model is made. In this run, the forcing F
 of the model linearly increases from a start value (F_start) to a end value (F_end).
 Then 2 networks are trained on this run (one using F as input, one not using F).
 These networks are then evaluated on different lorenz95 runs, each with a different fixed value of F
 Additional to these 2 main networks, for each test lorenz95 runs with fixed F, a network (without using F as input)
 is trained on this run, and evaluated on a second run with same fixed F (but different initial condition).


@author: Sebastian Scher, August 2019
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

from keras import backend as K
import tensorflow as tf
import keras
# if you want to limit the number of CPUs use, uncomment the following and set
# intra_op_parallelism_threads and inter_op_parallelism_threads to whatever you want

# config = tf.ConfigProto(intra_op_parallelism_threads=20, inter_op_parallelism_threads=20,
#                         allow_soft_placement=True)
# session = tf.Session(config=config)
# K.set_session(session)

name='F_forcing' # for plots


N = 40  # number of variables

def lorenz96(x,t,F):

  # compute state derivatives
  d = np.zeros(N)
  # first the 3 edge cases: i=1,2,N
  d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
  d[1] = (x[2] - x[N-1]) * x[0]- x[1]
  d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
  # then the general case
  #for i in range(2, N-1):
  #    d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
  d[2:N-1] =  (x[2+1:N-1+1] - x[2-2:N-1-2]) * x[2-1:N-1-1] - x[2:N-1]  ## this is equivalent but faster
  # add the forcing term
  d = d + F

  return d


def make_lorenz_run(y0, Nsteps  , F_arr):
    # note: F is not normalized here
    res = np.zeros((Nsteps,N))
    res[0] = y0
    sol = y0
    for i in range(Nsteps-1):  #-1 because we also include the initial state in Nsteps
        F = F_arr[i]
        # print(i)
        # we make only one step, but we gert a 2d array back. 9with only one element)
        # therefore, we extract this element with [1]
        sol = odeint(lorenz96, sol, t=[0,tstep], args=(F,))[1]
        res[i] = sol
    return res


def make_network_climrun_with_F(network, x_init, Nsteps, F):
    # crreate a "climate run" with the network. F needs to be normed
    network_clim_run = np.zeros((Nsteps + 1, N))
    network_clim_run[0] = x_init
    state = x_init
    # add empty time and channel dimension
    state = state[np.newaxis,:, np.newaxis]
    for i in range(Nsteps):
        # add F to input
        state = np.concatenate ([state, np.ones((1,N,1)) * F], axis=2)
        state = network.predict(state)
        network_clim_run[i + 1] = state.squeeze()

    return network_clim_run


def make_network_climrun_no_F(network, x_init, Nsteps):
    # crreate a "climate run" with the network
    network_clim_run = np.zeros((Nsteps + 1, N))
    network_clim_run[0] = x_init
    state = x_init
    # add empty time and channel dimension
    state = state[np.newaxis,:, np.newaxis]
    for i in range(Nsteps):
        state = network.predict(state)
        network_clim_run[i + 1] = state.squeeze()

    return network_clim_run


plotdir='plots_lorenz95_exp2'
os.system(f'mkdir {plotdir}')


# paramters for experiments

Nsteps_train = 100
Nsteps_test = 10000

F_start = 8
F_end = 9
F_test_points = [4, 5, 6, 7, 8, 8.5, 9, 10, 12, 14]

N_training = 10 # how often ot repeat the experiment

# fixed parameters (from tuning)
tstep=0.01
t_arr = np.arange(0, Nsteps_train) * tstep
n_epoch = 30
lead_time = 10  # in steps
hidden_size=100
lr = 0.001
Nsteps_clim = Nsteps_test // lead_time
params = {100:{"activation": "sigmoid", "conv_depth": 32, "kernel_size": 5, "lr": 0.003, "n_conv": 9},
          10:{"activation": "relu", "conv_depth": 128, "kernel_size": 5, "lr": 0.003, "n_conv": 2},
          1:{"activation": "relu", "conv_depth": 128, "kernel_size": 3, "lr": 0.003, "n_conv": 1}}

params_with_F = {100:{"activation": "sigmoid", "conv_depth": 32, "kernel_size": 5, "lr": 0.003, "n_conv": 9},
          10:{"activation": "relu", "conv_depth": 128, "kernel_size": 3, "lr": 0.003, "n_conv": 4},
          1:{"activation": "relu", "conv_depth": 128, "kernel_size": 5, "lr": 0.003, "n_conv": 1}}

param_string = '_'.join([str(e) for e in (Nsteps_train, Nsteps_test, n_epoch, lead_time,
                                          F_start, F_end,  name, N_training
                                          )])

F_arr = F_start * np.ones(len(t_arr)) + (F_end - F_start) / Nsteps_train / tstep * t_arr


# two different initial conditions.
x_init1 = 6*np.ones(N) # initial state (equilibrium)
x_init1[19] += 0.01 # add small perturbation to 20th variable
x_init2 = 6*np.ones(N)
x_init2[19] += -0.03
# for the main training run we do not need a corresponding test run, as the networks
# will only be tested on the fixed F runs that will be made later on
modelrun_train = make_lorenz_run(x_init1, Nsteps_train, F_arr)


# for lorenz95, we dont have to normalize per variable, because all should have the same
# st and mean anyway, so we compute the total mean,  and the std for each gridpoint and then
# average all std
# the same normilzation values will be used for all other runs later on
norm_mean = modelrun_train.mean()
norm_std = modelrun_train.std(axis=0).mean()
modelrun_train = (modelrun_train  - norm_mean) / norm_std

F_norm_mean = F_arr.mean()
F_norm_std = F_arr.std()
F_arr_normed = (F_arr - F_norm_mean) / F_norm_std

# now add F as a second layer (repeating it for every gridpoint)
modelrun_train_with_F = np.stack([modelrun_train ,np.tile(F_arr_normed, (N,1)).T], axis=2)

assert(modelrun_train_with_F.shape == (Nsteps_train,N,2))



class PeriodicPadding(keras.layers.Layer):
    def __init__(self, axis, padding, **kwargs):
        """
        layer with periodic padding for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along
        padding: number of cells to pad
        """

        super(PeriodicPadding, self).__init__(**kwargs)

        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(padding, int):
            padding = (padding,)

        self.axis = axis
        self.padding = padding

    def build(self, input_shape):
        super(PeriodicPadding, self).build(input_shape)

    # in order to be able to load the saved model we need to define
    # get_config
    def get_config(self):
        config = {
            'axis': self.axis,
            'padding': self.padding,

        }
        base_config = super(PeriodicPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):

        tensor = input
        for ax, p in zip(self.axis, self.padding):
            # create a slice object that selects everything form all axes,
            # except only 0:p for the specified for right, and -p: for left
            ndim = len(tensor.shape)
            ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
            ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
            right = tensor[ind_right]
            left = tensor[ind_left]
            middle = tensor
            tensor = tf.concat([right, middle, left], axis=ax)
        return tensor

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        for ax, p in zip(self.axis, self.padding):
            output_shape[ax] += 2 * p
        return tuple(output_shape)


def train_network(X_train, y_train, lr,kernel_size, conv_depth, n_conv, activation):
    """
    :param X_train:
    :param y_train:
    :param kernel_size:
    :param conv_depth:
    :param n_conv: >=1
    :return:
    """

    n_channel = 2
    n_pad = int(np.floor(kernel_size/2))
    layers = [
        PeriodicPadding(axis=1,padding=n_pad,input_shape=(N ,n_channel)),
        keras.layers.Conv1D(kernel_size=kernel_size,filters=conv_depth, activation=activation,
                            padding='valid')]

    for i in range(n_conv-1):
        layers.append(PeriodicPadding(axis=1,padding=n_pad))
        layers.append(keras.layers.Conv1D(kernel_size=kernel_size,filters=conv_depth, activation=activation,
                            padding='valid'))

    layers.append(PeriodicPadding(axis=1, padding=n_pad))
    layers.append(keras.layers.Conv1D(kernel_size=kernel_size, filters=1,  activation='linear', padding='valid'))


    model = keras.Sequential(layers)

    #  we have to add an empty channel dimension for y_train
    y_train = y_train[..., np.newaxis]
    optimizer = keras.optimizers.adam(lr=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # print(model.summary())
    hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=0, validation_split=0.1 ,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=4,

                                                              verbose=0, mode='auto')]
                     )


    return model


def train_network_noforcing(X_train, y_train, lr,kernel_size, conv_depth, n_conv, activation):
    """
    :param X_train:
    :param y_train:
    :param kernel_size:
    :param conv_depth:
    :param n_conv: >=1
    :return:
    """

    n_channel = 1 # empty
    n_pad = int(np.floor(kernel_size/2))
    layers = [
        PeriodicPadding(axis=1,padding=n_pad,input_shape=(N ,n_channel)),
        keras.layers.Conv1D(kernel_size=kernel_size,filters=conv_depth, activation=activation,
                            padding='valid')]

    for i in range(n_conv-1):
        layers.append(PeriodicPadding(axis=1,padding=n_pad))
        layers.append(keras.layers.Conv1D(kernel_size=kernel_size,filters=conv_depth, activation=activation,
                            padding='valid'))

    layers.append(PeriodicPadding(axis=1, padding=n_pad))
    layers.append(keras.layers.Conv1D(kernel_size=kernel_size, filters=1,  activation='linear', padding='valid'))


    model = keras.Sequential(layers)

    #  we have to add an empty channel dimension
    y_train = y_train[..., np.newaxis]
    X_train = X_train[..., np.newaxis]
    optimizer = keras.optimizers.adam(lr=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # print(model.summary())
    hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=0, validation_split=0.1 ,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=4,

                                                              verbose=0, mode='auto')]
                     )


    return model


# X and y data are the same, but shifted by lead-time,
# we call this _main, because it is the main run with varying F
X_train_main = modelrun_train_with_F[:-lead_time]
y_train_main = modelrun_train[lead_time:]

# this is now the model run for training. For testing, we make additional runs with different fixed F values

runs_fixedF_train = []
runs_fixedF_test = []
for F in F_test_points:
    F_normed = (F - F_norm_mean) / F_norm_std
    # make lorenz runs with fixed F. since the make_lorenz_run function expects
    # an array with F-values, we simply repeeat the fixed F values
    run_train = make_lorenz_run(x_init1, Nsteps_train, F_arr= np.ones(Nsteps_train)*F)
    run_test = make_lorenz_run(x_init1, Nsteps_test, F_arr= np.ones(Nsteps_test)*F)
    # normalize with same weights as used for training
    run_train_norm = (run_train - norm_mean) / norm_std
    run_test_norm = (run_test - norm_mean) / norm_std
    # shoft by lead time
    X_train = run_train_norm[:-lead_time]
    y_train = run_train_norm[lead_time:]
    X_test = run_test_norm[:-lead_time]
    y_test = run_test_norm[lead_time:]
    # now add F as a second layer to train (repeating it for every gridpoint)
    # here we need to add the normalized value of F
    X_train = np.stack([X_train, np.ones(X_train.shape) * F_normed], axis=2)
    X_test = np.stack([X_test, np.ones(X_test.shape) * F_normed], axis=2)

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
    main_network_with_F = train_network(X_train_main, y_train_main, **params_with_F[lead_time])
    # for the network without F, we dont need F, so only pass in the first channel
    # (which contains the 40 lorenz gridpoints)
    main_network_no_F = train_network_noforcing(X_train_main[:,:,0], y_train_main, **params[lead_time])

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
        preds_no_F = main_network_no_F.predict(X_test[:,:,0, np.newaxis]).squeeze()
        err_no_F = np.abs(preds_no_F - y_test).mean()

        # now train a network on the fixed F run
        network_fixed = train_network_noforcing(X_train[:,:,0], y_train, **params[lead_time])
        preds_fixed = network_fixed.predict(X_test[:,:,0, np.newaxis]).squeeze()
        err_fixed = np.abs(preds_fixed - y_test).mean()

        # store in dataframe
        res_training_loop.append(pd.DataFrame({'mae':err_with_F, 'mae_noF':err_no_F,'mae_ref':err_fixed,'F':F,
                                               'n_training':i_training}, index=[0]))

        # make network climate runs and compute climate statistics
        # we start the climate runs with the first state from X_test, without the f channel
        network_with_F_clim = make_network_climrun_with_F(main_network_with_F,X_test[0,:,0],Nsteps_clim, F_normed )
        network_no_F_clim = make_network_climrun_no_F(main_network_no_F,X_test[0,:,0],Nsteps_clim)
        network_ref_clim = make_network_climrun_no_F(network_fixed,X_test[0,:,0],Nsteps_clim)
        # the network climate has a timestep of lead_time, so we have to reduce the test run as well
        # to get the same time frequency
        model_reduced = X_test[::lead_time]
        # remove channel with F
        model_reduced = model_reduced[:,:,0]

        # now compute std and mean of the clims and store in dataframe. We compute the std over all timesteps
        # and variables in one go (we could also compute std for every gridpoing, and then average, but it should not
        # matter too much since the model is so symmetric)

        df = dict(model_mean = model_reduced.mean(),
                  model_std = model_reduced.std(),
                  main_net_with_F_mean = network_with_F_clim.mean(),
                  main_net_with_F_std = network_with_F_clim.std(),
                  main_net_no_F_mean = network_no_F_clim.mean(),
                  main_net_no_F_std = network_no_F_clim.std(),
                  ref_net_mean = network_ref_clim.mean(),
                  ref_net_std = network_ref_clim.std(),
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
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz95_exp2_{param_string}_shortterm.pdf')

plt.figure(figsize=(8,4))
sns.boxplot('F','MAE', hue='type',
             data=res_training_loop.melt(id_vars=('F', 'n_training'), value_name='MAE', var_name='type'))
sns.despine()
plt.ylim(ymin=0)
plt.title(f'N_train:{Nsteps_train:.0e}')
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz95_exp2_{param_string}_shortterm_boxplot.pdf')



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
plt.savefig(f'{plotdir}/lorenz95_exp2_{param_string}_clim_std.pdf')

# climate mean
plt.figure(figsize=(8,4))
sub = res_clim[['main_net_with_F_mean', 'main_net_no_F_mean','ref_net_mean','model_mean', 'F', 'n_training']]
sns.lineplot('F','clim mean', hue='type', estimator=None, units='n_training',
             data=sub.melt(id_vars=('F', 'n_training'), value_name='clim mean', var_name='type'))

sns.despine()
plt.fill_between([F_start, F_end], y1=[plt.gca().get_ylim()[0], plt.gca().get_ylim()[0]], y2=[plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]],
                 alpha=0.3, color='grey', zorder=10)
plt.title(f'N_train:{Nsteps_train:.0e}')
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz95_exp2_{param_string}_clim_mean.pdf')