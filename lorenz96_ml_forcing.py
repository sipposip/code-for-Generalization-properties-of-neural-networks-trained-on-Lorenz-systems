"""
python3


@author: Sebastian Scher, march 2019
"""
import os

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
import keras



name='F_forcing' # for plots


N = 40  # number of variables
F = 6 # forcing

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


plotdir='plots_lorenz96'
os.system(f'mkdir {plotdir}')


# paramters for experiments

Nsteps = 100000
tstep=0.01
t_arr = np.arange(0, Nsteps) * tstep
n_epoch = 30
lead_time = 10  # in steps
hidden_size=100
lr = 0.001
F_start = 4
F_end = 16
n_F_steps = 20

#exptype='steps'
exptype='linear'

if exptype == 'steps':
    F_arr = np.array([ F_start * np.ones(len(t_arr)//n_F_steps) + (F_end - F_start)/n_F_steps * i
                  for i in range(n_F_steps)  ]).flatten()
elif exptype == 'linear':
    F_arr = F_start * np.ones(len(t_arr)) + (F_end - F_start) / Nsteps / tstep * t_arr


N_start_perc = 20  # fraction to be used of the modelrun for training, both from the start and the end of the rn
N_start = int(Nsteps * N_start_perc/100)



# we make two runs, started with slightly different initial conditions
# one will be the training and one the test run
x_init1 = F*np.ones(N) # initial state (equilibrium)
x_init1[19] += 0.01 # add small perturbation to 20th variable
x_init2 = F*np.ones(N)
x_init2[19] += -0.03
modelrun_train = make_lorenz_run(x_init1, Nsteps, F_arr)
modelrun_test = make_lorenz_run(x_init2, Nsteps, F_arr)



plt.figure()
plt.contourf(modelrun_train)
plt.colorbar()

# for loezn96, we dont have to normalize per variable, because all should have the same
# st and mean anywary, so we compute the total mean,  and the std for each gridpoint and then
# average all std
norm_mean = modelrun_train.mean()
norm_std = modelrun_train.std(axis=0).mean()
modelrun_train = (modelrun_train  - norm_mean) / norm_std
modelrun_test = (modelrun_test - norm_mean) / norm_std

F_arr_normed = (F_arr - F_arr.mean()) / F_arr.std()


# now add F as a second layer (repeating it for every gridpoint)
modelrun_train_with_F = np.stack([modelrun_train ,np.tile(F_arr_normed, (N,1)).T], axis=2)
modelrun_test_with_F = np.stack([modelrun_test ,np.tile(F_arr_normed, (N,1)).T], axis=2)
assert(modelrun_train_with_F.shape == (Nsteps,N,2))

params = {100:{"activation": "sigmoid", "conv_depth": 32, "kernel_size": 5, "lr": 0.003, "n_conv": 9},
          10:{"activation": "relu", "conv_depth": 128, "kernel_size": 5, "lr": 0.003, "n_conv": 2},
          1:{"activation": "relu", "conv_depth": 128, "kernel_size": 3, "lr": 0.003, "n_conv": 1}}

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

    print(model.summary())
    hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=0, validation_split=0.1 ,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=4,

                                                              verbose=1, mode='auto')]
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

    print(model.summary())
    hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=0, validation_split=0.1 ,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=4,

                                                              verbose=1, mode='auto')]
                     )


    return model


# X and y data are the same, but shifted by lead-time,
X_train_full = modelrun_train_with_F[:-lead_time]
y_train_full = modelrun_train[lead_time:]

X_test_full = modelrun_test_with_F[:-lead_time]
y_test_full = modelrun_test[lead_time:]


# split up the simulations in a start and a end (and a discarded middle) part.

X_train_start = X_train_full[:N_start]
y_train_start = y_train_full[:N_start]
X_test_start = X_test_full[:N_start]
y_test_start = y_test_full[:N_start]

X_train_end = X_train_full[-N_start:]
y_train_end = y_train_full[-N_start:]
X_test_end = X_test_full[-N_start:]
y_test_end = y_test_full[-N_start:]

F_arr_start = F_arr_normed[:N_start]
F_arr_end = F_arr_normed[-N_start:]

N_training = 10


# now loop over relizations (we do the same training a couple of times

res_training_loop = []
for i in range(N_training):
    # parameters string for filenames
    param_string = '_'.join([str(e) for e in (Nsteps, n_epoch, lead_time, F_start, F_end, N_start_perc, name, exptype,
                                              'training'+str(i).zfill(3))])
    # now train a network on the start part
    network_start = train_network(X_train_start, y_train_start, **params[lead_time])


    # now the same, but trained on the end
    network_end = train_network(X_train_end, y_train_end, **params[lead_time])


    # and now train a network that does not use the F-input
    network_base_start = train_network_noforcing(X_train_start[:,:,0], y_train_start, **params[lead_time])
    network_base_end = train_network_noforcing(X_train_end[:,:,0], y_train_end, **params[lead_time])


    # now do forecasting with the start-trained network, both on the start and on the end climate
    # the model predict includes the empty channel dim, which we squeeze away
    pred_startnet_on_start = network_start.predict(X_test_start).squeeze()
    pred_startnet_on_end = network_start.predict(X_test_end).squeeze()

    abserr_startnet_on_start = np.abs(pred_startnet_on_start - y_test_start).mean(axis=1)
    abserr_startnet_on_end = np.abs(pred_startnet_on_end - y_test_end).mean(axis=1)

    pred_endnet_on_start = network_end.predict(X_test_start).squeeze()
    pred_endnet_on_end = network_end.predict(X_test_end).squeeze()

    abserr_endnet_on_start = np.abs(pred_endnet_on_start - y_test_start).mean(axis=1)
    abserr_endnet_on_end = np.abs(pred_endnet_on_end - y_test_end).mean(axis=1)

    pred_startbasenet_on_start = network_base_start.predict(X_test_start[:,:,0,np.newaxis]).squeeze()
    pred_startbasenet_on_end = network_base_start.predict(X_test_end[:,:,0,np.newaxis]).squeeze()
    pred_endbasenet_on_start = network_base_end.predict(X_test_start[:,:,0,np.newaxis]).squeeze()
    pred_endbasenet_on_end = network_base_end.predict(X_test_end[:,:,0,np.newaxis]).squeeze()

    abserr_startbasenet_on_start = np.abs(pred_startbasenet_on_start - y_test_start).mean(axis=1)
    abserr_startbasenet_on_end = np.abs(pred_startbasenet_on_end - y_test_end).mean(axis=1)
    abserr_endbasenet_on_start = np.abs(pred_endbasenet_on_start - y_test_start).mean(axis=1)
    abserr_endbasenet_on_end = np.abs(pred_endbasenet_on_end - y_test_end).mean(axis=1)

    # predict with the lorenz equations, with the mean F of the other epoch
    # for this we need to "un"normalize X_test
    X_test_end_notnormed = X_test_end * norm_std + norm_mean
    X_test_start_notnormed = X_test_start * norm_std + norm_mean
    # we also need to unnormalize F_arr, the weights for F are the last element of the norm weights
    mean_F_start_notnormed = F_arr[:N_start].mean()
    mean_F_end_notnormed = F_arr[-N_start:].mean()


    def predict_with_lorenz(X,F):
        preds = []
        for x in X:
            pred = odeint(lorenz96, x, t=[0,tstep * lead_time], args=(F,))[1]
            preds.append(pred)
        return np.array(preds)


    pred_lorenz_startF_on_end = predict_with_lorenz(X_test_end_notnormed[:,:,0], F=mean_F_start_notnormed)
    pred_lorenz_endF_on_start = predict_with_lorenz(X_test_start_notnormed[:,:,0], F=mean_F_end_notnormed)
    pred_lorenz_startF_on_start = predict_with_lorenz(X_test_start_notnormed[:,:,0], F=mean_F_start_notnormed)
    pred_lorenz_endF_on_end = predict_with_lorenz(X_test_end_notnormed[:,:,0], F=mean_F_end_notnormed)
    # now we have to normalize the predictions to get the same MAE as for the network predictions
    pred_lorenz_startF_on_end = (pred_lorenz_startF_on_end - norm_mean) / norm_std
    pred_lorenz_endF_on_start = (pred_lorenz_endF_on_start - norm_mean) / norm_std
    pred_lorenz_startF_on_start = (pred_lorenz_startF_on_start - norm_mean) / norm_std
    pred_lorenz_endF_on_end = (pred_lorenz_endF_on_end - norm_mean) / norm_std


    abserr_lorenz_startF_on_end = np.abs(pred_lorenz_startF_on_end - y_test_end).mean()
    abserr_lorenz_endF_on_start = np.abs(pred_lorenz_endF_on_start - y_test_start).mean()
    abserr_lorenz_startF_on_start = np.abs(pred_lorenz_startF_on_start - y_test_start).mean()
    abserr_lorenz_endF_on_end = np.abs(pred_lorenz_endF_on_end - y_test_end).mean()



    #%%
    sns.set_context('notebook', font_scale=1.0, rc={"lines.linewidth": 3.})
    sns.set_palette('colorblind')
    plt.rcParams['savefig.bbox'] = 'tight'

    plt.figure(figsize=(14,10))
    plt.subplot(211)
    plt.plot(t_arr, F_arr)
    plt.axvline(N_start*tstep)
    plt.axvline((Nsteps - N_start)*tstep)
    plt.xlabel('t')
    plt.ylabel('F')
    sns.despine()
    plt.title(f'Ntrain:{N_start}')
    plt.subplot(212)

    df = pd.DataFrame({'mae':[abserr_startnet_on_start.mean(), abserr_endnet_on_start.mean(),
                              abserr_endnet_on_end.mean(), abserr_startnet_on_end.mean(),
                              abserr_startbasenet_on_start.mean(), abserr_endbasenet_on_start.mean(),
                              abserr_endbasenet_on_end.mean(), abserr_startbasenet_on_end.mean(),
                              abserr_lorenz_endF_on_start.mean(),abserr_lorenz_startF_on_end.mean(),
                              abserr_lorenz_startF_on_start.mean(),abserr_lorenz_endF_on_end.mean()

                              ],
                       'fc_on': ['start', 'start',
                                 'end', 'end',
                                 'start', 'start',
                                 'end', 'end',
                                 'start', 'end',
                                 'start', 'end'
                               ],
                       'network': ['trained_on_start', 'trained_on_end',
                                    'trained_on_end', 'trained_on_start',
                                    'trained_on_start_noF', 'trained_on_end_noF',
                                    'trained_on_end_noF', 'trained_on_start_noF',
                                    'lorenz_fixedFend', 'lorenz_fixedFstart',
                                    'lorenz_fixedFstart', 'lorenz_fixedFend'

                       ]})

    os.system('mkdir -p data')
    df.to_pickle(f'data/error_df_{param_string}.pkl')

    sns.barplot('fc_on', 'mae', hue='network', data=df)
    sns.despine()
    plt.legend(fancybox=True, framealpha=0.5)
    plt.savefig(f'{plotdir}/lorenz95_experiment_result_overview_{param_string}.svg')


    df['itraining'] = i
    res_training_loop.append(df)

res_df = pd.concat(res_training_loop)
res_df = res_df.reset_index()

plt.figure(figsize=(14,10))
plt.subplot(211)
plt.plot(t_arr, F_arr)
plt.axvline(N_start*tstep)
plt.axvline((Nsteps - N_start)*tstep)
plt.xlabel('t')
plt.ylabel('F')
sns.despine()
plt.title(f'Ntrain:{N_start}')
plt.subplot(212)
sns.barplot('fc_on', 'mae', hue='network', data=res_df,
            ci='sd', # this plots the std of the data as error bars (instead of the standard bootstrapping in seaborn)
           # capsize=0.1
            )
sns.despine()
plt.legend(fancybox=True, framealpha=0.5)
plt.savefig(f'{plotdir}/lorenz95_experiment_result_overview_{param_string}_alltraining.svg')

plt.figure(figsize=(14,10))
plt.subplot(211)
plt.plot(t_arr, F_arr)
plt.axvline(N_start*tstep)
plt.axvline((Nsteps - N_start)*tstep)
plt.xlabel('t')
plt.ylabel('F')
sns.despine()
plt.title(f'Ntrain:{N_start}')
plt.subplot(212)
sns.boxplot('fc_on', 'mae', hue='network', data=res_df)
sns.despine()
plt.legend(fancybox=True, framealpha=0.5)
plt.savefig(f'{plotdir}/lorenz95_experiment_result_overview_{param_string}_alltraining_boxplot.svg')