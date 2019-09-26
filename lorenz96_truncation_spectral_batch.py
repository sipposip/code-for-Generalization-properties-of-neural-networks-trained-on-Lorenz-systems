
import os
import sys
from tqdm import tqdm, trange
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
import keras
import tensorflow as tf
from keras import backend as K
# if you want to limit the number of CPUs use, uncomment the following and set
# intra_op_parallelism_threads and inter_op_parallelism_threads to whatever you want

# config = tf.ConfigProto(intra_op_parallelism_threads=20, inter_op_parallelism_threads=20,
#                         allow_soft_placement=True)
# session = tf.Session(config=config)
# K.set_session(session)


plotdir='plots_lorenz95_trunc'
os.system(f'mkdir {plotdir}')


leave_out_reg = np.array([[int((sys.argv[1])),int((sys.argv[2]))], # limits var1
                 [int((sys.argv[3])),int((sys.argv[4]))]]) # limtis var2


N = 40  # number of variables
F = 8 # forcing

def lorenz96(x,t):

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


def make_lorenz_run(y0, Nsteps):
    res = np.zeros((Nsteps,N))
    res[0] = y0
    sol = y0
    for i in trange(Nsteps):

        # we make only one step, but we gert a 2d array back. 9with only one element)
        # therefore, we extract this element with [1]
        sol = odeint(lorenz96, sol, t=[0,tstep])[1]
        res[i] = sol
    return res



Ntrain = int(1e5)
Ntest = Ntrain
Nspinup = 1000
tstep=0.01
n_epoch=30
lead_time=10

# two different initial conditions.
x_init1 = 6*np.ones(N) # initial state (equilibrium)
x_init1[19] += 0.01 # add small perturbation to 20th variable
x_init2 = 6*np.ones(N)
x_init2[19] += -0.03
modelrun_train = make_lorenz_run(x_init1, Ntrain+Nspinup)
modelrun_test = make_lorenz_run(x_init2, Ntest+Nspinup)
# remove spinup

modelrun_train = modelrun_train[Nspinup:]
modelrun_test = modelrun_test[Nspinup:]



norm_mean = modelrun_train.mean()
norm_std = modelrun_train.std(axis=0).mean()
modelrun_train = (modelrun_train  - norm_mean) / norm_std
modelrun_test = (modelrun_test  - norm_mean) / norm_std











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
    hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=1, validation_split=0.1 ,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=4,

                                                              verbose=0, mode='auto')]
                     )


    return model



X_train_full = modelrun_train[:-lead_time]
y_train_full = modelrun_train[lead_time:]
X_test = modelrun_test[:-lead_time]
y_test = modelrun_test[lead_time:]

# compute absolute powe spectrum for every timeslice
X_train_sp = np.abs(np.fft.rfft(X_train_full, axis=1))
X_test_sp = np.abs(np.fft.rfft(X_test, axis=1))

sp_corrmatrix = np.corrcoef(X_train_sp.T)
plt.figure()
plt.imshow(sp_corrmatrix, cmap = plt.cm.RdBu_r, vmax=1, vmin=-1)
plt.colorbar()
plt.savefig(f'{plotdir}/lorenz95_corrmatrix_spectral.svg')

min_idcs_sp = np.where(sp_corrmatrix**2 == np.min(sp_corrmatrix**2))
if len(min_idcs_sp[0]) > 1:
    min_idcs_sp = min_idcs_sp[0]

var1, var2 = min_idcs_sp
var1 = np.squeeze(var1)
var2 = np.squeeze(var2)

plt.figure()
sc = plt.scatter(X_train_sp[:,min_idcs_sp[0]], X_train_sp[:,min_idcs_sp[1]],alpha=0.5, s=1)
plt.xlabel(f'spectral coef {min_idcs_sp[0]}')
plt.ylabel(f'spectral coef {min_idcs_sp[1]}')

sns.despine()
plt.savefig(f'{plotdir}/lorenz95_most_independent_variable_poincarre_trajectory_spectral_spectral.png', dpi=400)



# define the region to be left out during the training.
# the phase-space region is defined on the poincare-section spanned
# by the two most independent wavenumber



param_string = '_'.join([str(e) for e in leave_out_reg])

x = X_train_sp
trunc_idcs = ~((x[:,var1]>leave_out_reg[0,0]) & (x[:,var1]<leave_out_reg[0,1]) &
                         (x[:,var2]>leave_out_reg[1,0]) & (x[:,var2]<leave_out_reg[1,1]))

trunc_idcs = np.squeeze(trunc_idcs)

X_train_trunc = X_train_full[trunc_idcs]
y_train_trunc = y_train_full[trunc_idcs]

X_train_trunc_sp = np.abs(np.fft.rfft(X_train_trunc, axis=1))

nplot = Ntrain
plt.figure()

sc = plt.scatter(X_train_trunc_sp[:nplot, var1],
                 X_train_trunc_sp[:nplot, var2],
                 alpha=0.5, s=1)

plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
sns.despine()
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz_95_model_spectral_{param_string}.png', dpi=400)


## now do the training, and prediction. we repeat this 10 times to capture randomness in the training
N_repeat = 10
mae_full = []
mae_trunc = []
for _ in range(N_repeat):
    network_full = train_network_noforcing(X_train_full, y_train_full, **params[lead_time])
    network_trunc = train_network_noforcing(X_train_trunc, y_train_trunc, **params[lead_time])

    # make predictions. we have to add an empty channel dimension, and
    # then squeeze it away form the predictions
    preds_full = network_full.predict(X_test[:,:,np.newaxis]).squeeze()
    preds_trunc = network_trunc.predict(X_test[:,:,np.newaxis]).squeeze()
    _mae_full = np.mean(np.abs(preds_full - y_test), axis=1)
    _mae_trunc = np.mean(np.abs(preds_trunc - y_test), axis=1)

    mae_full.append(_mae_full)
    mae_trunc.append(_mae_trunc)

mae_full = np.array(mae_full)
mae_trunc = np.array(mae_trunc)
# average over training repeats
mae_full = np.mean(mae_full, axis=0)
mae_trunc = np.mean(mae_trunc, axis=0)

vmax = np.max([np.max(mae_full), np.max(mae_trunc)])
vmin = np.min([np.min(mae_full), np.min(mae_trunc)])


fig,ax = plt.subplots(1)
plt.scatter(X_test_sp[:,var1],
            X_test_sp[:,var2],
            c = mae_full, vmin=vmin, vmax=vmax,
            )
plt.title('full network')
plt.colorbar()
rect = matplotlib.patches.Rectangle((leave_out_reg[0,0],leave_out_reg[1,0]),
                                    leave_out_reg[0,1] - leave_out_reg[0,0],
                                    leave_out_reg[1,1]-leave_out_reg[1,0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {min_idcs_sp[0]}')
plt.ylabel(f'spectral coef {min_idcs_sp[1]}')
plt.savefig(f'{plotdir}/mae_full_spectral_{param_string}.png', dpi=400)

fig,ax = plt.subplots(1)
plt.scatter(X_test_sp[:,var1],
            X_test_sp[:,var2],
            c = mae_trunc, vmin=vmin, vmax=vmax,
            )
plt.title('trunc network')
plt.colorbar()
rect = matplotlib.patches.Rectangle((leave_out_reg[0,0],leave_out_reg[1,0]),
                                    leave_out_reg[0,1] - leave_out_reg[0,0],
                                    leave_out_reg[1,1]-leave_out_reg[1,0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {min_idcs_sp[0]}')
plt.ylabel(f'spectral coef {min_idcs_sp[1]}')
plt.savefig(f'{plotdir}/mae_trunc_spectral_{param_string}.png', dpi=400)

# plot diff

fig,ax = plt.subplots(1)
plt.scatter(X_test_sp[:,var1],
            X_test_sp[:,var2],
            c = mae_trunc - mae_full,
            )
plt.title('trunc-full')
plt.colorbar()
rect = matplotlib.patches.Rectangle((leave_out_reg[0,0],leave_out_reg[1,0]),
                                    leave_out_reg[0,1] - leave_out_reg[0,0],
                                    leave_out_reg[1,1]-leave_out_reg[1,0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {min_idcs_sp[0]}')
plt.ylabel(f'spectral coef {min_idcs_sp[1]}')
plt.savefig(f'{plotdir}/mae_diff_spectral_{param_string}.png', dpi=400)



