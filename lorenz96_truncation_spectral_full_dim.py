
import matplotlib
matplotlib.use('agg')

import os
from tqdm import tqdm, trange
# import pandas as pd
import seaborn as sns
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

from keras import backend as K
# if you want to limit the number of CPUs use, uncomment the following and set
# intra_op_parallelism_threads and inter_op_parallelism_threads to whatever you want

# config = tf.ConfigProto(intra_op_parallelism_threads=20, inter_op_parallelism_threads=20,
#                         allow_soft_placement=True)
# session = tf.Session(config=config)
# K.set_session(session)


plotdir='plots_lorenz95_trunc_fulldim'
os.system(f'mkdir {plotdir}')

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
# here we need the test run only as a way to generate an initial state,
# as we will create the real test run later on
test_init = modelrun_test[-1]


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


# compute absolute power spectrum for every timeslice
X_train_sp = np.abs(np.fft.rfft(X_train_full, axis=1))
X_test_sp = np.abs(np.fft.rfft(X_test, axis=1))

# we use same limits for every spectral coefficient
# note that we have N/2+1 coefficients

N_coeff = N//2+1
assert(N_coeff == X_train_sp.shape[1])
lims = [0., 10.]

param_string = 'spectrull_full_dim_'+'_'.join([str(e) for e in lims])

# logfile for storing output
logfile=open(f'{plotdir}/lorenz95_model_spectral_{param_string}_results.txt','w')



trunc_idcs = []
for x in X_train_sp:
    # dont include the point in case all spectral coefficients lie within the limits
    include = ~np.all([ x[i] > lims[0] and x[i] < lims[1] for i in range(N_coeff)])
    trunc_idcs.append(include)

trunc_idcs = np.array(trunc_idcs)
print(f'n excluded = {np.sum(~trunc_idcs)}')

# now to have enough test data, we run the lorenz95 model until we have enough states in the cut-out region
# this we call test2 in this script
X_test2 = []
y_test2 = []
x = test_init
# we run until we have found 1e3 states
while len(X_test2) < int(1e3):
    # make a one-timestep prediction
    x = make_lorenz_run(x, Nsteps=1).squeeze()
    # normalize and compute spectral coefficients
    x_norm = (x - norm_mean) / norm_std
    x_sp = np.abs(np.fft.rfft(x_norm))
    # if it is in the desired region, save the state
    in_cut_put_region = np.all([x_sp[i] > lims[0] and x_sp[i] < lims[1] for i in range(N_coeff)])
    if in_cut_put_region:
        X_test2.append(x_norm)
        print(len(X_test2))
        # now we also need the target (model lead_time steps later)
        target = make_lorenz_run(x, Nsteps=lead_time)
        # only last timestep
        target = target[-1]
        target_norm = (target - norm_mean) / norm_std
        y_test2.append(target_norm)

X_test2 = np.array(X_test2)
y_test2 = np.array(y_test2)

# combine with standard test data
X_test_full = np.concatenate([X_test, X_test2], axis=0)
y_test_full = np.concatenate([y_test, y_test2], axis=0)

X_test_sp = np.abs(np.fft.rfft(X_test_full, axis=1))
X_test2_sp = np.abs(np.fft.rfft(X_test2, axis=1))

# sanity check: check whether all points in X_test2 fullfull the selection criterion
for x in X_test2_sp:
    assert(np.all([ x[i] > lims[0] and x[i] < lims[1] for i in range(N_coeff)]))

# for plotting, it is convenient to have the indices for X_test_sp that correspond to
# X_test2 (so the points in the selected region). Since they were stacked together,
# it is the second part
test_idcs_leftout_region = np.arange(len(X_test), len(X_test_full))
assert(len(test_idcs_leftout_region) == len(X_test2))
for x in X_test_sp[test_idcs_leftout_region]:
    assert(np.all([ x[i] > lims[0] and x[i] < lims[1] for i in range(N_coeff)]))

X_train_trunc = X_train_full[trunc_idcs]
y_train_trunc = y_train_full[trunc_idcs]

X_train_trunc_sp = np.abs(np.fft.rfft(X_train_trunc, axis=1))


# spectral coefficients to plot (same as in spectral poincare-selection)
var1 = 1
var2 = 12
nplot = Ntrain


# plot the training and test data

plt.figure()
sc = plt.scatter(X_train_trunc_sp[:, var1],
                 X_train_trunc_sp[:, var2],
                 alpha=0.5, s=1)

plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
sns.despine()
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz_95_model_spectral_{param_string}.png', dpi=400)

plt.figure()
sc = plt.scatter(X_test_sp[:, var1],
                 X_test_sp[:, var2],
                 alpha=0.5, s=1)

plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
sns.despine()
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz_95_model_spectral_testdata{param_string}.png', dpi=400)

plt.figure()
sc = plt.scatter(X_test_sp[test_idcs_leftout_region, var1],
                 X_test_sp[test_idcs_leftout_region, var2],
                 alpha=0.5, s=1)

plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
sns.despine()
plt.tight_layout()
plt.savefig(f'{plotdir}/lorenz_95_model_spectral_testdata_only_cutout_{param_string}.png', dpi=400)


# now do the training, and prediction. we repeat this 10 times to capture randomness in the training
N_repeat = 10
mae_full = []
mae_trunc = []
trained_nets_full = []
trained_nets_trunc = []
for i in range(N_repeat):
    print(f'training network number {i}')
    network_full = train_network_noforcing(X_train_full, y_train_full, **params[lead_time])
    network_trunc = train_network_noforcing(X_train_trunc, y_train_trunc, **params[lead_time])

    # make predictions. we have to add an empty channel dimension, and
    # then squeeze it away form the predictions
    preds_full = network_full.predict(X_test_full[:,:,np.newaxis]).squeeze()
    preds_trunc = network_trunc.predict(X_test_full[:,:,np.newaxis]).squeeze()
    _mae_full = np.mean(np.abs(preds_full - y_test_full), axis=1)
    _mae_trunc = np.mean(np.abs(preds_trunc - y_test_full), axis=1)

    mae_full.append(_mae_full)
    mae_trunc.append(_mae_trunc)
    trained_nets_full.append(network_full)
    trained_nets_trunc.append(network_trunc)

mae_full_all = np.array(mae_full)
mae_trunc_all = np.array(mae_trunc)
# average over training repeats (first axis)
mae_full = np.mean(mae_full_all, axis=0)
mae_trunc = np.mean(mae_trunc_all, axis=0)
# compute standard deviation of mean error between different realizations
std_mean_error_full = np.std(np.mean(mae_full_all, axis=1))
std_mean_error_trunc = np.std(np.mean(mae_trunc_all, axis=1))
print(f'std_mean_error_full:{std_mean_error_full}', file=logfile )
print(f'std_mean_error_trunc:{std_mean_error_trunc}', file=logfile)

assert(len(X_test_sp) == len(X_test_full))
# make "climate" runs with the trained networks.
# for this we initialize with a random chosen initial state from the test set. However, we make sure
# that it is not a state from the cut-out region
while True:
    idx = np.random.randint(0,len(X_test_full)-1)
    state_init = X_test_full[idx]
    # if the state is not in the cut-out region, we use and and continue
    x = X_test_sp[idx]
    if ~np.all([ x[i] > lims[0] and x[i] < lims[1] for i in range(N_coeff)]):
        break

# now make a long climate run
nclim_test = int(1e4)
for net_trunc in trained_nets_trunc:
    state = state_init
    clim_run = [state]
    for _ in trange(nclim_test):
        # we have to add an empty time and empty channel dimension
        state = net_trunc.predict(state[np.newaxis, :, np.newaxis]).squeeze()
        clim_run.append(state)

    clim_run = np.array(clim_run)
    # check whether we have at least one state in clim_run that is in the cut-out region
    # for this we need the spectral components
    clim_run_sp = np.abs(np.fft.rfft(clim_run, axis=1))
    in_cutout = []
    for x in clim_run_sp:
        in_cutout.append(np.all([ x[i] > lims[0] and x[i] < lims[1] for i in range(N_coeff)]))

    print(f'{np.sum(in_cutout)} states found in network climate simulation')
    print(f'{np.sum(in_cutout)} states found in network climate simulation', file=logfile)


# min and max for plotting
vmax = np.max([np.max(mae_full), np.max(mae_trunc)])
vmin = np.min([np.min(mae_full), np.min(mae_trunc)])


# now plot the forecast errors

fig,ax = plt.subplots(1)
plt.scatter(X_test_sp[:,var1],
            X_test_sp[:,var2],
            c = mae_full, vmin=vmin, vmax=vmax,
            )
plt.title('full network')
plt.colorbar()
rect = matplotlib.patches.Rectangle((lims[0],lims[0]),
                                    lims[1] - lims[0], lims[1] - lims[0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
plt.savefig(f'{plotdir}/mae_full_spectral_{param_string}.png', dpi=400)

fig,ax = plt.subplots(1)
plt.scatter(X_test_sp[test_idcs_leftout_region,var1],
            X_test_sp[test_idcs_leftout_region,var2],
            c = mae_full[test_idcs_leftout_region], vmin=vmin, vmax=vmax,
            )
plt.title('full network')
plt.colorbar()
rect = matplotlib.patches.Rectangle((lims[0],lims[0]),
                                    lims[1] - lims[0], lims[1] - lims[0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
plt.savefig(f'{plotdir}/mae_full_spectral_trunc_test_only{param_string}.png', dpi=400)



fig,ax = plt.subplots(1)
plt.scatter(X_test_sp[:,var1],
            X_test_sp[:,var2],
            c = mae_trunc, vmin=vmin, vmax=vmax,
            )
plt.title('trunc network')
plt.colorbar()
rect = matplotlib.patches.Rectangle((lims[0],lims[0]),
                                    lims[1] - lims[0], lims[1] - lims[0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
plt.savefig(f'{plotdir}/mae_trunc_spectral_{param_string}.png', dpi=400)


fig,ax = plt.subplots(1)
plt.scatter(X_test_sp[test_idcs_leftout_region,var1],
            X_test_sp[test_idcs_leftout_region,var2],
            c = mae_trunc[test_idcs_leftout_region], vmin=vmin, vmax=vmax,
            )
plt.title('trunc network')
plt.colorbar()
rect = matplotlib.patches.Rectangle((lims[0],lims[0]),
                                    lims[1] - lims[0], lims[1] - lims[0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
plt.savefig(f'{plotdir}/mae_trunc_spectral_trunc_test_only{param_string}.png', dpi=400)


# plot diff

fig,ax = plt.subplots(1)
diff = mae_trunc - mae_full
vmax = np.max(np.abs(diff))
vmin = -vmax
plt.scatter(X_test_sp[:,var1],
            X_test_sp[:,var2],
            c = diff,
            cmap = plt.cm.RdBu_r,
            vmax=vmax, vmin=vmin
            )
plt.title('trunc-full')
plt.colorbar()
rect = matplotlib.patches.Rectangle((lims[0],lims[0]),
                                    lims[1] - lims[0], lims[1] - lims[0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
plt.savefig(f'{plotdir}/mae_diff_spectral_{param_string}.png', dpi=400)

fig,ax = plt.subplots(1)
diff = mae_trunc - mae_full
vmax = np.max(np.abs(diff))
vmin = -vmax
plt.scatter(X_test_sp[test_idcs_leftout_region,var1],
            X_test_sp[test_idcs_leftout_region,var2],
            c = diff[test_idcs_leftout_region],
            cmap = plt.cm.RdBu_r,
            vmax=vmax, vmin=vmin
            )
plt.title('trunc-full')
plt.colorbar()
rect = matplotlib.patches.Rectangle((lims[0],lims[0]),
                                    lims[1] - lims[0], lims[1] - lims[0],
                                    linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xlabel(f'spectral coef {var1}')
plt.ylabel(f'spectral coef {var2}')
plt.savefig(f'{plotdir}/mae_diff_spectral_test_only{param_string}.png', dpi=400)

logfile.close()
