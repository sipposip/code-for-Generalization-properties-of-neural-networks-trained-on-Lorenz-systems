"""
python3


@author: Sebastian Scher, march 2019
"""
import os
import matplotlib
matplotlib.use('agg')
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import keras

os.system('mkdir -p plots_forcing')


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


def make_lorenz_run(y0, t_arr, sigma_arr):
    res = np.zeros((len(t_arr+1),4))
    res[0] = np.array([*y0,sigma_arr[0]])
    sol = y0
    for i in range(len(t_arr)):
        sigma = sigma_arr[i]
        # print(i)
        # we make only one step, but we gert a 2d array back. 9with only one element)
        # therefore, we extract this element with [0]
        sol = odeint(lorenz, sol, t=[0,tstep], args=(sigma,))[1]
        res[i] = np.array([*sol, sigma])
    return res


def predict_with_lorenz(X,sigma):
    preds = []
    for x in X:
        pred = odeint(lorenz, x, t=[0,tstep * lead_time], args=(sigma,))[1]
        preds.append(pred)
    return np.array(preds)


# paramters for experiments

Nsteps = 10000000 / 2
tstep = 0.01
hidden_size = 8  # 15 in https://sci-hub.tw/10.1109/CIMSA.2005.1522829
# 8 in https://www.researchgate.net/publication/322809990_Artificial_neural_networks_model_design_of_Lorenz_chaotic_system_for_EEG_pattern_recognition_and_prediction
n_epoch = 30
lead_time = 1  # in steps
t_arr = np.arange(0, Nsteps) * tstep
sigma_start = 10
sigma_end = 20
#sigma_arr = sigma_start * np.ones(len(t_arr)) + (sigma_end - sigma_start)/Nsteps/tstep*t_arr
n_sigma_steps = 20

#exptype='steps'
exptype='linear'

if exptype == 'steps':
    sigma_arr = np.array([ sigma_start * np.ones(len(t_arr)//n_sigma_steps) + (sigma_end - sigma_start)/n_sigma_steps * i
                  for i in range(n_sigma_steps)  ]).flatten()
elif exptype == 'linear':
    sigma_arr = sigma_start * np.ones(len(t_arr)) + (sigma_end - sigma_start) / Nsteps / tstep * t_arr

N_start_perc = 20  # fraction to be used of the modelrun for training, both from the start and the end of the rn
N_start = int(Nsteps * N_start_perc/100)


# we make two runs, started with slightly different initial conditions
# one will be the training and one the test run
modelrun_train = make_lorenz_run([1, 1, 1], t_arr, sigma_arr)
modelrun_test = make_lorenz_run([1.1, 1, 1], t_arr, sigma_arr)

# model run has diension (time,variable), where variable is [x,y,z]
# normalize with train run
norm_mean = modelrun_train.mean(axis=0)
norm_std = modelrun_train.std(axis=0)
modelrun_train = (modelrun_train  - norm_mean) / norm_std
modelrun_test = (modelrun_test - norm_mean) / norm_std

sigma_arr_normed = modelrun_train[:,3]


def lorenz3dplot_lines(data, title="", **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5, **kwargs)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    return sc


def lorenz3dplot_scatter(data, title="", **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], lw=0.5, **kwargs)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    return sc

def compute_3d_density(data):
    extent = (-3, 3.01)
    spacing = 0.3
    bins = np.arange(extent[0], extent[1], spacing)
    hist3d = np.histogramdd(data, bins=[bins, bins, bins])

    return hist3d




def train_network(X_train, y_train):
    network = keras.Sequential([
        keras.layers.Dense(hidden_size, input_shape=(4,), activation='sigmoid'),

        keras.layers.Dense(3, activation='linear')
    ])

    print(network.summary())

    network.compile(optimizer='adam', loss='mean_squared_error')

    network.fit(X_train, y_train, epochs=n_epoch, verbose=2, validation_split=0.1)

    return network

def train_network_noforcing(X_train, y_train):
    network = keras.Sequential([
        keras.layers.Dense(hidden_size, input_shape=(3,), activation='sigmoid'),

        keras.layers.Dense(3, activation='linear')
    ])

    print(network.summary())

    network.compile(optimizer='adam', loss='mean_squared_error')

    network.fit(X_train, y_train, epochs=n_epoch, verbose=2, validation_split=0.1)

    return network


def make_network_climrun(network, y_init, sigma_arr):
    # crreate a "climate run" with the network

    N_clim = len(sigma_arr)

    network_clim_run = np.zeros((N_clim , 3))
    y_init = y_init[:3]
    network_clim_run[0] = y_init
    state = y_init
    state = np.expand_dims(state, axis=0)
    for i in range(N_clim-1):
        # add sigma
        state_with_sigma = np.concatenate([state, sigma_arr[i][np.newaxis, np.newaxis]], axis=1)
        state = network.predict(state_with_sigma)
        network_clim_run[i + 1] = state

    return network_clim_run


# X and y data are the same, but shifted by lead-time, and for y we remove sigma (4th variable]
X_train_full = modelrun_train[:-lead_time]
y_train_full = modelrun_train[lead_time:][:,:3]

X_test_full = modelrun_test[:-lead_time]
y_test_full = modelrun_test[lead_time:][:,:3]


# split up the simulations in a start and a end (and a discarded middle) part.

X_train_start = X_train_full[:N_start]
y_train_start = y_train_full[:N_start]
X_test_start = X_test_full[:N_start]
y_test_start = y_test_full[:N_start]

X_train_end = X_train_full[-N_start:]
y_train_end = y_train_full[-N_start:]
X_test_end = X_test_full[-N_start:]
y_test_end = y_test_full[-N_start:]

sigma_arr_start = sigma_arr_normed[:N_start]
sigma_arr_end = sigma_arr_normed[-N_start:]

N_training = 10


# now loop over relizations (we do the same training a couple of times

res_training_loop = []
for i in range(N_training):

    # parameters string for filenames
    param_string = '_'.join(
        [str(e) for e in (Nsteps, tstep, n_epoch, lead_time, sigma_start, sigma_end, N_start_perc, name, exptype,
                          'training'+str(i).zfill(3))])

    # now train a network on the start part
    network_start = train_network(X_train_start, y_train_start)
    # and make a climate simulation with the network with the start forcing
    y_init_clim_start = X_test_start[np.random.randint(N_start)]
    network_clim_start_on_start = make_network_climrun(network_start,y_init_clim_start, sigma_arr_start)

    # ..... and with the end forcing
    y_init_clim_end = X_test_end[np.random.randint(N_start)]
    network_clim_start_on_end = make_network_climrun(network_start,y_init_clim_end, sigma_arr_end)

    # now the same, but trained on the end
    network_end = train_network(X_train_end, y_train_end)
    network_clim_end_on_start = make_network_climrun(network_end,y_init_clim_start, sigma_arr_start)
    network_clim_end_on_end = make_network_climrun(network_end,y_init_clim_end, sigma_arr_end)


    # and now train a network that does not use the sigma-input
    network_base_start = train_network_noforcing(X_train_start[:,:3], y_train_start)
    network_base_end = train_network_noforcing(X_train_end[:,:3], y_train_end)

    # plot the climates
    lorenz3dplot_lines(X_train_start[:,:3], 'model startclimate', alpha=0.8)
    plt.savefig(f'plots_forcing/model_startclimate{param_string}.png')

    lorenz3dplot_lines(X_train_end[:,:3], 'model endclimate', alpha=0.8)
    plt.savefig(f'plots_forcing/model_endclimate{param_string}.png')

    lorenz3dplot_lines(network_clim_start_on_start, 'network trained on start on start', alpha=0.8)
    plt.savefig(f'plots_forcing/network_start_on_start{param_string}.png')

    lorenz3dplot_lines(network_clim_start_on_end, 'network trained on start on end', alpha=0.8)
    plt.savefig(f'plots_forcing/network_start_on_end{param_string}.png')

    # now do forecasting with the start-trained network, both on the start and on the end climate
    pred_startnet_on_start = network_start.predict(X_test_start)
    pred_startnet_on_end = network_start.predict(X_test_end)

    abserr_startnet_on_start = np.abs(pred_startnet_on_start - y_test_start).mean(axis=1)
    abserr_startnet_on_end = np.abs(pred_startnet_on_end - y_test_end).mean(axis=1)

    pred_endnet_on_start = network_end.predict(X_test_start)
    pred_endnet_on_end = network_end.predict(X_test_end)

    abserr_endnet_on_start = np.abs(pred_endnet_on_start - y_test_start).mean(axis=1)
    abserr_endnet_on_end = np.abs(pred_endnet_on_end - y_test_end).mean(axis=1)

    pred_startbasenet_on_start = network_base_start.predict(X_test_start[:,:3])
    pred_startbasenet_on_end = network_base_start.predict(X_test_end[:,:3])
    pred_endbasenet_on_start = network_base_end.predict(X_test_start[:,:3])
    pred_endbasenet_on_end = network_base_end.predict(X_test_end[:,:3])

    abserr_startbasenet_on_start = np.abs(pred_startbasenet_on_start - y_test_start).mean(axis=1)
    abserr_startbasenet_on_end = np.abs(pred_startbasenet_on_end - y_test_end).mean(axis=1)
    abserr_endbasenet_on_start = np.abs(pred_endbasenet_on_start - y_test_start).mean(axis=1)
    abserr_endbasenet_on_end = np.abs(pred_endbasenet_on_end - y_test_end).mean(axis=1)

    # predict with the lorenz equations, with the mean sigma of the other epoch
    # for this we need to "un"normalize X_test
    X_test_end_notnormed = X_test_end * norm_std + norm_mean
    X_test_start_notnormed = X_test_start * norm_std + norm_mean
    # we also need to unnormalize sigma_arr, the weights for sigma are the last element of the norm weights
    mean_sigma_start_notnormed = sigma_arr_start.mean() * norm_std[3] + norm_mean[3]
    mean_sigma_end_notnormed = sigma_arr_end.mean() * norm_std[3] + norm_mean[3]


    pred_lorenz_startsigma_on_end = predict_with_lorenz(X_test_end_notnormed[:,:3], sigma=mean_sigma_start_notnormed)
    pred_lorenz_endsigma_on_start = predict_with_lorenz(X_test_start_notnormed[:,:3], sigma=mean_sigma_end_notnormed)
    pred_lorenz_startsigma_on_start = predict_with_lorenz(X_test_start_notnormed[:,:3], sigma=mean_sigma_start_notnormed)
    pred_lorenz_endsigma_on_end = predict_with_lorenz(X_test_end_notnormed[:,:3], sigma=mean_sigma_end_notnormed)
    # now we have to normalize the predictions to get the same MAE as for the network predictions
    pred_lorenz_startsigma_on_end = (pred_lorenz_startsigma_on_end - norm_mean[:3]) / norm_std[:3]
    pred_lorenz_endsigma_on_start = (pred_lorenz_endsigma_on_start - norm_mean[:3]) / norm_std[:3]
    pred_lorenz_startsigma_on_start = (pred_lorenz_startsigma_on_start - norm_mean[:3]) / norm_std[:3]
    pred_lorenz_endsigma_on_end = (pred_lorenz_endsigma_on_end - norm_mean[:3]) / norm_std[:3]

    abserr_lorenz_startsigma_on_end = np.abs(pred_lorenz_startsigma_on_end - y_test_end).mean()
    abserr_lorenz_endsigma_on_start = np.abs(pred_lorenz_endsigma_on_start - y_test_start).mean()
    abserr_lorenz_startsigma_on_start = np.abs(pred_lorenz_startsigma_on_start - y_test_start).mean()
    abserr_lorenz_endsigma_on_end = np.abs(pred_lorenz_endsigma_on_end - y_test_end).mean()

    # plot the weather prediction errors
    sc = lorenz3dplot_scatter(X_test_start, c=abserr_startnet_on_start, title='abserror start')
    plt.colorbar(sc)
    plt.savefig(f'plots_forcing/abserror_startnet_on_start_{param_string}.png')

    sc = lorenz3dplot_scatter(X_test_end, c=abserr_startnet_on_end, title='abserror end')
    plt.colorbar(sc)
    plt.savefig(f'plots_forcing/abserror_startnet_on_end_{param_string}.png')

    #%%
    sns.set_context('notebook', font_scale=1.0, rc={"lines.linewidth": 3.})
    sns.set_palette('colorblind')
    plt.rcParams['savefig.bbox'] = 'tight'

    plt.figure(figsize=(14,10))
    plt.subplot(211)
    plt.plot(t_arr, sigma_arr)
    plt.axvline(N_start*tstep)
    plt.axvline((Nsteps - N_start)*tstep)
    plt.xlabel('t')
    plt.ylabel('sigma')
    sns.despine()
    plt.title(f'Ntrain:{N_start}')
    plt.subplot(212)

    df = pd.DataFrame({'mae':[abserr_startnet_on_start.mean(), abserr_endnet_on_start.mean(),
                              abserr_endnet_on_end.mean(), abserr_startnet_on_end.mean(),
                              abserr_startbasenet_on_start.mean(), abserr_endbasenet_on_start.mean(),
                              abserr_endbasenet_on_end.mean(), abserr_startbasenet_on_end.mean(),
                              abserr_lorenz_endsigma_on_start.mean(),abserr_lorenz_startsigma_on_end.mean(),
                              abserr_lorenz_startsigma_on_start.mean(), abserr_lorenz_endsigma_on_end.mean()

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
                                    'trained_on_start_nosigma', 'trained_on_end_nosigma',
                                    'trained_on_end_nosigma', 'trained_on_start_nosigma',
                                    'lorenz_fixedsigmaend', 'lorenz_fixedsigmastart',
                                    'lorenz_fixedsigmastart', 'lorenz_fixedsigmaend'

                       ]})

    os.system('mkdir -p data')
    df.to_pickle(f'data/error_df_{param_string}.pkl')

    sns.barplot('fc_on', 'mae', hue='network', data=df)
    sns.despine()
    plt.legend(fancybox=True, framealpha=0.5)
    plt.savefig(f'plots_forcing/experiment_result_overview_{param_string}.svg')

    df['itraining'] = i
    res_training_loop.append(df)

res_df = pd.concat(res_training_loop)
res_df = res_df.reset_index()

plt.figure(figsize=(14,10))
plt.subplot(211)
plt.plot(t_arr, sigma_arr)
plt.axvline(N_start*tstep)
plt.axvline((Nsteps - N_start)*tstep)
plt.xlabel('t')
plt.ylabel('sigma')
sns.despine()
plt.title(f'Ntrain:{N_start}')
plt.subplot(212)
sns.barplot('fc_on', 'mae', hue='network', data=res_df,
            ci='sd', # this plots the std of the data as error bars (instead of the standard bootstrapping in seaborn)
           # capsize=0.1
            )
sns.despine()
plt.legend(fancybox=True, framealpha=0.5)
plt.savefig(f'plots_forcing/experiment_result_overview_{param_string}_alltraining.svg')

plt.figure(figsize=(14,10))
plt.subplot(211)
plt.plot(t_arr, sigma_arr)
plt.axvline(N_start*tstep)
plt.axvline((Nsteps - N_start)*tstep)
plt.xlabel('t')
plt.ylabel('sigma')
sns.despine()
plt.title(f'Ntrain:{N_start}')
plt.subplot(212)
sns.boxplot('fc_on', 'mae', hue='network', data=res_df)
sns.despine()
plt.legend(fancybox=True, framealpha=0.5)
plt.savefig(f'plots_forcing/experiment_result_overview_{param_string}_alltraining_boxplot.svg')
