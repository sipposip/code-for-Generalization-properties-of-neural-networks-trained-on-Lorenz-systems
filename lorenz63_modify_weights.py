
import os
import matplotlib
matplotlib.use('agg')
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from keras import backend as K

import keras
from keras import backend as K
import tensorflow as tf
import keras
# limit maximum number of CPUs
config = tf.ConfigProto(intra_op_parallelism_threads=20, inter_op_parallelism_threads=20,
                        allow_soft_placement=True)
session = tf.Session(config=config)
K.set_session(session)


plt.rcParams['savefig.bbox'] = 'tight'

plotdir = 'plots_modify_weights'
os.system(f'mkdir -p {plotdir}')


beta = 8. / 3.
rho = 28
sigma = 10


def lorenz(X, t=0):
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    dXdt = [dxdt, dydt, dzdt]
    return dXdt


def make_lorenz_run(y0, Nsteps, tstep):
    t = np.arange(0, Nsteps) * tstep
    sol = odeint(lorenz, y0, t)
    return sol


Nsteps = 100000
tstep = 0.01
modelrun1 = make_lorenz_run([8, 1, 1], Nsteps, tstep)
# model run has diension (time,variable), where variable is [x,y,z]
# normalize
modelrun1 = (modelrun1 - modelrun1.mean(axis=0)) / modelrun1.std(axis=0)


def lorenz3dplot_lines(data, title="", **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5, **kwargs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_title(title)
    return sc


def lorenz3dplot_scatter(data, title="", cmap=plt.cm.gist_heat_r, **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], lw=0.5, cmap=cmap,**kwargs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_title(title)
    return sc


def compute_3d_density(data):
    extent = (-3, 3.01)
    spacing = 0.3
    bins = np.arange(extent[0], extent[1], spacing)
    hist3d = np.histogramdd(data, bins=[bins, bins, bins], normed=True)

    return hist3d


n_epoch = 100
lead_time = 1  # in steps



# parameters from tuning
hidden_size = 128
n_hidden=2
lr=3e-05
selection_strategy='density'

param_string = '_'.join([str(e) for e in (Nsteps, tstep, n_epoch, lead_time, n_hidden, hidden_size)])

def train_network(X_train, y_train, selection_strategy, reference_clim = None, n_iter=10):
    """
    traina couple of networsk, and returned the best one
    :param X_train:
    :param y_train:
    :param selection_strategy: one of  'density', 'no_equal_points', 'no_fixpoint','density-full'
    :param reference_clim: reference timeseries used for 'density-full'. igonred for other selection strategies
    :return: trained keras network
    """

    # repeat training n_iter times
    res = []
    for _ in range(n_iter):

        # build and train network
        layers = []
        for _ in range(n_hidden):
            layers.append(keras.layers.Dense(hidden_size, input_shape=(3,), activation='relu'))

        layers.append(keras.layers.Dense(3, activation='linear'))
        network = keras.Sequential(layers)
        print(network.summary())

        optimizer = keras.optimizers.Adam(lr=lr)
        network.compile(optimizer=optimizer, loss='mean_squared_error')

        network.fit(X_train, y_train, epochs=n_epoch, verbose=2, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             min_delta=0,
                                                             patience=4,

                                                             verbose=1, mode='auto')]
                    )
        # now make a climate run with the network,  and check whether it does
        # meet the selection criterion in selection_strategy
        clim = make_network_climrun(network, X_train)

        if selection_strategy == 'no_fixpoint':
            # test whether there are adjactants states that are identical down to machine precision
            diffs = np.diff(clim,axis=0)
            # check whether there is a point where diff is zero for all variables
            if not np.any(np.all(diffs==0, axis=1)):
               res = network
               break
            else:
                print('reconstructed attractor collapsed, repeating training')
                continue

        elif selection_strategy == 'no_equal_points':
            # test whether there are points that are identical down to machine precision
            # this we can do by comparing the lenght of clim with the unique values (along axis=0) in clim
            if len(clim) == len(np.unique(clim, axis=0)):
               res = network
               break
            else:
                print('reconstructed attractor collapsed, repeating training')
                continue

        elif selection_strategy == 'density':
            # test whether the difference in 3d density is below a threshold given by dens_thresh
            dens_model, _ = compute_3d_density(X_train)
            dens_network, _ = compute_3d_density(clim)

            rmse_dens = np.sqrt(np.mean((dens_network - dens_model) ** 2))
            res.append([rmse_dens,network])

        elif selection_strategy == 'density-full':
            # test whether the difference in 3d density is below a threshold given by dens_thresh,
            # by using a reference climate (from the full network)
            dens_model, _ = compute_3d_density(reference_clim)
            dens_network, _ = compute_3d_density(clim)

            rmse_dens = np.sqrt(np.mean((dens_network - dens_model) ** 2))
            res.append([rmse_dens, network])



        else:
            raise ValueError(f'selection strategy {selection_strategy} not implemented')


    if selection_strategy in ('density', 'density-full'):
        # select the newtork with best attractor density error
        dens_errors = [e[0] for e in res]
        min_idx = np.argmin(dens_errors)
        network = res[min_idx][1]
        return  network

    elif selection_strategy in ('no_fixpoint', 'no_equal_points'):
        if network == []:
            print('no suitable network found')
            return None

    else:
        raise ValueError(f'selection strategy {selection_strategy} not implemented')


def make_network_climrun(model, test_data):
    # crreate a "climate run" with the network

    N_clim = Nsteps//2
    # as initial state, use a random state from the test data
    y_init_clim = test_data[np.random.randint(len(test_data))]

    network_clim_run = np.zeros((N_clim + 1, 3))
    network_clim_run[0] = y_init_clim
    state = y_init_clim
    state = np.expand_dims(state, axis=0)
    for i in range(N_clim):
        # print(i,N_clim)
        state = model.predict(state)
        network_clim_run[i + 1] = state

    return network_clim_run


def get_activations(model, X_batch):
    ''' activations of all hidden layers (=all layers except last one)'''
    n_layers = len(model.layers)
    n_hidden = n_layers - 1
    activs_per_layer = []
    for i in range(n_hidden):
        get_activations_ = K.function([model.layers[0].input, K.learning_phase()], [model.layers[i].output,])
        activations = get_activations_([X_batch,0])
        activs_per_layer.append(activations[0])

    return activs_per_layer


Nsteps = 100000
tstep = 0.01
modelrun1 = make_lorenz_run([8, 1, 1], Nsteps, tstep)
# model run has diension (time,variable), where variable is [x,y,z]
# normalize
modelrun1 = (modelrun1 - modelrun1.mean(axis=0)) / modelrun1.std(axis=0)

# split in first and second half
train_data_full = modelrun1[:int(Nsteps / 2)]
test_data_full = modelrun1[int(Nsteps / 2):]

# X and y data are the same, but shifted by lead-time

X_train = train_data_full[:-lead_time]
y_train = train_data_full[lead_time:]

X_test = test_data_full[:-lead_time]
y_test = test_data_full[lead_time:]

network_full = train_network(X_train, y_train, selection_strategy=selection_strategy)





#%% plot distribution af activations for different regions of the phase space


lim_left  =   [[-3,0],[-3,3],[-3,3]]  # left wing
lim_right  =  [[0,3],[-3,3],[-3,3]] # right wing
wing_indcs = {}

for name, lims in (('left',lim_left), ('right',lim_right)):
    idcs = (X_train[:,0] > lims[0][0]) & (X_train[:,0] < lims[0][1])  & \
           (X_train[:,1] > lims[1][0]) & (X_train[:,1] < lims[1][1])  & \
           (X_train[:,2] > lims[2][0]) & (X_train[:,2] < lims[2][1])

    wing_indcs[name] = idcs
    fig = plt.figure(figsize=(7, 7))
    ax1=ax = fig.gca(projection='3d')
    ax1.set_xlim(-2.5,2.5)
    ax1.set_ylim(-2.5,2.5)
    ax1.set_zlim(-2.5,2.5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    sc = ax1.scatter(X_train[idcs, 0], X_train[idcs, 1], X_train[idcs, 2], lw=0.5)
    plt.savefig(f'{plotdir}/region_activation_lims{lims}_3d.png')

    fig = plt.figure(figsize=(7, 7))
    ax2 = plt.gca()
    ax2.set_xlabel('neuron')
    ax2.set_ylabel('activation')

wing_indcs_test = {}
for name, lims in (('left',lim_left), ('right',lim_right)):
    idcs = (X_test[:,0] > lims[0][0]) & (X_test[:,0] < lims[0][1])  & \
           (X_test[:,1] > lims[1][0]) & (X_test[:,1] < lims[1][1])  & \
           (X_test[:,2] > lims[2][0]) & (X_test[:,2] < lims[2][1])
    wing_indcs_test[name] = idcs

# get the activations of the hidden layer for all X_train
activs= get_activations(network_full,X_train)
#  shape is (n_layer,n_time,n_neuron)
# since in general, n_neuron can be different for different layers, we cannot reshape to
# (n_time, n_layer, n_neuron)


activs_left = [layer_activs[wing_indcs['left']] for layer_activs in activs]
activs_right =[layer_activs[wing_indcs['right']] for layer_activs in activs]


# boxplot with both wings
for i in range(n_hidden):
    fig, ax = plt.subplots(1,figsize=(15,4))
    pos = np.arange(activs_left[i].shape[1])
    bp1 = plt.boxplot(activs_left[i], positions=pos+0.5, widths=0.2, showfliers=False)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp1[element], color='#1b9e77')
    bp2 = plt.boxplot(activs_right[i], positions=pos, widths=0.2, showfliers=False)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp2[element], color='#d95f02')

    ax.set_xticklabels([])
    plt.ylabel('activation')
    plt.xlabel(f'neuron in layer {i+1}')
    sns.despine()
    # manually add legend
    plt.legend(handles=[mpatches.Patch(color='#1b9e77', label='left wing'),
                        mpatches.Patch(color='#d95f02', label='right wing')])
    plt.savefig(f'{plotdir}/region_activation_lims_bothwings_layer{i}.pdf')


    # closup, ordered by difference in spread in activation
    std_activs_left = np.std(activs_left[i], axis=0)
    std_activs_right = np.std(activs_right[i], axis=0)
    std_activs_diff = np.abs(std_activs_right - std_activs_left)
    order_idcs = np.argsort(std_activs_diff)[::-1]

    # plot not all, but only the first 20
    order_idcs = order_idcs[:20]

    fig, ax = plt.subplots(1, figsize=(7, 4))
    pos = np.arange(activs_left[i][:,order_idcs].shape[1])
    bp1 = plt.boxplot(activs_left[i][:,order_idcs], positions=pos + 0.3, widths=0.2, showfliers=False)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color='#1b9e77')
    bp2 = plt.boxplot(activs_right[i][:,order_idcs], positions=pos, widths=0.2, showfliers=False)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp2[element], color='#d95f02')

    plt.xticks(np.arange(len(order_idcs))+0.25, labels=np.arange(len(order_idcs)))
    plt.ylabel('activation')
    plt.xlabel(f'neuron in layer {i + 1}, ordered')
    sns.despine()
    # manually add legend
    plt.legend(handles=[mpatches.Patch(color='#1b9e77', label='left wing'),
                        mpatches.Patch(color='#d95f02', label='right wing')])
    plt.savefig(f'{plotdir}/region_activation_lims_bothwings_layer{i}_ordered_closeup.pdf')



#%% now selectively switch off neurons and make predictions



 #number of neurons to deactivate
 # loop over n_deactivate, and store the result per wing
res_left = {'left':[], 'right':[]}
res_right = {'left':[], 'right':[]}
list_n_deactivate = [1,2, 3, 4, 5, 6, 8, 10, 20, 40, 60, 80, 100]
for n_deactivate in list_n_deactivate:

    for wing in ('left', 'right'):

        # copy the network
        network_modified = keras.models.clone_model(network_full)
        network_modified.build((None,3))
        network_modified.compile(optimizer='adam', loss='mean_squared_error')
        network_modified.set_weights(network_full.get_weights())


        # loop over layers, and in each layer switch off the neuron with least activity.
        # for this we fix the ouptut of that neuron. this can be done via setting the weights
        # from this neuron to all neruoan in the NEXT layer to zero, and setting the bias to the fixed value
        for i_layer in range(n_hidden):
            activs_layer = activs[i_layer]
            # i_layer +1 because we fix th weights of the NEXT layer
            weights_orig,bias_orig = network_full.layers[i_layer+1].get_weights()

            # now switch off the neuron with lowest activity (activity==variance of activations).
            # however, we also have "dead" neurons that have not activity at all in any wing. these we have to esclude first
            neuron_dead = np.where(activs_layer.std(axis=0)==0)[0]
            n_dead = len(neuron_dead)


            activ_std_wing =  activs_layer[wing_indcs[wing]].std(axis=0)

            # get the n_deactivate + n_neuron_dead smallest ones
            min_idcs_with_dead = np.argsort(activ_std_wing)[:n_deactivate + n_dead]
            # remove the indices that correspond to the dead neurons
            min_idcs = [i for i in min_idcs_with_dead if i not in neuron_dead]
            # it can happen that we now have more than n_deactivate indices. This happens when there
            # are many neurons that are "dead" only in the current wing. In this case we simply select
            # n_deactivate of them
            if len(min_idcs) > n_deactivate:
                min_idcs = min_idcs[:n_deactivate]
            assert(len(min_idcs) == n_deactivate)

            weights_new = weights_orig.copy()
            bias_new = bias_orig.copy()
            for fixed_neuron_idx in min_idcs:
                # set weights to zero
                weights_new[fixed_neuron_idx,:] = 0

                # no we have to compensate what the neuron with low variation contributed to the output
                # for this we add the mean output of the neuron in this wing, multiplied by the (original) weight form
                # # hidden to putpt to each bias in the output
                # note that with relu activation functions, this seems to be always zero anywary.
                mean_output_fixed_neuron = activs_layer[wing_indcs[wing], fixed_neuron_idx].mean()
                bias_new = bias_new + weights_orig[fixed_neuron_idx] * mean_output_fixed_neuron

            network_modified.layers[i_layer+1].set_weights([weights_new,bias_new])

        pred_full = network_full.predict(X_test)
        abserr_full = np.abs(pred_full - y_test).mean(axis=1)
        pred_mod = network_modified.predict(X_test)
        abserr_mod = np.abs(pred_mod - y_test).mean(axis=1)

        # for the case of 20 deactivated neuron, we fix the colorscale so that both wings have same colorscale (we need
        # this for the paper)
        if n_deactivate == 20:

            sc = lorenz3dplot_scatter(X_test, cmap=plt.cm.gist_heat_r, c=abserr_mod, vmin=0, vmax=0.9,
                                  title=f'deactivated {n_deactivate} neuron(s) per layer with least activity in {wing} wing')

        else:
            sc = lorenz3dplot_scatter(X_test, cmap=plt.cm.gist_heat_r, c=abserr_mod,
                                      title=f'deactivated {n_deactivate} neuron(s) per layer with least activity in {wing} wing')

        cb = plt.colorbar(sc)
        cb.set_label("MAE")
        plt.savefig(f'{plotdir}/modified_weights_abs_error_mod_{wing}wing_minneuron_{param_string}_n_deactivate{n_deactivate}.png')

        # plot difference between error of normal and of modified net
        sc = lorenz3dplot_scatter(X_test, cmap=plt.cm.gist_heat_r, c=abserr_mod-abserr_full,# vmin=0, vmax=1,
                                  title=f'deactivated {n_deactivate} neuron(s) per layer with least activity in {wing} wing')
        cb = plt.colorbar(sc)
        cb.set_label("MAE")
        plt.savefig(
            f'{plotdir}/modified_weights_abs_error_diff_{wing}wing_minneuron_{param_string}_n_deactivate{n_deactivate}.png')


        error_leftwing = np.mean(abserr_mod[wing_indcs_test['left']])
        error_rightwing = np.mean(abserr_mod[wing_indcs_test['right']])
        if wing == 'left':
            res_left['left'].append(error_leftwing)
            res_left['right'].append(error_rightwing)

        else:
            res_right['left'].append(error_leftwing)
            res_right['right'].append(error_rightwing)

plt.figure(figsize=(7,3))
plt.axhline(abserr_full[wing_indcs_test['left']].mean(),linestyle='--', label='left wing unmodified ', color='#1b9e77')
plt.axhline(abserr_full[wing_indcs_test['right']].mean(),linestyle=':', label='right wing unmodified ', color='#d95f02')
plt.plot(list_n_deactivate, res_left['left'],'-x', label='left wing modified', color='#1b9e77')
plt.plot(list_n_deactivate, res_left['right'], '-x',  label='right wing modified', color='#d95f02')
plt.xlabel('number of neurons with low activity \n in left wing deactivated')
plt.ylabel('mae')
plt.legend()
sns.despine()
plt.ylim((0, 1.2))
plt.savefig(f'{plotdir}/n_deactivate_vs_mae_{param_string}.svg')


plt.figure(figsize=(7,3))
plt.axhline(abserr_full[wing_indcs_test['left']].mean(),linestyle='--', label='left wing unmodified ', color='#1b9e77')
plt.axhline(abserr_full[wing_indcs_test['right']].mean(),linestyle=':', label='right wing unmodified ', color='#d95f02')
plt.plot(list_n_deactivate, res_right['left'],'-x', label='left wing modified', color='#1b9e77')
plt.plot(list_n_deactivate, res_right['right'], '-x',  label='right wing modified', color='#d95f02')
plt.xlabel('number of neurons with low activity \n in right wing deactivated')
plt.ylabel('mae')
plt.legend()
sns.despine()
plt.ylim((0, 1.2))
plt.savefig(f'{plotdir}/n_deactivate_vs_mae_{param_string}_right.svg')