import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# % matplotlib inline
plt.style.use('seaborn-whitegrid')

#import tensorflow as tf


def make_circle_data(N, r_min, r_max):
    theta = 2.*np.pi*np.random.random(N)
    r = r_min + (r_max-r_min)*np.random.random(N)
    X = np.zeros((N, 2))
    X[:,0] = r*np.sin(theta)
    X[:,1] = r*np.cos(theta)
    return X

def make_circle_dataset(N, r=1):
    N1 = N//2
    N2 = N - N1
    X, y = np.zeros((N, 2)), np.zeros(N)
    X[:N1,:] = make_circle_data(N1, 0., 1./3.)
    X[N1:,:] = make_circle_data(N2, 2./3., 1.)
    y[N1:] = np.ones(N2)
    X = X * r
    return X, y



def make_square_data(N, x1_min, x1_max, x2_min, x2_max):
    X = np.zeros((N, 2))
    X[:,0] = x1_min + (x1_max-x1_min)*np.random.random(N)
    X[:,1] = x2_min + (x2_max-x2_min)*np.random.random(N)
    return X

def make_xor_dataset(N, r=1.):
    X, y = np.zeros((N, 2)), np.zeros(N)
    X[:N//4,:] = make_square_data(N//4, 0., 1., 0., 1.)
    X[N//4:N//2,:] = make_square_data(N//4, 0., -1., 0., -1.)

    X[N//2:3*N//4,:] = make_square_data(N//4, 0., -1., 0., 1.)
    X[3*N//4:,:] = make_square_data(N//4, 0., 1., 0., -1.)
    y[N//2:] = np.ones((N//2))

    X = X * r
    return X, y





# see https://3diagramsperpage.wordpress.com/2014/05/25/arrowheads-for-axis-in-matplotlib/
def arrowed_plot(ax, arrow_scale=1.0, labels=('x1', 'x2')):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)

    ax.grid(False)

    # removing the axis ticks
    ax.set(xticks=[], yticks=[])
    ax.xaxis.set_ticks_position('none')  # tick markers
    ax.yaxis.set_ticks_position('none')

    # manual arrowhead width and length
    xhw = 0.02 * (xmax - xmin)
    xhl = 0.02 * (ymax - ymin)
    yhw = 0.02 * (xmax - xmin)
    yhl = 0.02 * (ymax - ymin)
    lw = 1.  # axis line width
    ohg = 0.3  # arrow overhang

    # draw x and y axis
    ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw,
             head_width=xhw, head_length=xhl, overhang=ohg,
             length_includes_head=True, clip_on=False)

    ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw,
             head_width=yhw, head_length=yhl, overhang=ohg,
             length_includes_head=True, clip_on=False)

    left = 0.
    right = 1.
    top = 1.
    bottom = 0.

    ax.text(right, 0.47 * (bottom + top), labels[0],
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=14)

    ax.text(0.53 * (left + right), top, labels[1],
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=14)

    ax.text(0.53 * (left + right), 0.47 * (bottom + top), '0',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=14)


def plot_hyperplane(ax, w1, w2, b, xlim=(-4, 4), ylim=(-4, 4), label=False):
    ax.set(xlim=xlim, ylim=ylim)
    arrowed_plot(ax)
    x1 = np.linspace(-8, 8, 100)
    x2 = (w1 * x1 + b) / -w2
    ax.plot(x1, x2, c='r', lw=4)
    if label:
        ax.set_xlabel('{:.2f}*x1 + {:.2f}*x2 + {:.2f} = 0'.format(w1, w2, b), fontsize=14)

import plotly.graph_objects as go

def plot_activation_function_plotly(f, df=None, ax=None, name='', xlim=(-5, 5), ylim=(-1, 1), labels=('x1', 'x2')):
    import numpy as np
    x = np.linspace(*xlim, 100)
    y = f(x)

    line_y = go.Scatter(x=x, y=y,
                       mode='lines',
                       name=labels[0])
    data = [line_y]
    if df is not None:
        grad = df(x)
        line_dy = go.Scatter(x=x, y=grad,
                            mode='lines',
                            name=labels[1])
        data.append(line_dy)

    layout = go.Layout(title=name, width=500, height=400,  xaxis_title="x",
    yaxis_title="f(x)",)

    figure = go.Figure(data=data, layout=layout)
    figure.show(config={'displayModeBar': False })
    # return figure


def plot_activation_function(f, df=None, ax=None, name='', xlim=(-5, 5), ylim=(-1, 1), labels=('x1', 'x2')):
    if ax is None:
        fig, ax = plt.subplots()

    x = np.linspace(*xlim, 100)
    y = f(x)

    if df is not None:
        grad = df(x)
        ax.plot(x, grad, c='r', lw=2)

    ax.set(xlim=xlim, ylim=ylim)
    ax.set_xlabel(name, fontsize=14)
    arrowed_plot(ax, labels=labels)
    ax.plot(x, y, c='b', lw=2, label=f(x))


def plot_dataset_2d(X, y, ax=None, title='', x1lim=(-3,3), x2lim=(-3,3), show_legend=True, fontsize=12):
    """Plot dataset with in an arrowed plot."""
    y = y.flatten()

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set(xlim=x1lim, ylim=x2lim)
    arrowed_plot(ax)
    for i in np.unique(y):
        idx = (y == i)
        ax.scatter(X[idx, 0], X[idx, 1], label='y={}'.format(i))
    if show_legend:
        plt.legend()
    ax.set_title(title, fontsize=fontsize, pad=20)
    return ax


def plot_model_output_2d(ax, model, x1_min, x1_max, x2_min, x2_max, add_features=None):
    xx1 = np.linspace(x1_min, x1_max, 100)
    xx2 = np.linspace(x2_min, x2_max, 100)
    xx1, xx2 = np.meshgrid(xx1, xx2)

    x1x2 = np.concatenate((xx1.ravel().reshape(-1, 1), xx2.ravel().reshape(-1, 1)), axis=1)
    if add_features == None:
        z = model.predict(x1x2).reshape(xx1.shape)
    else:
        z = model.predict(add_features(x1x2)).reshape(xx1.shape)

    ax.imshow(z[::-1, :], zorder=0, extent=[x1_min, x1_max, x2_min, x2_max], cmap='coolwarm', alpha=0.25)


def plot_model_and_data_2d(model, X_train, y_train, X_test, y_test, xlim=(-3, 3), ylim=(-3, 3), add_features=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].set(xlim=xlim, ylim=ylim)
    ax[1].set(xlim=xlim, ylim=ylim)

    arrowed_plot(ax[0])
    arrowed_plot(ax[1])

    for i in np.unique(y_train):
        idx = (y_train[:, 0] == i)
        ax[0].scatter(X_train[idx, 0], X_train[idx, 1], label='y={}'.format(i))

        idx = (y_test[:, 0] == i)
        ax[1].scatter(X_test[idx, 0], X_test[idx, 1], label='y={}'.format(i))
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Training set', fontsize=14)
    ax[1].set_title('Test set', fontsize=14)

    plot_model_output_2d(ax[0], model, xlim[0], xlim[1], ylim[0], ylim[1], add_features)
    plot_model_output_2d(ax[1], model, xlim[0], xlim[1], ylim[0], ylim[1], add_features)

    plt.tight_layout()

    return fig, ax


def plot_model_layer_2d(ax, model, layer, node, x1_min, x1_max, x2_min, x2_max, feature_function=None):
    xx1 = np.linspace(x1_min, x1_max, 100)
    xx2 = np.linspace(x2_min, x2_max, 100)
    xx1, xx2 = np.meshgrid(xx1, xx2)

    x1x2 = np.concatenate((xx1.ravel().reshape(-1, 1), xx2.ravel().reshape(-1, 1)), axis=1)
    if feature_function == None:
        z = model.predict_layer(x1x2, layer)[:, node].reshape(xx1.shape)
    else:
        z = model.predict_layer(feature_function(x1x2), layer)[:, node].reshape(xx1.shape)

    ax.imshow(z[::-1, :], zorder=0, extent=[x1_min, x1_max, x2_min, x2_max], cmap='coolwarm', alpha=0.25)


import matplotlib.gridspec as gridspec


def plot_weights_and_data_2d(model, X, y):
    n_layers = len(model.layers)
    n_max_nodes = max([model.layers[k][3] for k in range(n_layers)]) + 2

    max_Wb = max([np.abs(model.get_weights(k)[0]).max() for k in range(n_layers)] +
                 [np.abs(model.get_weights(k)[0]).max() for k in range(n_layers)])

    size = 3
    fig = plt.figure(figsize=(n_max_nodes * size, n_layers * size))
    gs = gridspec.GridSpec(n_layers, n_max_nodes)

    for k in range(n_layers):
        for j in range(model.layers[k][3]):
            ax = plt.subplot(gs[k, j])
            ax.set(xlim=(-5, 5), ylim=(-5, 5))
            arrowed_plot(ax)

            for i in np.unique(y):
                idx = (y[:, 0] == i)
                ax.scatter(X[idx, 0], X[idx, 1], label='y={}'.format(i), s=4.0)

            plot_model_layer_2d(ax, model, k, j, -5, 5, -5, 5)

            if j == 0:
                ax.set_ylabel('layer {}'.format(k))

            ax.set_title('node {}'.format(j))

        # Plot weights
        W, b = model.get_weights(k)
        ax = plt.subplot(gs[k, j + 1])
        im = ax.imshow(W, clim=(-max_Wb, max_Wb), cmap='PRGn')
        ax.grid(False)
        ax.set_title('weights')
        ax.set_xlabel('node')
        ax.set_ylabel('input')
        ax.set(xticks=range(W.shape[1]), yticks=range(W.shape[0]))
        fig.colorbar(im, ax=ax, orientation='horizontal')

        # Plot bias
        ax = plt.subplot(gs[k, j + 2])
        im = ax.imshow(b.reshape(1, W.shape[1]), clim=(-max_Wb, max_Wb), cmap='PRGn')
        ax.grid(False)
        ax.set_title('bias')
        ax.set_xlabel('node')
        ax.set_ylabel('bias')
        ax.set(xticks=range(W.shape[1]), yticks=[])
        fig.colorbar(im, ax=ax, orientation='horizontal')

        plt.tight_layout()


def plot_elements(history, elems, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))  # create a figure object
    for key in elems:
        stat = history[key]
        sns.lineplot(np.arange(len(stat)), stat, label=key, ax=ax)
    return ax

def plot_keras_loss(keras_loss,
                    true_label=0,
                    ax=None,
                    x_range=(-1, 1),
                    y_range=None,
                    name='',
                    **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    number_of_points = 200
    y_true = np.full((number_of_points, ), fill_value=true_label)
    y_pred = np.linspace(*x_range, number_of_points)
    loss = [keras_loss([true], [prediction], **kwargs).numpy() for true, prediction in zip(y_true, y_pred)]
    gradient = np.gradient(loss, y_pred)
    sns.lineplot(y_pred, loss, ax=ax, linewidth=5, label='Loss', color='b', alpha=0.7)
    sns.lineplot(y_pred, gradient, ax=ax, linewidth=5, label='Loss Derivative', color='r', alpha=0.7)
    if y_range is not None:
        plt.ylim(*y_range)
    plt.legend(loc=2, prop={'size': 15})
    plt.title(name, fontdict=dict(size=15))
    return ax

def plot_keras_activation(keras_activation, ax=None, x_range=(-1, 1)):
    if ax is None:
        fig, ax = plt.subplots()
    x = np.linspace(*x_range, 100)
    activation = [keras_activation(value).numpy() for value in x]
    sns.lineplot(x, activation, ax=ax, linewidth=5, label='loss', c='b')
    return ax