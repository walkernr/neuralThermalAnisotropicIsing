import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_utils import *


def parse_args():
    ''' parses command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output',
                        action='store_true')
    parser.add_argument('-nm', '--name', help='simulation name',
                        type=str, default='run')
    parser.add_argument('-n', '--lattice_sites', help='lattice sites (side length)',
                        type=int, default=81)
    parser.add_argument('-si', '--sample_interval', help='interval for selecting phase points (variational autoencoder)',
                        type=int, default=1)
    parser.add_argument('-sn', '--sample_number', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=512)
    parser.add_argument('-sc', '--scale_data', help='scale data (-1, 1) -> (0, 1)',
                        action='store_true')
    parser.add_argument('-w', '--wasserstein', help='wasserstein gan',
                        action='store_true')
    parser.add_argument('-cp', '--conv_padding', help='convolutional zero-padding',
                        action='store_true')
    parser.add_argument('-cn', '--conv_number', help='convolutional layer depth',
                        type=int, default=4)
    parser.add_argument('-fbl', '--filter_base_length', help='size of filters in base hidden convolutional layer',
                        type=int, default=3)
    parser.add_argument('-fbs', '--filter_base_stride', help='size of filter stride in base hidden convolutional layer',
                        type=int, default=3)
    parser.add_argument('-fb', '--filter_base', help='base number of filters in base hidden convolutional layer',
                        type=int, default=4)
    parser.add_argument('-fl', '--filter_length', help='size of filters following base convolution',
                        type=int, default=3)
    parser.add_argument('-fs', '--filter_stride', help='size of filter strides following base convolution',
                        type=int, default=3)
    parser.add_argument('-ff', '--filter_factor', help='multiplicative factor of filters after base convolution',
                        type=int, default=2)
    parser.add_argument('-gd', '--generator_dropout', help='toggle generator dropout layers',
                        action='store_true')
    parser.add_argument('-dd', '--discriminator_dropout', help='toggle discriminator dropout layers',
                        action='store_true')
    parser.add_argument('-zd', '--z_dimension', help='sample noise dimension',
                        type=int, default=100)
    parser.add_argument('-cd', '--c_dimension', help='sample classification dimension',
                        type=int, default=5)
    parser.add_argument('-ud', '--u_dimension', help='sample continuous dimension',
                        type=int, default=0)
    parser.add_argument('-ki', '--kernel_initializer', help='kernel initializer',
                        type=str, default='lecun_normal')
    parser.add_argument('-an', '--activation', help='activation function',
                        type=str, default='selu')
    parser.add_argument('-dop', '--discriminator_optimizer', help='optimizer for discriminator',
                        type=str, default='sgd')
    parser.add_argument('-gop', '--gan_optimizer', help='optimizer for gan',
                        type=str, default='adam')
    parser.add_argument('-dlr', '--discriminator_learning_rate', help='learning rate for discriminator',
                        type=float, default=1e-3)
    parser.add_argument('-glr', '--gan_learning_rate', help='learning rate for gan',
                        type=float, default=1e-4)
    parser.add_argument('-gl', '--gan_lambda', help='gan regularization lambda',
                        type=float, default=1.0)
    parser.add_argument('-ta', '--trainer_alpha', help='trainer alpha label smoothing',
                        type=float, default=0.1)
    parser.add_argument('-tb', '--trainer_beta', help='trainer beta label flipping',
                        type=float, default=0.05)
    parser.add_argument('-dc', '--discriminator_cycles', help='number of discriminator training cycles per batch',
                        type=int, default=1)
    parser.add_argument('-gc', '--gan_cycles', help='number of gan training cycles per batch',
                        type=int, default=2)
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=128)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=128)
    args = parser.parse_args()
    return (args.verbose,
            args.name, args.lattice_sites, args.sample_interval, args.sample_number, args.scale_data, args.wasserstein, args.conv_padding, args.conv_number,
            args.filter_base_length, args.filter_base_stride, args.filter_base, args.filter_length, args.filter_stride, args.filter_factor,
            args.generator_dropout, args.discriminator_dropout, args.z_dimension, args.c_dimension, args.u_dimension,
            args.kernel_initializer, args.activation,
            args.discriminator_optimizer, args.gan_optimizer,
            args.discriminator_learning_rate, args.gan_learning_rate,
            args.gan_lambda, args.trainer_alpha, args.trainer_beta, args.discriminator_cycles, args.gan_cycles,
            args.batch_size, args.random_seed)


def load_losses(name, lattice_sites, interval, num_samples, scaled, seed, prfx, num_batches):
    ''' load loss histories from file '''
    # file parameters
    params = (name, lattice_sites, interval, num_samples, scaled, seed)
    file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx+'.loss.npy'
    losses = np.load(file_name)
    # change loss histories into lists
    dsc_fake_loss = losses[:, :, 0].reshape(-1, num_batches)
    dsc_real_loss = losses[:, :, 1].reshape(-1, num_batches)
    gan_loss = losses[:, :, 2].reshape(-1, num_batches)
    ent_cat_loss = losses[:, :, 3].reshape(-1, num_batches)
    ent_con_loss = losses[:, :, 4].reshape(-1, num_batches)
    return dsc_fake_loss, dsc_real_loss, gan_loss, ent_cat_loss, ent_con_loss


def plot_batch_losses(losses, cmap, file_prfx, verbose=False):
    file_name = os.getcwd()+'/'+file_prfx+'.loss.batch.png'
    # initialize figure and axes
    fig, ax = plt.subplots()
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    n_epochs, n_batches = losses[0].shape[:2]
    n_iters = n_epochs*n_batches
    # plot losses
    loss_list = ['Discriminator (Fake) Loss', 'Discriminator (Real) Loss',
                 'Generator Loss', 'Categorical Control Loss', 'Continuous Control Loss']
    color_list = np.linspace(0.2, 0.8, len(losses))
    for i in trange(len(losses), desc='Plotting Batch Losses', disable=not verbose):
        ax.plot(np.arange(1, n_iters+1), losses[i].reshape(-1), color=cmap(color_list[i]), label=loss_list[i])
    ax.legend(loc='upper right')
    # label axes
    ax.set_xticks(n_batches*np.arange(1, n_epochs+1), minor=True)
    ax.set_xticks(n_batches*np.arange(1, n_epochs+1)[::2], minor=False)
    ax.set_xticklabels(n_batches*np.arange(1, n_epochs+1)[::2], rotation=60)
    # ax.set_yticks(-np.log(0.125*np.arange(1, 8)), minor=False)
    # ax.set_yticklabels(np.round(-np.log(0.125*np.arange(1, 8)), 2))
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_epoch_losses(losses, cmap, file_prfx, verbose=False):
    file_name = os.getcwd()+'/'+file_prfx+'.loss.epoch.png'
    # initialize figure and axes
    fig, ax = plt.subplots()
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot losses
    loss_list = ['Discriminator (Fake) Loss', 'Discriminator (Real) Loss',
                 'Generator Loss', 'Categorical Control Loss', 'Continuous Control Loss']
    color_list = np.linspace(0.2, 0.8, len(losses))
    for i in trange(len(losses), desc='Plotting Epoch Losses', disable=not verbose):
        ax.plot(np.arange(1, losses[i].shape[0]+1), losses[i].mean(1), color=cmap(color_list[i]), label=loss_list[i])
    ax.legend(loc='upper right')
    # label axes
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_losses(losses, cmap,
                name, lattice_length, interval, num_samples, scaled, seed,
                prfx, verbose=False):
    # file name parameters
    params = (name, lattice_length, interval, num_samples, scaled, seed)
    file_prfx = '{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx
    plot_batch_losses(losses, cmap, file_prfx, verbose)
    plot_epoch_losses(losses, cmap, file_prfx, verbose)


def plot_diagram(data, temp_x, temp_y, cmap, file_prfx, alias):
    # file name parameters
    file_name = os.getcwd()+'/'+file_prfx+'.{}.png'.format(alias)
    # initialize figure and axes
    fig, ax = plt.subplots()
    # initialize colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('top', size='5%', pad=0.8)
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot diagram
    im = ax.imshow(data.T, aspect='equal', interpolation='none', origin='lower', cmap=cmap)
    # generate grid
    ax.grid(which='both', axis='both', linestyle='-', color='k', linewidth=1)
    # label ticks
    ax.set_xticks(np.arange(temp_x.size), minor=True)
    ax.set_yticks(np.arange(temp_y.size), minor=True)
    ax.set_xticks(np.arange(temp_x.size)[::4], minor=False)
    ax.set_yticks(np.arange(temp_y.size)[::4], minor=False)
    ax.set_xticklabels(np.round(temp_x, 2)[::4], rotation=-60)
    ax.set_yticklabels(np.round(temp_y, 2)[::4])
    # label axes
    ax.set_xlabel(r'$T_x$')
    ax.set_ylabel(r'$T_y$')
    # place colorbal
    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(data.min(), data.max(), 3))
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_diagrams(c_data, u_data, fields, temps, cmap,
                  name, lattice_length, interval, num_samples, scaled, seed,
                  prfx, verbose=False):
    params = (name, lattice_length, interval, num_samples, scaled, seed)
    file_prfx = '{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx
    c_m_diag = c_data.mean(2)
    c_s_diag = c_data.std(2)
    u_m_diag = u_data.mean(2)
    u_s_diag = u_data.std(2)
    c_m_dim = c_m_diag.shape[-1]
    c_s_dim = c_s_diag.shape[-1]
    u_m_dim = u_m_diag.shape[-1]
    u_s_dim = u_s_diag.shape[-1]
    d0, d1 = 'Means', 'StDvs'
    for i in trange(c_m_dim, desc='Plotting Discrete Control {}'.format(d0), disable=not verbose):
        plot_diagram(c_m_diag[:, :, i], fields, temps, cmap, file_prfx, 'c_m_{}'.format(i))
    for i in trange(c_s_dim, desc='Plotting Discrete Control {}'.format(d1), disable=not verbose):
        plot_diagram(c_s_diag[:, :, i], fields, temps, cmap, file_prfx, 'c_s_{}'.format(i))
    for i in trange(u_m_dim, desc='Plotting Continuous Control {}'.format(d0), disable=not verbose):
        plot_diagram(u_m_diag[:, :, i], fields, temps, cmap, file_prfx, 'u_m_{}'.format(i))
    for i in trange(u_s_dim, desc='Plotting Continuous Control {}'.format(d1), disable=not verbose):
        plot_diagram(u_s_diag[:, :, i], fields, temps, cmap, file_prfx, 'u_s_{}'.format(i))


if __name__ == '__main__':
    plt.rc('font', family='sans-serif')
    FTSZ = 28
    FIGW = 16
    PPARAM = {'figure.figsize': (FIGW, FIGW),
              'lines.linewidth': 4.0,
              'legend.fontsize': FTSZ,
              'axes.labelsize': FTSZ,
              'axes.titlesize': FTSZ,
              'axes.linewidth': 2.0,
              'xtick.labelsize': FTSZ,
              'xtick.major.size': 20,
              'xtick.major.width': 2.0,
              'ytick.labelsize': FTSZ,
              'ytick.major.size': 20,
              'ytick.major.width': 2.0,
              'font.size': FTSZ}
    plt.rcParams.update(PPARAM)
    CM = plt.get_cmap('plasma')

    (VERBOSE,
     NAME, N, I, NS, SC, W, CP,
     CN, FBL, FBS, FB, FL, FS, FF,
     GD, DD, ZD, CD, UD,
     KI, AN,
     DOPT, GOPT, DLR, GLR,
     GLAMB, TALPHA, TBETA, DC, GC,
     BS, SEED) = parse_args()

    params = (W,
              CN,
              FBL, FBS, FB,
              FL, FS, FF,
              GD, DD, ZD, CD, UD,
              KI, AN,
              DOPT, GOPT, DLR, GLR,
              GLAMB, BS, TALPHA, TBETA, DC, GC)
    PRFX = 'infogan.{:d}.{}.{}.{}.{}.{}.{}.{}.{:d}.{:d}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{:.0e}.{:.0e}.{}.{:.0e}.{:.0e}.{}.{}'.format(*params)

    TX, TY = load_thermal_params(NAME, N)
    TX, TY = TX[::I], TY[::I]
    NTX, NTY = TX.size, TY.size
    BN = NTX*NTY*NS//BS

    L = load_losses(NAME, N, I, NS, SC, SEED, PRFX, BN)
    plot_losses(L, CM, NAME, N, I, NS, SC, SEED, PRFX, VERBOSE)

    if CD > 0 and UD > 0:
        C = load_output_data('categorical_control', NAME, N, I, NS, SC, SEED, PRFX)
        U = load_output_data('continuous_control', NAME, N, I, NS, SC, SEED, PRFX)
    elif CD > 0:
        C = load_output_data('categorical_control', NAME, N, I, NS, SC, SEED, PRFX)
        U = np.zeros((NTX, NTY, NS, UD))
    elif UD > 0:
        C = np.zeros((NTX, NTY, NS, UD))
        U = load_output_data('continuous_control', NAME, N, I, NS, SC, SEED, PRFX)
    plot_diagrams(C, U, TX, TY, CM, NAME, N, I, NS, SC, SEED, PRFX, VERBOSE)