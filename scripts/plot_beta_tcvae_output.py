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
    parser.add_argument('-do', '--dropout', help='toggle dropout layers',
                        action='store_true')
    parser.add_argument('-zd', '--z_dimension', help='sample noise dimension',
                        type=int, default=2)
    parser.add_argument('-ka', '--kld_annealing', help='toggle kld annealing',
                        action='store_true')
    parser.add_argument('-ra', '--alpha', help='total correlation alpha',
                        type=float, default=0.01)
    parser.add_argument('-rb', '--beta', help='total correlation beta',
                        type=float, default=0.08)
    parser.add_argument('-rl', '--lamb', help='total correlation lambda',
                        type=float, default=0.01)
    parser.add_argument('-ki', '--kernel_initializer', help='kernel initializer',
                        type=str, default='lecun_normal')
    parser.add_argument('-an', '--activation', help='activation function',
                        type=str, default='selu')
    parser.add_argument('-op', '--optimizer', help='optimizer',
                        type=str, default='adam')
    parser.add_argument('-lr', '--learning_rate', help='learning rate',
                        type=float, default=1e-4)
    parser.add_argument('-bs', '--batch_size', help='size of batches',
                        type=int, default=128)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=128)
    args = parser.parse_args()
    return (args.verbose,
            args.name, args.lattice_sites, args.sample_interval, args.sample_number, args.scale_data, args.conv_padding,
            args.conv_number, args.filter_base_length, args.filter_base_stride, args.filter_base, args.filter_length, args.filter_stride, args.filter_factor,
            args.dropout, args.z_dimension, args.kld_annealing, args.alpha, args.beta, args.lamb,
            args.kernel_initializer, args.activation, args.optimizer, args.learning_rate,
            args.batch_size, args.random_seed)


def load_losses(name, lattice_sites, interval, num_samples, scaled, seed, prfx, num_batches):
    ''' load loss histories from file '''
    # file parameters
    params = (name, lattice_sites, interval, num_samples, scaled, seed)
    file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx+'.loss.npy'
    losses = np.load(file_name)
    # change loss histories into lists
    vae_loss = losses[:, :, 0].reshape(-1, num_batches)
    tc_loss = losses[:, :, 1].reshape(-1, num_batches)
    rc_loss = losses[:, :, 2].reshape(-1, num_batches)
    return vae_loss, tc_loss, rc_loss


def plot_histogram(u, cmap, file_prfx, alias, domain_name, verbose=False):
    file_name = os.getcwd()+'/'+file_prfx+'.{}.png'.format(alias)
    # initialize figure and axes
    fig, ax = plt.subplots()
    # remove spines on top and right
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set axis ticks to left and bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.hist(u, bins=64, density=True, color=cmap(0.25))
    ax.set_xlabel(domain_name)
    ax.set_ylabel('Density')
    # save figure
    fig.savefig(file_name)
    plt.close()


def plot_bc_error_accuracy(error, accuracy, cmap,
                           name, lattice_sites, interval, num_samples, scaled, seed,
                           prfx, verbose=False):
    # file name parameters
    params = (name, lattice_sites, interval, num_samples, scaled, seed)
    file_prfx = '{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx
    plot_histogram(error[:, 0], cmap, file_prfx, 'bc_err_hist', 'Binary Crossentropy (Reconstruction Loss)', verbose)
    plot_histogram(accuracy[:, 0], cmap, file_prfx, 'bc_acc_hist', 'Classification Accuracy (Reconstruction)', verbose)


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
    loss_list = ['VAE Loss', 'Latent Loss', 'Reconstruction Loss']
    color_list = np.linspace(0.2, 0.8, len(losses))
    for i in trange(len(losses), desc='Plotting Batch Losses', disable=not verbose):
        ax.plot(np.arange(1, n_iters+1), losses[i].reshape(-1), color=cmap(color_list[i]), label=loss_list[i])
    ax.legend(loc='upper right')
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
    loss_list = ['VAE Loss', 'Latent Loss', 'Reconstruction Loss']
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
                name, lattice_sites, interval, num_samples, scaled, seed,
                prfx, verbose=False):
    # file name parameters
    params = (name, lattice_sites, interval, num_samples, scaled, seed)
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


def plot_diagrams(m_data, s_data, temp_x, temp_y, cmap,
                  name, lattice_sites, interval, num_samples, scaled, seed,
                  prfx, alias, verbose=False):
    params = (name, lattice_sites, interval, num_samples, scaled, seed)
    file_prfx = '{}.{}.{}.{}.{:d}.{}.'.format(*params)+prfx
    if s_data is not None:
        mm_diag = m_data.mean(2)
        ms_diag = m_data.std(2)
        sm_diag = s_data.mean(2)
        ss_diag = s_data.std(2)
        mm_dim = mm_diag.shape[-1]
        ms_dim = ms_diag.shape[-1]
        sm_dim = sm_diag.shape[-1]
        ss_dim = ss_diag.shape[-1]
        if alias == ['m', 's'] or alias == ['m_p', 's_p']:
            d0, d1 = 'Means', 'Sigmas'
        elif alias == ['bc_err', 'bc_acc']:
            d0, d1 = 'BC Errors', 'BC Accuracies'
        for i in trange(mm_dim, desc='Plotting Mean VAE {}'.format(d0), disable=not verbose):
            plot_diagram(mm_diag[:, :, i], temp_x, temp_y, cmap, file_prfx, '{}_m_{}'.format(alias[0], i))
        for i in trange(sm_dim, desc='Plotting Mean VAE {}'.format(d1), disable=not verbose):
            plot_diagram(sm_diag[:, :, i], temp_x, temp_y, cmap, file_prfx, '{}_m_{}'.format(alias[1], i))
        for i in trange(ms_dim, desc='Plotting StDv VAE {}'.format(d0), disable=not verbose):
            plot_diagram(ms_diag[:, :, i], temp_x, temp_y, cmap, file_prfx, '{}_s_{}'.format(alias[0], i))
        for i in trange(ss_dim, desc='Plotting StDv VAE {}'.format(d1), disable=not verbose):
            plot_diagram(ss_diag[:, :, i], temp_x, temp_y, cmap, file_prfx, '{}_s_{}'.format(alias[1], i))
    else:
        zm_diag = m_data.mean(2)
        zs_diag = m_data.std(2)
        zm_dim = zm_diag.shape[-1]
        zs_dim = zs_diag.shape[-1]
        for i in trange(zm_dim, desc='Plotting Mean VAE Encodings', disable=not verbose):
            plot_diagram(zm_diag[:, :, i], temp_x, temp_y, cmap, file_prfx, '{}_m_{}'.format(alias, i))
        for i in trange(zs_dim, desc='Plotting StDv VAE Encodings', disable=not verbose):
            plot_diagram(zs_diag[:, :, i], temp_x, temp_y, cmap, file_prfx, '{}_s_{}'.format(alias, i))


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
     NAME, N, I, NS, SC, CP,
     CN, FBL, FBS, FB, FL, FS, FF,
     DO, ZD, KA, ALPHA, BETA, LAMBDA,
     KI, AN, OPT, LR,
     BS, SEED) = parse_args()

    FPARAM = (CN, FBL, FBS, FB, FL, FS, FF,
              DO, ZD, KA, ALPHA, BETA, LAMBDA,
              KI, AN, OPT, LR,
              BS)
    PRFX = 'btcvae.{}.{}.{}.{}.{}.{}.{}.{:d}.{}.{:d}.{:.0e}.{:.0e}.{:.0e}.{}.{}.{}.{:.0e}.{}'.format(*FPARAM)

    TX, TY = load_thermal_params(NAME, N)
    TX, TY = TX[::I], TY[::I]
    NTX, NTY = TX.size, TY.size
    BN = NTX*NTY*NS//BS

    L = load_losses(NAME, N, I, NS, SC, SEED, PRFX, BN)
    plot_losses(L, CM, NAME, N, I, NS, SC, SEED, PRFX, VERBOSE)

    if np.any(np.array([ALPHA, BETA, LAMBDA]) > 0):
        MU = load_output_data('vae_mean', NAME, N, I, NS, SC, SEED, PRFX)
        SIGMA = load_output_data('vae_sigma', NAME, N, I, NS, SC, SEED, PRFX)
        Z = load_output_data('vae_z', NAME, N, I, NS, SC, SEED, PRFX)
        plot_diagrams(MU, SIGMA, TX, TY, CM, NAME, N, I, NS, SC, SEED, PRFX, ['m', 's'], VERBOSE)
        plot_diagrams(Z, None, TX, TY, CM, NAME, N, I, NS, SC, SEED, PRFX, 'z', VERBOSE)
    else:
        Z = load_output_data('vae_z', NAME, N, I, NS, SC, SEED, PRFX)
        plot_diagrams(Z, None, TX, TY, CM, NAME, N, I, NS, SC, SEED, PRFX, 'z', VERBOSE)

    # BCERR = load_output_data('bc_err', NAME, N, I, NS, SC, SEED, PRFX)
    # BCACC = load_output_data('bc_acc', NAME, N, I, NS, SC, SEED, PRFX)
    # plot_diagrams(BCERR.reshape(NTX, NTY, NS, -1), BCACC.reshape(NTX, NTY, NS, -1), TX, TY, CM, NAME, N, I, NS, SC, SEED, PRFX, ['bc_err', 'bc_acc'], VERBOSE)
    # plot_bc_error_accuracy(BCERR, BCACC, CM, NAME, N, I, NS, SC, SEED, PRFX, VERBOSE)