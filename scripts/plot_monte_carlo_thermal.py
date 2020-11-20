# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 05:51:23 2020

@author: Nicholas
"""

import argparse
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('font', family='sans-serif')
FTSZ = 28
FIGW = 16
PPARAMS = {'figure.figsize': (FIGW, FIGW),
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
plt.rcParams.update(PPARAMS)
SCALE = lambda a, b: (a-np.min(b))/(np.max(b)-np.min(b))
CM = plt.get_cmap('plasma')

def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-nm', '--name', help='simulation name',
                        type=str, default='init')
    parser.add_argument('-n', '--lattice_sites', help='lattice sites',
                        type=int, default=81)
    args = parser.parse_args()
    return args.verbose, args.name, args.lattice_sites

VERBOSE, NAME, N = parse_args()
# current working directory and prefix
CWD = os.getcwd()
PREF = CWD+'/%s.%d' % (NAME, N)
# external fields and temperatures
TX = np.load(PREF+'.tx.npy')
TY = np.load(PREF+'.ty.npy')
NTX, NTY = TX.size, TY.size

DAT = np.load(PREF+'.dat.npy')

ENER = DAT[:, :, :, 2]
MAG = DAT[:, :, :, 3]

MENER = ENER.mean(2)
MMAG = np.abs(MAG).mean(2)

def plot_diagram(data, alias):
    file_name = PREF+'.{}.png'.format(alias)
    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('top', size='5%', pad=0.8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    im = ax.imshow(data, aspect='equal', interpolation='none', origin='lower', cmap=CM)
    ax.grid(which='both', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(NTX), minor=True)
    ax.set_yticks(np.arange(NTY), minor=True)
    ax.set_xticks(np.arange(NTX)[::4], minor=False)
    ax.set_yticks(np.arange(NTY)[::4], minor=False)
    ax.set_xticklabels(np.round(TX, 2)[::4], rotation=-60)
    ax.set_yticklabels(np.round(TY, 2)[::4])
    # label axes
    ax.set_xlabel(r'$T_x$')
    ax.set_ylabel(r'$T_y$')
    # place colorbal
    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.linspace(data.min(), data.max(), 3))
    # save figure
    fig.savefig(file_name)
    plt.close()

plot_diagram(MENER, 'ener')
plot_diagram(MMAG, 'mag')

DENER = np.diagonal(ENER, offset=0, axis1=0, axis2=1)
DMAG = np.diagonal(MAG, offset=0, axis1=0, axis2=1)

DMENER = DENER.mean(0)
DMMAG = np.abs(DMAG).mean(0)
DSPHT = np.square(DENER.std(0))/np.square(TX)
DMSUSC = np.square(np.abs(DMAG).std(0))/TX

def plot_diagonal(data, alias, label):
    file_name = PREF+'.diag.{}.png'.format(alias)
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.plot(TX, data, color='k')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(label)
    fig.savefig(file_name)
    plt.close()


plot_diagonal(DMENER, 'ener', r'$E$')
plot_diagonal(DMMAG, 'mag', r'$m$')
plot_diagonal(DSPHT, 'spht', r'$c$')
plot_diagonal(DMSUSC, 'msusc', r'$\chi$')