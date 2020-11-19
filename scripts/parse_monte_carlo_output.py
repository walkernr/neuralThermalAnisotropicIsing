# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:45:15 2020

@author: Nicholas
"""

import argparse
import os
import numpy as np

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

DAT = np.loadtxt(PREF+'.dat', dtype=np.float32).reshape(NTX, NTY, -1, 5)
DMP = np.loadtxt(PREF+'.dmp', dtype=np.int8).reshape(NTX, NTY, -1, N, N)

if VERBOSE:
    print('all data loaded')

np.save(PREF+'.dat.npy', DAT)
np.save(PREF+'.dmp.npy', DMP)

if VERBOSE:
    print('all data dumped')
