# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 00:13:46 2019

@author: Nicholas
"""

import argparse
import os
import numpy as np
from tqdm import tqdm, trange
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import (Input, Flatten, Reshape, Concatenate, Lambda,
                                     Dense, BatchNormalization, Conv2D, Conv2DTranspose,
                                     SpatialDropout2D, AlphaDropout, Activation, LeakyReLU)
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adamax, Nadam
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.python.training.checkpoint_management import CheckpointManager
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_utils import *
from conv_utils import *
from dist_utils import *


def parse_args():
    ''' parses command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output',
                        action='store_true')
    parser.add_argument('-r', '--restart', help='restart mode',
                        action='store_true')
    parser.add_argument('-pt', '--plot', help='plot results',
                        action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel (cpu) mode',
                        action='store_true')
    parser.add_argument('-g', '--gpu', help='gpu mode (will default to cpu if unable)',
                        action='store_true')
    parser.add_argument('-nt', '--threads', help='number of parallel threads',
                        type=int, default=20)
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
    parser.add_argument('-rs', '--random_sampling', help='random batch sampling',
                        action='store_true')
    parser.add_argument('-ep', '--epochs', help='number of training epochs',
                        type=int, default=16)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=128)
    args = parser.parse_args()
    return (args.verbose, args.restart, args.plot, args.parallel, args.gpu, args.threads,
            args.name, args.lattice_sites, args.sample_interval, args.sample_number, args.scale_data, args.wasserstein, args.conv_padding, args.conv_number,
            args.filter_base_length, args.filter_base_stride, args.filter_base, args.filter_length, args.filter_stride, args.filter_factor,
            args.generator_dropout, args.discriminator_dropout, args.z_dimension, args.c_dimension, args.u_dimension,
            args.kernel_initializer, args.activation,
            args.discriminator_optimizer, args.gan_optimizer,
            args.discriminator_learning_rate, args.gan_learning_rate,
            args.gan_lambda, args.trainer_alpha, args.trainer_beta, args.discriminator_cycles, args.gan_cycles,
            args.batch_size, args.random_sampling, args.epochs, args.random_seed)


class ClipConstraint(Constraint):
    '''
    Clipping constraint for weights in WGAN model
    '''
    def __init__(self, clip_value):
        ''' initializer '''
        self.clip_value = clip_value


    def __call__(self, weights):
        ''' call function clips weight '''
        return K.clip(weights, -self.clip_value, self.clip_value)


    def get_config(self):
        ''' get weight clipper configuration '''
        return {'clip_value': self.clip_value}


class InfoGAN():
    '''
    InfoGAN Model
    Generative adversarial modeling of the Ising spin configurations
    '''
    def __init__(self, input_shape=(81, 81, 1), scaled=False, wasserstein=False, padded=False, conv_number=4,
                 filter_base_length=3, filter_base_stride=3, filter_base=32, filter_length=3, filter_stride=3, filter_factor=1,
                 gen_drop=False, dsc_drop=False,
                 z_dim=100, c_dim=5, u_dim=0,
                 krnl_init='lecun_normal', act='selu',
                 dsc_opt_n='sgd', gan_opt_n='adam', dsc_lr=1e-2, gan_lr=1e-3, lamb=1.0,
                 batch_size=128,
                 alpha=0.0, beta=0.0, n_dsc=1, n_gan=2):
        ''' initialize model parameters '''
        self.eps = 1e-8
        self.wasserstein = wasserstein
        self.padded = padded
        if self.padded:
            self.padding = 'same'
        else:
            self.padding = 'valid'
        # convolutional parameters
        # number of convolutions
        self.conv_number = conv_number
        # number of filters for first convolution
        self.filter_base = filter_base
        # multiplicative factor for filters in subsequent convolutions
        self.filter_factor = filter_factor
        # filter side length
        self.filter_base_length = filter_base_length
        self.filter_length = filter_length
        # filter stride
        self.filter_base_stride = filter_base_stride
        self.filter_stride = filter_stride
        # convolutional input and output shapes
        self.input_shape = input_shape
        self.final_conv_shape = get_final_conv_shape(self.input_shape, self.conv_number,
                                                     self.filter_base_length, self.filter_length,
                                                     self.filter_base_stride, self.filter_stride,
                                                     self.filter_base, self.filter_factor, self.padded)
        # generator and discriminator dropout
        self.gen_drop = gen_drop
        self.dsc_drop = dsc_drop
        # latent noise dimension
        self.z_dim = z_dim
        # categorical control variable dimension
        self.c_dim = c_dim
        # continuous control variable dimension
        self.u_dim = u_dim
        # discriminator/auxiliary dense dimension
        self.d_q_dim = self.z_dim+self.c_dim+self.u_dim
        # kernel initializer and activation
        self.krnl_init = krnl_init
        self.act = act
        # discriminator and gan optimizers
        self.dsc_opt_n = dsc_opt_n
        self.gan_opt_n = gan_opt_n
        # discriminator and gan learning rates
        self.dsc_lr = dsc_lr
        self.gan_lr = gan_lr
        # regularization constant
        self.lamb = lamb
        # batch size and callbacks
        self.batch_size = batch_size
        # scaling
        if scaled:
            self.gen_out_act = 'sigmoid'
        else:
            self.gen_out_act = 'tanh'
        # training alpha and beta
        self.alpha = alpha
        self.beta = beta
        # training cycles
        self.n_dsc = n_dsc
        self.n_gan = n_gan
        # loss histories
        self.dsc_fake_loss_history = []
        self.dsc_real_loss_history = []
        self.gan_loss_history = []
        self.ent_cat_loss_history = []
        self.ent_con_loss_history = []
        # past epochs (changes if loading past trained model)
        self.past_epochs = 0
        # checkpoint managers
        self.dsc_mngr = None
        self.gan_mngr = None
        # build full model
        self._build_model()


    def get_file_prefix(self):
        ''' gets parameter tuple and filename string prefix '''
        params = (self.wasserstein,
                  self.conv_number,
                  self.filter_base_length, self.filter_base_stride, self.filter_base,
                  self.filter_length, self.filter_stride, self.filter_factor,
                  self.gen_drop, self.dsc_drop, self.z_dim, self.c_dim, self.u_dim,
                  self.krnl_init, self.act,
                  self.dsc_opt_n, self.gan_opt_n, self.dsc_lr, self.gan_lr,
                  self.lamb, self.batch_size, self.alpha, self.beta, self.n_dsc, self.n_gan)
        file_name = 'infogan.{:d}.{}.{}.{}.{}.{}.{}.{}.{:d}.{:d}.{}.{}.{}.{}.{}.{}.{}.{:.0e}.{:.0e}.{:.0e}.{}.{:.0e}.{:.0e}.{}.{}'.format(*params)
        return file_name


    def _build_model(self):
        ''' builds each component of the InfoGAN model '''
        self._build_generator()
        self._build_discriminator()
        self._build_auxiliary()
        self._build_gan()


    def binary_crossentropy_loss(self, category, prediction):
        ''' binary crossentropy loss for real/fake discrimination '''
        return -K.mean(category*K.log(prediction+self.eps)+(1-category)*K.log(1-prediction+self.eps))


    def wasserstein_loss(self, category, prediction):
        ''' Wasserstein loss for real/fake discrimination '''
        return K.mean(category*prediction)


    def mutual_information_categorical_loss(self, category, prediction):
        ''' mutual information loss for categorical control variables '''
        entropy = -K.mean(K.sum(category*K.log(category+self.eps), axis=1))
        conditional_entropy = -K.mean(K.sum(category*K.log(prediction+self.eps), axis=1))
        return entropy+conditional_entropy


    def _build_generator(self):
        ''' builds generator model '''
        # latent unit gaussian and categorical inputs
        self.gen_z_input = Input(batch_shape=(self.batch_size, self.z_dim), name='gen_z_input')
        if self.c_dim > 0 and self.u_dim > 0:
            self.gen_c_input = Input(batch_shape=(self.batch_size, self.c_dim), name='gen_c_input')
            self.gen_u_input = Input(batch_shape=(self.batch_size, self.u_dim), name='gen_u_input')
            x = Concatenate(name='gen_latent_concat')([self.gen_z_input, self.gen_c_input, self.gen_u_input])
            self.gen_inputs = [self.gen_z_input, self.gen_c_input, self.gen_u_input]
        elif self.c_dim > 0:
            self.gen_c_input = Input(batch_shape=(self.batch_size, self.c_dim), name='gen_c_input')
            x = Concatenate(name='gen_latent_concat')([self.gen_z_input, self.gen_c_input])
            self.gen_inputs = [self.gen_z_input, self.gen_c_input]
        elif self.u_dim >0:
            self.gen_u_input = Input(batch_shape=(self.batch_size, self.u_dim), name='gen_u_input')
            x = Concatenate(name='gen_latent_concat')([self.gen_z_input, self.gen_u_input])
            self.gen_inputs = [self.gen_z_input, self.gen_u_input]
        # dense layer with same feature count as final convolution
        u = 0
        x = Dense(units=np.prod(self.final_conv_shape),
                  kernel_initializer=self.krnl_init,
                  name='gen_dense_{}'.format(u))(x)
        if self.act == 'lrelu':
            x = LeakyReLU(alpha=0.1, name='gen_dense_lrelu_{}'.format(u))(x)
            x = BatchNormalization(name='gen_dense_batchnorm_{}'.format(u))(x)
        elif self.act == 'selu':
            x = Activation(activation='selu', name='gen_dense_selu_{}'.format(u))(x)
        u += 1
        # reshape to final convolution shape
        convt = Reshape(target_shape=self.final_conv_shape, name='gen_rshp_0')(x)
        if self.gen_drop:
            if self.act == 'lrelu':
                convt = SpatialDropout2D(rate=0.5, name='gen_rshp_drop_0')(convt)
            elif self.act == 'selu':
                convt = AlphaDropout(rate=0.5, noise_shape=(self.batch_size, 1, 1, self.final_conv_shape[-1]), name='gen_rshp_drop_0')(convt)
        v = 0
        # transform to sample shape with transposed convolutions
        for i in range(self.conv_number-1, 0, -1):
            filter_number = get_filter_number(i-1, self.filter_base, self.filter_factor)
            convt = Conv2DTranspose(filters=filter_number, kernel_size=self.filter_length,
                                    kernel_initializer=self.krnl_init,
                                    padding=self.padding, strides=self.filter_stride,
                                    name='gen_convt_{}'.format(v))(convt)
            if self.act == 'lrelu':
                convt = LeakyReLU(alpha=0.1, name='gen_convt_lrelu_{}'.format(v))(convt)
                convt = BatchNormalization(name='gen_convt_batchnorm_{}'.format(v))(convt)
                if self.gen_drop:
                    convt = SpatialDropout2D(rate=0.5, name='gen_convt_drop_{}'.format(v))(convt)
            elif self.act == 'selu':
                convt = Activation(activation='selu', name='gen_convt_selu_{}'.format(v))(convt)
                if self.gen_drop:
                    convt = AlphaDropout(rate=0.5, noise_shape=(self.batch_size, 1, 1, filter_number), name='gen_convt_drop_{}'.format(v))(convt)
            v += 1
        self.gen_x_output = Conv2DTranspose(filters=1, kernel_size=self.filter_base_length,
                                            kernel_initializer='glorot_uniform', activation=self.gen_out_act,
                                            padding=self.padding, strides=self.filter_base_stride,
                                            name='gen_x_output')(convt)
        self.gen_outputs = [self.gen_x_output]
        # build generator
        self.generator = Model(inputs=self.gen_inputs, outputs=self.gen_outputs, name='generator')


    def _build_discriminator(self):
        ''' builds discriminator model '''
        # takes sample (real or fake) as input
        self.dsc_x_input = Input(batch_shape=(self.batch_size,)+self.input_shape, name='dsc_x_input')
        self.dsc_inputs = [self.dsc_x_input]
        if self.wasserstein:
            out_act = 'linear'
            loss = self.wasserstein_loss
            conv_constraint = ClipConstraint(0.01)
        else:
            out_act = 'sigmoid'
            loss = self.binary_crossentropy_loss
            conv_constraint = None
        conv = self.dsc_x_input
        # iterative convolutions over input
        for i in range(self.conv_number):
            filter_number = get_filter_number(i, self.filter_base, self.filter_factor)
            filter_length, filter_stride = get_filter_length_stride(i, self.filter_base_length, self.filter_base_stride, self.filter_length, self.filter_stride)
            conv = Conv2D(filters=filter_number, kernel_size=filter_length,
                          kernel_initializer=self.krnl_init, kernel_constraint=conv_constraint,
                          padding=self.padding, strides=filter_stride,
                          name='dsc_conv_{}'.format(i))(conv)
            if self.act == 'lrelu':
                conv = LeakyReLU(alpha=0.1, name='dsc_conv_lrelu_{}'.format(i))(conv)
                conv = BatchNormalization(name='dsc_conv_batchnorm_{}'.format(i))(conv)
                if self.dsc_drop:
                    conv = SpatialDropout2D(rate=0.5, name='dsc_conv_drop_{}'.format(i))(conv)
            elif self.act == 'selu':
                conv = Activation(activation='selu', name='dsc_conv_selu_{}'.format(i))(conv)
                if self.dsc_drop:
                    conv = AlphaDropout(rate=0.5, noise_shape=(self.batch_size, 1, 1, filter_number), name='dsc_conv_drop_{}'.format(i))(conv)
        # flatten final convolutional layer
        x = Flatten(name='dsc_fltn_0')(conv)
        u = 0
        # the dense layer is saved as a hidden encoding layer
        self.dsc_enc = x
        # dense layer
        # x = Dense(units=self.d_q_dim,
        #           kernel_initializer=self.krnl_init,
        #           name='dsc_dense_{}'.format(u))(x)
        # if self.act == 'lrelu':
        #     x = LeakyReLU(alpha=0.1, name='dsc_dense_lrelu_{}'.format(u))(x)
        #     x = BatchNormalization(name='dsc_dense_batchnorm_{}'.format(u))(x)
        # elif self.act == 'selu':
        #     x = Activation(activation='selu', name='dsc_dense_selu_{}'.format(u))(x)
        # u += 1
        # discriminator classification output (0, 1) -> (fake, real)
        self.dsc_v_output = Dense(units=1,
                                  kernel_initializer='glorot_uniform', activation=out_act,
                                  name='dsc_v_output')(x)
        self.dsc_outputs = [self.dsc_v_output]
        # build discriminator
        self.discriminator = Model(inputs=self.dsc_inputs, outputs=self.dsc_outputs, name='discriminator')
        # define optimizer
        if self.dsc_opt_n == 'sgd':
            self.dsc_opt = SGD(learning_rate=self.dsc_lr)
        elif self.dsc_opt_n == 'sgdm':
            self.dsc_opt = SGD(learning_rate=self.dsc_lr, momentum=0.5)
        elif self.dsc_opt_n == 'nsgd':
            self.dsc_opt = SGD(learning_rate=self.dsc_lr, momentum=0.5, nesterov=True)
        elif self.dsc_opt_n == 'rmsprop':
            self.dsc_opt = RMSprop(learning_rate=self.dsc_lr)
        elif self.dsc_opt_n == 'rmsprop_cent':
            self.dsc_opt = RMSprop(learning_rate=self.dsc_lr, centered=True)
        elif self.dsc_opt_n == 'adam':
            self.dsc_opt = Adam(learning_rate=self.dsc_lr, beta_1=0.5)
        elif self.dsc_opt_n == 'adam_ams':
            self.dsc_opt = Adam(learning_rate=self.dsc_lr, beta_1=0.5, amsgrad=True)
        elif self.dsc_opt_n == 'adamax':
            self.dsc_opt = Adamax(learning_rate=self.dsc_lr, beta_1=0.5)
        elif self.dsc_opt_n == 'adamax_ams':
            self.dsc_opt = Adamax(learning_rate=self.dsc_lr, beta_1=0.5, amsgrad=True)
        elif self.dsc_opt_n == 'nadam':
            self.dsc_opt = Nadam(learning_rate=self.dsc_lr, beta_1=0.5)
        # compile discriminator
        self.discriminator.compile(loss=loss, optimizer=self.dsc_opt)


    def _build_auxiliary(self):
        ''' builds auxiliary classification reconstruction model '''
        if self.wasserstein:
            # takes sample (real or fake) as input
            self.aux_x_input = Input(batch_shape=(self.batch_size,)+self.input_shape, name='aux_x_input')
            self.aux_inputs = [self.aux_x_input]
            conv = self.aux_x_input
            # iterative convolutions over input
            for i in range(self.conv_number):
                filter_number = get_filter_number(i, self.filter_base, self.filter_factor)
                filter_length, filter_stride = get_filter_length_stride(i, self.filter_base_length, self.filter_base_stride, self.filter_length, self.filter_stride)
                conv = Conv2D(filters=filter_number, kernel_size=filter_length,
                              kernel_initializer=self.krnl_init,
                              padding=self.padding, strides=filter_stride,
                              name='aux_conv_{}'.format(i))(conv)
                if self.act == 'lrelu':
                    conv = LeakyReLU(alpha=0.1, name='aux_conv_lrelu_{}'.format(i))(conv)
                    conv = BatchNormalization(name='aux_conv_batchnorm_{}'.format(i))(conv)
                    if self.dsc_drop:
                        conv = SpatialDropout2D(rate=0.5, name='aux_conv_drop_{}'.format(i))(conv)
                elif self.act == 'selu':
                    conv = Activation(activation='selu', name='aux_conv_selu_{}'.format(i))(conv)
                    if self.dsc_drop:
                        conv = AlphaDropout(rate=0.5, noise_shape=(self.batch_size, 1, 1, filter_number), name='aux_conv_drop_{}'.format(i))(conv)
            # flatten final convolutional layer
            x = Flatten(name='aux_fltn_0')(conv)
            u = 0
            # dense layer
            # x = Dense(units=self.d_q_dim,
            #           kernel_initializer=self.krnl_init,
            #           name='aux_dense_{}'.format(u))(x)
            # if self.act == 'lrelu':
            #     x = LeakyReLU(alpha=0.1, name='aux_dense_lrelu_{}'.format(u))(x)
            #     x = BatchNormalization(name='aux_dense_batchnorm_{}'.format(u))(x)
            # elif self.act == 'selu':
            #     x = Activation(activation='selu', name='aux_dense_selu_{}'.format(u))(x)
            # u += 1
            # auxiliary output is a reconstruction of the categorical assignments fed into the generator
            if self.c_dim > 0 and self.u_dim > 0:
                self.aux_c_output = Dense(self.c_dim,
                                          kernel_initializer='glorot_uniform', activation='softmax',
                                          name='aux_c_output')(x)
                self.aux_u_output = Dense(self.u_dim,
                                          kernel_initializer='glorot_uniform', activation='tanh',
                                          name='aux_u_output')(x)
                self.aux_outputs = [self.aux_c_output, self.aux_u_output]
            elif self.c_dim > 0:
                self.aux_c_output = Dense(self.c_dim,
                                          kernel_initializer='glorot_uniform', activation='softmax',
                                          name='aux_c_output')(x)
                self.aux_outputs = [self.aux_c_output]
            elif self.u_dim > 0:
                self.aux_u_output = Dense(self.u_dim,
                                          kernel_initializer='glorot_uniform', activation='tanh',
                                          name='aux_u_output')(x)
                self.aux_outputs = [self.aux_u_output]
            # build auxiliary classifier
            self.auxiliary = Model(inputs=self.aux_inputs, outputs=self.aux_outputs,
                                    name='auxiliary')
        else:
            # initialize with dense layer taking the hidden generator layer as input
            u = 0
            # x = Dense(units=self.d_q_dim,
            #           kernel_initializer=self.krnl_init,
            #           name='aux_dense_{}'.format(u))(self.dsc_enc)
            # if self.act == 'lrelu':
            #     x = LeakyReLU(alpha=0.1, name='aux_dense_lrelu_{}'.format(u))(x)
            #     x = BatchNormalization(name='aux_dense_batchnorm_{}'.format(u))(x)
            # elif self.act == 'selu':
            #     x = Activation(activation='selu', name='aux_dense_selu_{}'.format(u))(x)
            # u += 1
            x = self.dsc_enc
            # auxiliary output is a reconstruction of the categorical assignments fed into the generator
            if self.c_dim > 0 and self.u_dim > 0:
                self.aux_c_output = Dense(self.c_dim,
                                          kernel_initializer='glorot_uniform', activation='softmax',
                                          name='aux_c_output')(x)
                self.aux_u_output = Dense(self.u_dim,
                                          kernel_initializer='glorot_uniform', activation='tanh',
                                          name='aux_u_output')(x)
                self.aux_outputs = [self.aux_c_output, self.aux_u_output]
            elif self.c_dim > 0:
                self.aux_c_output = Dense(self.c_dim,
                                          kernel_initializer='glorot_uniform', activation='softmax',
                                          name='aux_c_output')(x)
                self.aux_outputs = [self.aux_c_output]
            elif self.u_dim > 0:
                self.aux_u_output = Dense(self.u_dim,
                                          kernel_initializer='glorot_uniform', activation='tanh',
                                          name='aux_u_output')(x)
                self.aux_outputs = [self.aux_u_output]
            # build auxiliary classifier
            self.auxiliary = Model(inputs=self.dsc_inputs, outputs=self.aux_outputs,
                                   name='auxiliary')


    def _build_gan(self):
        ''' builds generative adversarial network '''
        # static discriminator output
        if self.wasserstein:
            dsc_loss = self.wasserstein_loss
        else:
            dsc_loss = self.binary_crossentropy_loss
        self.discriminator.trainable = False
        # discriminated generator output
        gen_output = self.generator(self.gen_inputs)
        gan_dsc_output = self.discriminator(gen_output)
        # auxiliary output
        gan_aux_outputs = self.auxiliary(gen_output)
        # build GAN
        if self.wasserstein:
            self.gan_dsc = Model(inputs=self.gen_inputs, outputs=gan_dsc_output, name='infocgan_discriminator')
            self.gan_aux = Model(inputs=self.gen_inputs, outputs=gan_aux_outputs, name='infocgan_auxiliary')
            # define GAN optimizer
            if self.gan_opt_n == 'sgd':
                self.gan_dsc_opt = SGD(learning_rate=self.gan_lr)
                self.gan_aux_opt = SGD(learning_rate=self.gan_lr)
            elif self.gan_opt_n == 'sgdm':
                self.gan_dsc_opt = SGD(learning_rate=self.gan_lr, momentum=0.5)
                self.gan_aux_opt = SGD(learning_rate=self.gan_lr, momentum=0.5)
            elif self.gan_opt_n == 'nsgd':
                self.gan_dsc_opt = SGD(learning_rate=self.gan_lr, momentum=0.5, nesterov=True)
                self.gan_aux_opt = SGD(learning_rate=self.gan_lr, momentum=0.5, nesterov=True)
            elif self.gan_opt_n == 'rmsprop':
                self.gan_dsc_opt = RMSprop(learning_rate=self.gan_lr)
                self.gan_aux_opt = RMSprop(learning_rate=self.gan_lr)
            elif self.gan_opt_n == 'rmsprop_cent':
                self.gan_dsc_opt = RMSprop(learning_rate=self.gan_lr, centered=True)
                self.gan_aux_opt = RMSprop(learning_rate=self.gan_lr, centered=True)
            elif self.gan_opt_n == 'adam':
                self.gan_dsc_opt = Adam(learning_rate=self.gan_lr, beta_1=0.5)
                self.gan_aux_opt = Adam(learning_rate=self.gan_lr, beta_1=0.5)
            elif self.gan_opt_n == 'adam_ams':
                self.gan_dsc_opt = Adam(learning_rate=self.gan_lr, beta_1=0.5, amsgrad=True)
                self.gan_aux_opt = Adam(learning_rate=self.gan_lr, beta_1=0.5, amsgrad=True)
            elif self.gan_opt_n == 'adamax':
                self.gan_dsc_opt = Adamax(learning_rate=self.gan_lr, beta_1=0.5)
                self.gan_aux_opt = Adamax(learning_rate=self.gan_lr, beta_1=0.5)
            elif self.gan_opt_n == 'adamax_ams':
                self.gan_dsc_opt = Adamax(learning_rate=self.gan_lr, beta_1=0.5, amsgrad=True)
                self.gan_aux_opt = Adamax(learning_rate=self.gan_lr, beta_1=0.5, amsgrad=True)
            elif self.gan_opt_n == 'nadam':
                self.gan_dsc_opt = Nadam(learning_rate=self.gan_lr, beta_1=0.5)
                self.gan_aux_opt = Nadam(learning_rate=self.gan_lr, beta_1=0.5)
            # compile GAN
            self.gan_dsc.compile(loss=dsc_loss, optimizer=self.gan_dsc_opt)
            if self.c_dim > 0 and self.u_dim > 0:
                aux_loss = {'auxiliary': 'categorical_crossentropy',
                            'auxiliary_1': 'mean_squared_error'}
                aux_loss_weights = {'auxiliary': self.lamb,
                                    'auxiliary_1': self.lamb}
            elif self.c_dim > 0:
                aux_loss = 'categorical_crossentropy'
                aux_loss_weights = [self.lamb]
            elif self.u_dim > 0:
                aux_loss = 'mean_squared_error'
                aux_loss_weights = [self.lamb]
            self.gan_aux.compile(loss=aux_loss, loss_weights=aux_loss_weights, optimizer=self.gan_aux_opt)
        else:
            if self.c_dim > 0 and self.u_dim > 0:
                gan_outputs = [gan_dsc_output]+gan_aux_outputs
            else:
                gan_outputs = [gan_dsc_output, gan_aux_outputs]
            self.gan = Model(inputs=self.gen_inputs, outputs=gan_outputs, name='infocgan')
            # define GAN optimizer
            if self.gan_opt_n == 'sgd':
                self.gan_opt = SGD(learning_rate=self.gan_lr)
            elif self.gan_opt_n == 'sgdm':
                self.gan_opt = SGD(learning_rate=self.gan_lr, momentum=0.5)
            elif self.gan_opt_n == 'nsgd':
                self.gan_opt = SGD(learning_rate=self.gan_lr, momentum=0.5, nesterov=True)
            elif self.gan_opt_n == 'rmsprop':
                self.gan_opt = RMSprop(learning_rate=self.gan_lr)
            elif self.gan_opt_n == 'rmsprop_cent':
                self.gan_opt = RMSprop(learning_rate=self.gan_lr, centered=True)
            elif self.gan_opt_n == 'adam':
                self.gan_opt = Adam(learning_rate=self.gan_lr, beta_1=0.5)
            elif self.gan_opt_n == 'adam_ams':
                self.gan_opt = Adam(learning_rate=self.gan_lr, beta_1=0.5, amsgrad=True)
            elif self.gan_opt_n == 'adamax':
                self.gan_opt = Adamax(learning_rate=self.gan_lr, beta_1=0.5)
            elif self.gan_opt_n == 'adamax_ams':
                self.gan_opt = Adamax(learning_rate=self.gan_lr, beta_1=0.5, amsgrad=True)
            elif self.gan_opt_n == 'nadam':
                self.gan_opt = Nadam(learning_rate=self.gan_lr, beta_1=0.5)
            if self.c_dim > 0 and self.u_dim > 0:
                gan_loss = {'discriminator': dsc_loss,
                            'auxiliary': 'categorical_crossentropy',
                            'auxiliary_1': 'mean_squared_error'}
                gan_loss_weights = {'discriminator': 1.0,
                                    'auxiliary': self.lamb,
                                    'auxiliary_1': self.lamb}
            elif self.c_dim > 0:
                gan_loss = {'discriminator': dsc_loss,
                            'auxiliary': 'categorical_crossentropy'}
                gan_loss_weights = {'discriminator': 1.0,
                                    'auxiliary': self.lamb}
            elif self.u_dim > 0:
                gan_loss = {'discriminator': dsc_loss,
                            'auxiliary': 'mean_squared_error'}
                gan_loss_weights = {'discriminator': 1.0,
                                    'auxiliary': self.lamb}
            # compile GAN
            self.gan.compile(loss=gan_loss, loss_weights=gan_loss_weights, optimizer=self.gan_opt)
        self.discriminator.trainable = True


    def sample_latent_distribution(self, num_samples=None):
        ''' draws samples from the latent gaussian and categorical distributions '''
        if num_samples is None:
            num_samples = self.batch_size
        # noise
        z = sample_gaussian(num_samples, self.z_dim)
        output = [z]
        # categorical control variable
        if self.c_dim > 0:
            c = sample_categorical(num_samples, self.c_dim)
            output.append(c)
        # continuous control variable
        if self.u_dim > 0:
            u = sample_uniform(-1.0, 1.0, num_samples, self.u_dim)
            output.append(u)
        return output


    def generate(self, num_samples=None, verbose=False):
        ''' generate new configurations using samples from the latent distributions '''
        if num_samples is None:
            num_samples = self.batch_size
        # sample latent space
        inputs = self.sample_latent_distribution(num_samples)
        # generate configurations
        return self.generator.predict(inputs, batch_size=self.batch_size, verbose=verbose)


    def generate_controlled(self, c, u, num_samples=None, verbose=False):
        ''' generate new configurations using control variables '''
        if num_samples is None:
            num_samples = self.batch_size
        # sample latent space
        z = self.sample_latent_distribution(num_samples)[0]
        inputs = [z]
        if self.c_dim > 0:
            c = np.tile(c, (num_samples, 1))
            inputs.append(c)
        if self.u_dim > 0:
            u = np.tile(u, (num_samples, 1))
            inputs.append(u)
        # generate configurations
        return self.generator.predict(inputs, batch_size=self.batch_size, verbose=verbose)


    def discriminate(self, x_batch, verbose=False):
        ''' discriminate input configurations '''
        return self.discriminator.predict(x_batch, batch_size=self.batch_size, verbose=verbose)


    def get_aux_dist(self, x_batch, verbose=False):
        ''' predict categorical assignments of input configurations '''
        return self.auxiliary.predict(x_batch, batch_size=self.batch_size, verbose=verbose)


    def model_summaries(self):
        ''' print model summaries '''
        self.generator.summary()
        self.discriminator.summary()
        self.auxiliary.summary()
        if self.wasserstein:
            self.gan_dsc.summary()
            self.gan_aux.summary()
        else:
            self.gan.summary()


    def save_weights(self, name, lattice_sites, interval, num_samples, scaled, seed):
        ''' save weights to file '''
        # file parameters
        params = (name, lattice_sites, interval, num_samples, scaled, seed)
        if self.wasserstein:
            file_name_gen = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.gen.weights.h5'
            file_name_dsc = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.dsc.weights.h5'
            file_name_aux = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.aux.weights.h5'
            self.generator.save_weights(file_name_gen)
            self.discriminator.save_weights(file_name_dsc)
            self.auxiliary.save_weights(file_name_aux)
        else:
            file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.gan.weights.h5'
            # save weights
            self.gan.save_weights(file_name)


    def load_weights(self, name, lattice_sites, interval, num_samples, scaled, seed):
        ''' load weights from file '''
        # file parameters
        params = (name, lattice_sites, interval, num_samples, scaled, seed)
        if self.wasserstein:
            file_name_gen = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.gen.weights.h5'
            file_name_dsc = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.dsc.weights.h5'
            file_name_aux = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.aux.weights.h5'
            self.generator.load_weights(file_name_gen, by_name=True)
            self.discriminator.load_weights(file_name_dsc, by_name=True)
            self.auxiliary.load_weights(file_name_aux, by_name=True)
        else:
            file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.gan.weights.h5'
            # load weights
            self.gan.load_weights(file_name, by_name=True)


    def get_losses(self):
        ''' retrieve loss histories '''
        # reshape arrays into (epochs, batches)
        dsc_fake_loss = np.array(self.dsc_fake_loss_history).reshape(-1, self.num_batches)
        dsc_real_loss = np.array(self.dsc_real_loss_history).reshape(-1, self.num_batches)
        gan_loss = np.array(self.gan_loss_history).reshape(-1, self.num_batches)
        ent_cat_loss = np.array(self.ent_cat_loss_history).reshape(-1, self.num_batches)
        ent_con_loss = np.array(self.ent_con_loss_history).reshape(-1, self.num_batches)
        return dsc_fake_loss, dsc_real_loss, gan_loss, ent_cat_loss, ent_con_loss


    def save_losses(self, name, lattice_sites, interval, num_samples, scaled, seed):
        ''' save loss histories to file '''
        # retrieve losses
        losses = self.get_losses()
        # file parameters
        params = (name, lattice_sites, interval, num_samples, scaled, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.loss.npy'
        np.save(file_name, np.stack(losses, axis=-1))


    def load_losses(self, name, lattice_sites, interval, num_samples, scaled, seed):
        ''' load loss histories from file '''
        # file parameters
        params = (name, lattice_sites, interval, num_samples, scaled, seed)
        file_name = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.loss.npy'
        losses = np.load(file_name)
        # set past epochs
        self.past_epochs = losses.shape[0]
        self.num_batches = losses.shape[1]
        # change loss histories into lists
        self.dsc_fake_loss_history = list(losses[:, :, 0].reshape(-1))
        self.dsc_real_loss_history = list(losses[:, :, 1].reshape(-1))
        self.gan_loss_history = list(losses[:, :, 2].reshape(-1))
        self.ent_cat_loss_history = list(losses[:, :, 3].reshape(-1))
        self.ent_con_loss_history = list(losses[:, :, 4].reshape(-1))


    def initialize_checkpoint_managers(self, name, lattice_sites, interval, num_samples, scaled, seed):
        ''' initialize training checkpoint managers '''
        # initialize checkpoints
        self.dsc_ckpt = Checkpoint(step=tf.Variable(0), optimizer=self.dsc_opt, net=self.discriminator)
        if self.wasserstein:
            self.gan_dsc_ckpt = Checkpoint(step=tf.Variable(0), optimizer=self.gan_dsc_opt, net=self.gan_dsc)
            self.gan_aux_ckpt = Checkpoint(step=tf.Variable(0), optimizer=self.gan_aux_opt, net=self.gan_aux)
        else:
            self.gan_ckpt = Checkpoint(step=tf.Variable(0), optimizer=self.gan_opt, net=self.gan)
        # file parameters
        params = (name, lattice_sites, interval, num_samples, scaled, seed)
        directory = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.ckpts'
        # initialize checkpoint managers
        self.dsc_mngr = CheckpointManager(self.dsc_ckpt, directory+'/discriminator/', max_to_keep=4)
        if self.wasserstein:
            self.gan_dsc_mngr = CheckpointManager(self.gan_dsc_ckpt, directory+'/gan/discriminator', max_to_keep=4)
            self.gan_aux_mngr = CheckpointManager(self.gan_aux_ckpt, directory+'/gan/auxiliary', max_to_keep=4)
        else:
            self.gan_mngr = CheckpointManager(self.gan_ckpt, directory+'/gan/', max_to_keep=4)


    def load_latest_checkpoint(self, name, lattice_sites, interval, num_samples, scaled, seed):
        ''' load latest training checkpoint from file '''
        # initialize checkpoint managers
        self.initialize_checkpoint_managers(name, lattice_sites, interval, num_samples, scaled, seed)
        self.load_losses(name, lattice_sites, interval, num_samples, scaled, seed)
        # file parameters
        params = (name, lattice_sites, interval, num_samples, scaled, seed)
        directory = os.getcwd()+'/{}.{}.{}.{}.{:d}.{}.'.format(*params)+self.get_file_prefix()+'.ckpts'
        # restore checkpoints
        self.dsc_ckpt.restore(self.dsc_mngr.latest_checkpoint).assert_consumed()
        if self.wasserstein:
            self.gan_dsc_ckpt.restore(self.gan_dsc_mngr.latest_checkpoint).assert_consumed()
            self.gan_aux_ckpt.restore(self.gan_aux_mngr.latest_checkpoint).assert_consumed()
        else:
            self.gan_ckpt.restore(self.gan_mngr.latest_checkpoint).assert_consumed()


    def train_discriminator(self, x_batch, real=False):
        ''' train discriminator '''
        if real:
            target = np.ones(self.batch_size, dtype=np.float32)
            # label smoothing
            if self.alpha > 0:
                target -= np.random.uniform(low=0, high=self.alpha, size=self.batch_size)
            if self.wasserstein:
                target *= -1.0
            # label randomizing
            if self.beta > 0:
                flp_size = np.int32(self.beta*self.batch_size)
                flp_ind = np.random.choice(self.batch_size, size=flp_size)
                if self.wasserstein:
                    target[flp_ind] = np.ones(flp_size, dtype=np.float32)
                else:
                    target[flp_ind] = np.zeros(flp_size, dtype=np.float32)
            # discriminator loss
            dsc_loss = self.discriminator.train_on_batch(x_batch, target)
        else:
            if self.wasserstein:
                target = np.ones(self.batch_size, dtype=np.float32)
            else:
                target = np.zeros(self.batch_size, dtype=np.float32)
            # discriminator loss
            dsc_loss = self.discriminator.train_on_batch(x_batch, target)
        return dsc_loss


    def train_generator(self, z_sample):
        ''' train generator and auxiliary '''
        # inputs are true samples, so the discrimination targets are of unit value
        target = np.ones(self.batch_size, dtype=np.float32)
        gan_loss = np.zeros(3)
         # GAN and entropy losses
        if self.wasserstein:
            target *= -1
            gan_dsc_loss = self.gan_dsc.train_on_batch(z_sample, target)
            gan_aux_loss = self.gan_aux.train_on_batch(z_sample, z_sample[1:])
            gan_loss[0] = gan_dsc_loss
            if self.c_dim > 0 and self.u_dim > 0:
                gan_loss[1:] = gan_aux_loss[1:]
            elif self.c_dim > 0:
                gan_loss[1] = gan_aux_loss
                gan_loss[2] = 0
            elif self.u_dim > 0:
                gan_loss[1] = 0
                gan_loss[2] = gan_aux_loss
        else:
            gan_dsc_aux_loss = self.gan.train_on_batch(z_sample, (target, *z_sample[1:]))
            if self.c_dim > 0 and self.u_dim > 0:
                gan_loss[i] = gan_dsc_aux_loss[1:]
            elif self.c_dim > 0:
                gan_loss[0] = gan_dsc_aux_loss[1]
                gan_loss[1] = gan_dsc_aux_loss[2]
                gan_loss[2] = 0
            elif self.u_dim > 0:
                gan_loss[0] = gan_dsc_aux_loss[1]
                gan_loss[1] = 0
                gan_loss[2] = gan_dsc_aux_loss[2]
        return gan_loss


    def train_infogan(self, x_batch, n_dsc, n_gan):
        ''' train infoGAN '''
        z_sample = self.sample_latent_distribution(num_samples=self.batch_size)
        x_generated = self.generator.predict(x=z_sample, batch_size=self.batch_size)
        dsc_real_loss = np.zeros(n_dsc)
        dsc_fake_loss = np.zeros(n_dsc)
        gan_loss = np.zeros((n_gan, 3))
        for i in range(n_dsc):
            dsc_real_loss[i] = self.train_discriminator(x_batch=x_batch, real=True)
            dsc_fake_loss[i] = self.train_discriminator(x_batch=x_generated, real=False)
        for i in range(n_gan):
            gan_loss[i] = self.train_generator(z_sample=z_sample)
        self.dsc_real_loss_history.append(dsc_real_loss.mean())
        self.dsc_fake_loss_history.append(dsc_fake_loss.mean())
        self.gan_loss_history.append(gan_loss[:, 0].mean())
        self.ent_cat_loss_history.append(gan_loss[:, 1].mean())
        self.ent_con_loss_history.append(gan_loss[:, 2].mean())


    def rolling_loss_average(self, epoch, batch):
        ''' calculate rolling loss averages over batches during training '''
        epoch = epoch+self.past_epochs
        # catch case where there are no calculated losses yet
        if batch == 0:
            gan_loss = 0
            dscf_loss = 0
            dscr_loss = 0
            ent_cat_loss = 0
            ent_con_loss = 0
        # calculate rolling average
        else:
            # start index for current epoch
            start = self.num_batches*epoch
            # stop index for current batch (given epoch)
            stop = self.num_batches*epoch+batch+1
            # average loss histories
            gan_loss = np.mean(self.gan_loss_history[start:stop])
            dscf_loss = np.mean(self.dsc_fake_loss_history[start:stop])
            dscr_loss = np.mean(self.dsc_real_loss_history[start:stop])
            # only calculate categorical control loss if the dimension is nonzero
            if self.c_dim > 0:
                ent_cat_loss = np.mean(self.ent_cat_loss_history[start:stop])
            else:
                ent_cat_loss = 0
            # only calculate continuous control loss if the dimension is nonzero
            if self.u_dim > 0:
                ent_con_loss = np.mean(self.ent_con_loss_history[start:stop])
            else:
                ent_con_loss = 0
        return gan_loss, dscf_loss, dscr_loss, ent_cat_loss, ent_con_loss


    def fit(self, x_train, num_epochs=4, save_step=None, random_sampling=False, verbose=False):
        ''' fit model '''
        self.num_temp_x, self.num_temp_y, self.num_samples, _, _, = x_train.shape
        self.num_batches = (self.num_temp_x*self.num_temp_y*self.num_samples)//self.batch_size
        if random_sampling:
            # x_train = extract_unique_data(x_train, self.num_temp_x, self.num_temp_y, self.num_samples, self.input_shape)
            x_train = x_train.reshape(self.num_temp_x*self.num_temp_y*self.num_samples, *self.input_shape)
        else:
            x_train = reorder_training_data(x_train, self.num_temp_x, self.num_temp_y, self.num_samples, self.input_shape, self.batch_size)
        num_epochs += self.past_epochs
        # loop through epochs
        mode = 'loss'
        for i in range(self.past_epochs, num_epochs):
            # construct progress bar for current epoch
            if random_sampling:
                batch_range = trange(self.num_batches, desc='', disable=not verbose)
            else:
                b = np.arange(self.num_batches)
                np.random.shuffle(b)
                batch_range = tqdm(b, desc='', disable=not verbose)
            # loop through batches
            u = 0
            for j in batch_range:
                # set batch loss description
                batch_loss = self.rolling_loss_average(i, u)
                batch_acc = np.exp(-np.array(batch_loss[:-1]))
                if mode == 'loss':
                    desc = 'Epoch: {}/{} GAN Lss: {:.4f} DSCF Lss: {:.4f} DSCR Lss: {:.4f} CAT Lss: {:.4f} CON Lss: {:.4f}'.format(i+1, num_epochs, *batch_loss)
                elif mode == 'accuracy':
                    desc = 'Epoch: {}/{} GAN Acc: {:.4f} DSCF Acc: {:.4f} DSCR Acc: {:.4f} CAT Acc: {:.4f} CON Lss: {:.4f}'.format(i+1, num_epochs, *batch_acc, batch_loss[-1])
                batch_range.set_description(desc)
                # fetch batch
                if random_sampling:
                    x_batch = draw_random_batch(x_train, self.batch_size)
                else:
                    x_batch = draw_indexed_batch(x_train, self.batch_size, j)
                # train infogan on batch
                self.train_infogan(x_batch, self.n_dsc, self.n_gan)
                u += 1
            # if checkpoint managers are initialized
            if self.dsc_mngr is not None and self.gan_mngr is not None:
                # increment checkpoints
                self.dsc_ckpt.step.assign_add(1)
                self.gan_ckpt.step.assign_add(1)
                # if save step is reached
                if np.int32(self.dsc_ckpt.step) % save_step == 0:
                    # save model checkpoint
                    dsc_save_path = self.dsc_mngr.save()
                    gan_save_path = self.gan_mngr.save()
                    print('Checkpoint DSC: {}'.format(dsc_save_path))
                    print('Checkpoint GAN: {}'.format(gan_save_path))

if __name__ == '__main__':
    (VERBOSE, RSTRT, PLOT, PARALLEL, GPU, THREADS,
     NAME, N, I, NS, SC, W, CP,
     CN, FBL, FBS, FB, FL, FS, FF,
     GD, DD, ZD, CD, UD,
     KI, AN,
     DOPT, GOPT, DLR, GLR,
     GLAMB, TALPHA, TBETA, DC, GC,
     BS, RS, EP, SEED) = parse_args()

    TX, TY, CONF, THRM = load_data(NAME, N, I, NS, SC, SEED, VERBOSE)
    del THRM
    NTX, NTY = TX.size, TY.size
    IS = (N, N, 1)

    if SEED == -1:
        np.random.seed(None)
    else:
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
    if GPU:
        DEVICE = '/GPU:0'
    else:
        if not PARALLEL:
            THREADS = 1
        DEVICE = '/CPU:0'
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(THREADS)
    tf.device(DEVICE)

    K.clear_session()
    MDL = InfoGAN(IS, SC, W, CP, CN, FBL, FBS, FB, FL, FS, FF, GD, DD, ZD, CD, UD, KI, AN, DOPT, GOPT, DLR, GLR, GLAMB, BS, TALPHA, TBETA, DC, GC)
    PRFX = MDL.get_file_prefix()
    if RSTRT:
        MDL.load_losses(NAME, N, I, NS, SC, SEED)
        MDL.load_weights(NAME, N, I, NS, SC, SEED)
        if VERBOSE:
            MDL.model_summaries()
        # MDL.load_latest_checkpoint(NAME, N, I, NS, SC, SEED)
        MDL.fit(CONF, num_epochs=EP, save_step=EP, random_sampling=RS, verbose=VERBOSE)
        MDL.save_losses(NAME, N, I, NS, SC, SEED)
        MDL.save_weights(NAME, N, I, NS, SC, SEED)
    else:
        try:
            MDL.load_losses(NAME, N, I, NS, SC, SEED)
            MDL.load_weights(NAME, N, I, NS, SC, SEED)
            if VERBOSE:
                MDL.model_summaries()
        except:
            if VERBOSE:
                MDL.model_summaries()
            # MDL.initialize_checkpoint_managers(NAME, N, I, NS, SC, SEED)
            MDL.fit(CONF, num_epochs=EP, save_step=EP, random_sampling=RS, verbose=VERBOSE)
            MDL.save_losses(NAME, N, I, NS, SC, SEED)
            MDL.save_weights(NAME, N, I, NS, SC, SEED)
    L = MDL.get_losses()
    if CD > 0 and UD > 0:
        C, U = MDL.get_aux_dist(CONF.reshape(-1, *IS), VERBOSE)
        C = C.reshape(NTX, NTY, NS, CD)
        U = U.reshape(NTX, NTY, NS, UD)
        save_output_data(C, 'categorical_control', NAME, N, I, NS, SC, SEED, PRFX)
        save_output_data(U, 'continuous_control', NAME, N, I, NS, SC, SEED, PRFX)
    elif CD > 0:
        C = MDL.get_aux_dist(CONF.reshape(-1, *IS), VERBOSE)
        U = np.zeros((NTX, NTY, NS, UD))
        C = C.reshape(NTX, NTY, NS, CD)
        save_output_data(C, 'categorical_control', NAME, N, I, NS, SC, SEED, PRFX)
    elif UD > 0:
        U = MDL.get_aux_dist(CONF.reshape(-1, *IS), VERBOSE)
        C = np.zeros((NTX, NTY, NS, UD))
        U = U.reshape(NTX, NTY, NS, UD)
        save_output_data(U, 'continuous_control', NAME, N, I, NS, SC, SEED, PRFX)