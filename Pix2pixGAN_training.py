import numpy as np
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
import time
import matplotlib
matplotlib.use('Agg')


# define the discriminator model
def define_discriminator(dynamic_datas_shape, image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=dynamic_datas_shape)
    # reshape the dynamic datas
    d_ = Dense(60*60)(in_src_image)
    d_ = LeakyReLU(alpha=0.2)(d_)
    d_ = Reshape((60, 60, 1))(d_)
    # target image input
    in_target_image = Input(shape=image_shape)
    merged = Concatenate()([d_, in_target_image])
    # C64
    d = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (3, 3), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (3, 3), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True, strides=(2, 2)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (3, 3), strides, padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True, strides=(2, 2)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (3, 3), strides, padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.2)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator(dynamic_datas_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=dynamic_datas_shape)
    # reshape the dynamic datas
    d_ = Dense(60 * 60)(in_image)
    d_ = LeakyReLU(alpha=0.2)(d_)
    d_ = Reshape((60, 60, 1))(d_)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(d_, 32, batchnorm=False)
    e2 = define_encoder_block(e1, 64)
    e3 = define_encoder_block(e2, 128, strides=(3, 3))

    # bottleneck, no batch norm and relu
    b = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(e3)
    b = Activation('relu')(b)

    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d3 = decoder_block(b, e3, 128, strides=(1, 1))
    d4 = decoder_block(d3, e2, 64, strides=(3, 3))
    d5 = decoder_block(d4, e1, 32)

    # output
    g = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5)
    g = Activation('tanh')(g)
    # g = Activation('sigmoid')(g)

    out_image = g
    # define model
    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, dynamic_datas_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):  
            layer.trainable = False
    # define the source image
    in_src = Input(shape=dynamic_datas_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    # loss = adversarial loss + lambda * L1 loss  L1 = 1-100
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 40])
    return model


# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    X2 = X2 - 1
    X2[X2 == 0] = 1
    # log
    X2 = np.log(X2)
    # Outlier Processing
    # Reset inactive cells
    X2[X2 == 0] = 100
    # Set lower bound (No upper bound is set for this code)
    X2[X2 < 4] = 4
    # Restore inactive cells
    X2[X2 == 100] = 4
    print('min: ', np.amin(X2))
    print('max: ', np.amax(X2))
    # scale X2 to [-1,1]
    X2 = (X2 - np.amin(X2))/(np.amax(X2) - np.amin(X2))*2 - 1
    return [X1, X2]


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=4):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    print("X_realB shape", X_realB.shape)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    print("X_fakeB shape", X_fakeB.shape)
    # scale all pixels from [-1,1] to [0,1]
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    titles = ['Gen_GAN', 'Gen_GAN', 'Gen_GAN', 'Gen_GAN',
              'Expected', 'Expected', 'Expected', 'Expected']
    fig, ax = pyplot.subplots(2, 4, figsize=(21, 11))
    ax = ax.flatten()
    a = []
    # plot generated target image
    for i in range(n_samples):
        a.append(ax[i].imshow(X_fakeB[i], pyplot.get_cmap('jet'), vmin=0, vmax=1))
        ax[i].axis()
        # show title
        ax[i].set_title(titles[i], fontsize=18)
        fig.colorbar(a[i], ax=ax[i], fraction=0.046, pad=0.046)
    # plot real target image
    for i in range(n_samples):
        a.append(ax[i+n_samples].imshow(X_realB[i], pyplot.get_cmap('jet'), vmin=0, vmax=1))
        ax[i+n_samples].axis()
        # show title
        ax[i+n_samples].set_title(titles[n_samples + i], fontsize=18)
        fig.colorbar(a[i+n_samples], ax=ax[i+n_samples], fraction=0.046, pad=0.046)

    # save plot to file
    pyplot.subplots_adjust(wspace=0.3, hspace=0.2)
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1, dpi=400, bbox_inches='tight', pad_inches=0)
    pyplot.close()
    
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=400, n_batch=4):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # train
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch) #100/1 =100
    # calculate the number of training iterations.
    n_steps = bat_per_epo * n_epochs #100*500=50000
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo*100) == 0:
            summarize_performance(i, g_model, dataset)
 

# time start
time_start = time.time()
# load image data
dataset = load_real_samples('EGG_d_perm.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

# define input shape based on the loaded dataset
dynamic_datas_shape = 400
image_shape = (60, 60, 1)
# define the models
d_model = define_discriminator(dynamic_datas_shape, image_shape)
g_model = define_generator(dynamic_datas_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, dynamic_datas_shape)
# train model
train(d_model, g_model, gan_model, dataset)
# time end
time_end = time.time()
print('time cost:', time_end-time_start, 's')
