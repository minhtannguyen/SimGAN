"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).

Note: Only Python 3 support currently.
"""

import os
import sys
import keras
from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import h5py
import numpy as np
import tensorflow as tf

print('tf-version',tf.__version__, 'keras-version', keras.__version__)

data_dir = '/home/ubuntu/data/eye-gaze'
cache_dir = '.'

# load the data file and extract dimensions
with h5py.File(os.path.join(data_dir,'gaze.h5'),'r') as t_file:
    print(list(t_file.keys()))
    assert 'image' in t_file, "Images are missing"
    assert 'look_vec' in t_file, "Look vector is missing"
    assert 'path' in t_file, "Paths are missing"
    print('Synthetic images found:',len(t_file['image']))
    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        print('image',ikey,'shape:',ival.shape)
        img_height, img_width = ival.shape
        img_channels = 1
    syn_image_stack = np.stack([np.expand_dims(a,-1) for a in t_file['image'].values()],0)

with h5py.File(os.path.join(data_dir,'real_gaze.h5'),'r') as t_file:
    print(list(t_file.keys()))
    assert 'image' in t_file, "Images are missing"
    print('Real Images found:',len(t_file['image']))
    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        print('image',ikey,'shape:',ival.shape)
        img_height, img_width = ival.shape
        img_channels = 1
    real_image_stack = np.stack([np.expand_dims(a,-1) for a in t_file['image'].values()],0)

#
# training params
#

nb_steps = 10000 # originally 10000, but this makes the kernel time out
batch_size = 49
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step
log_interval = 100
pre_steps = 15 # for pretraining

# Utils functions
"""
Module to plot a batch of images along w/ their corresponding label(s)/annotations and save the plot to disc.

Use cases:
Plot images along w/ their corresponding ground-truth label & model predicted label,
Plot images generated by a GAN along w/ any annotations used to generate these images,
Plot synthetic, generated, refined, and real images and see how they compare as training progresses in a GAN,
etc...
"""
import matplotlib
from matplotlib import pyplot as plt

from itertools import groupby
from skimage.util.montage import montage2d


def plot_batch(image_batch, figure_path, label_batch=None):
    all_groups = {label: montage2d(np.stack([img[:, :, 0] for img, lab in img_lab_list], 0))
                  for label, img_lab_list in groupby(zip(image_batch, label_batch), lambda x: x[1])}
    fig, c_axs = plt.subplots(1, len(all_groups), figsize=(len(all_groups) * 4, 8), dpi=600)
    for c_ax, (c_label, c_mtg) in zip(c_axs, all_groups.items()):
        c_ax.imshow(c_mtg, cmap='bone')
        c_ax.set_title(c_label)
        c_ax.axis('off')
    fig.savefig(os.path.join(figure_path))
    plt.close()


"""
Module implementing the image history buffer described in `2.3. Updating Discriminator using a History of
Refined Images` of https://arxiv.org/pdf/1612.07828v1.pdf.

"""


class ImageHistoryBuffer(object):
    def __init__(self, shape, max_size, batch_size):
        """
        Initialize the class's state.

        :param shape: Shape of the data to be stored in the image history buffer
                      (i.e. (0, img_height, img_width, img_channels)).
        :param max_size: Maximum number of images that can be stored in the image history buffer.
        :param batch_size: Batch size used to train GAN.
        """
        self.image_history_buffer = np.zeros(shape=shape)
        self.max_size = max_size
        self.batch_size = batch_size

    def add_to_image_history_buffer(self, images, nb_to_add=None):
        """
        To be called during training of GAN. By default add batch_size // 2 images to the image history buffer each
        time the generator generates a new batch of images.

        :param images: Array of images (usually a batch) to be added to the image history buffer.
        :param nb_to_add: The number of images from `images` to add to the image history buffer
                          (batch_size / 2 by default).
        """
        if not nb_to_add:
            nb_to_add = self.batch_size // 2

        if len(self.image_history_buffer) < self.max_size:
            np.append(self.image_history_buffer, images[:nb_to_add], axis=0)
        elif len(self.image_history_buffer) == self.max_size:
            self.image_history_buffer[:nb_to_add] = images[:nb_to_add]
        else:
            assert False

        np.random.shuffle(self.image_history_buffer)

    def get_from_image_history_buffer(self, nb_to_get=None):
        """
        Get a random sample of images from the history buffer.

        :param nb_to_get: Number of images to get from the image history buffer (batch_size / 2 by default).
        :return: A random sample of `nb_to_get` images from the image history buffer, or an empty np array if the image
                 history buffer is empty.
        """
        if not nb_to_get:
            nb_to_get = self.batch_size // 2

        try:
            return self.image_history_buffer[:nb_to_get]
        except IndexError:
            return np.zeros(shape=0)


# Network architectures
def refiner_network(input_image_tensor):
    """
    The refiner network, R, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.

    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same', weights=[1e-2*np.ones([nb_kernel_rows, nb_kernel_cols, 64, nb_features]), np.zeros(nb_features)])(input_features)
        y = layers.Activation('relu')(y)
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same', weights=[1e-2*np.ones([nb_kernel_rows, nb_kernel_cols, nb_features, nb_features]), np.zeros(nb_features)])(y)

        y = layers.merge([input_features, y], mode='sum')
        return layers.Activation('relu')(y)

    # an input image of size w x h is convolved with 3 x 3 filters that output 64 feature maps
    x = layers.Convolution2D(64, 3, 3, border_mode='same', activation='relu', weights=[1e-2*np.ones([3, 3, 1, 64]), np.zeros(64)])(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    for _ in range(4):
        x = resnet_block(x)

    # the output of the last ResNet block is passed to a 1 x 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image
    # return x
    return layers.Convolution2D(img_channels, 1, 1, border_mode='same', activation='tanh', weights=[1e-2*np.ones([1, 1, 64, img_channels]), np.zeros(img_channels)])(x)

def discriminator_network(input_image_tensor):
    """
    The discriminator network, D, contains 5 convolution layers and 2 max-pooling layers.

    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: Output tensor that corresponds to the probability of whether an image is real or refined.
    """
    x = layers.Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2), activation='relu',
                             weights=[1e-2*np.ones([3, 3, img_channels, 96]), np.zeros(96)])(input_image_tensor)
    x = layers.Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu',
                             weights=[1e-2*np.ones([3, 3, 96, 64]), np.zeros(64)])(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same', strides=(1, 1))(x)
    x = layers.Convolution2D(32, 3, 3, border_mode='same', subsample=(1, 1), activation='relu',
                             weights=[1e-2*np.ones([3, 3, 64, 32]), np.zeros(32)])(x)
    x = layers.Convolution2D(32, 1, 1, border_mode='same', subsample=(1, 1), activation='relu',
                             weights=[1e-2*np.ones([1, 1, 32, 32]), np.zeros(32)])(x)
    x = layers.Convolution2D(2, 1, 1, border_mode='same', subsample=(1, 1), activation='relu',
                             weights=[1e-2 * np.ones([1, 1, 32, 2]), np.zeros(2)])(x)

    # here one feature map corresponds to `is_real` and the other to `is_refined`,
    # and the custom loss function is then `tf.nn.sparse_softmax_cross_entropy_with_logits`
    return layers.Reshape((-1, 2))(x)

# Combining models
#
# define model input and output tensors
#

synthetic_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
refined_image_tensor = refiner_network(synthetic_image_tensor)

refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
discriminator_output = discriminator_network(refined_or_real_image_tensor)

#
# define models
#

refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
discriminator_model = models.Model(input=refined_or_real_image_tensor, output=discriminator_output,
                                   name='discriminator')

# combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
refiner_model_output = refiner_model(synthetic_image_tensor)
combined_output = discriminator_model(refiner_model_output)
combined_model = models.Model(input=synthetic_image_tensor, output=[refiner_model_output, combined_output],
                              name='combined')

discriminator_model_output_shape = discriminator_model.output_shape

print(refiner_model.summary())
print(discriminator_model.summary())
print(combined_model.summary())

# Visualization
from keras.utils.visualize_util import model_to_dot
from IPython.display import SVG
# Define model
try:
    model_to_dot(refiner_model, show_shapes=True).write_svg('refiner_model.svg')
    SVG('refiner_model.svg')
except ImportError:
    print('Not running the patched version of keras/pydot!')
    pass

try:
    model_to_dot(discriminator_model, show_shapes=True).write_svg('discriminator_model.svg')
    SVG('discriminator_model.svg')
except ImportError:
    pass

# Losses

#
# define custom l1 loss function for the refiner
#

def self_regularization_loss(y_true, y_pred):
    delta = 0.0000  # FIXME: need to figure out an appropriate value for this
    return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

#
# define custom local adversarial loss (softmax for each image section) for the discriminator
# the adversarial loss function is the sum of the cross-entropy losses over the local patches
#

def local_adversarial_loss(y_true, y_pred):
    # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
    # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
    y_true = tf.reshape(y_true, (-1, 2))
    y_pred = tf.reshape(y_pred, (-1, 2))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    return tf.reduce_mean(loss)

# Compile the model
sgd = optimizers.SGD(lr=1e-3)

refiner_model.compile(optimizer=sgd, loss=self_regularization_loss)
discriminator_model.compile(optimizer=sgd, loss=local_adversarial_loss)
discriminator_model.trainable = False
combined_model.compile(optimizer=sgd, loss=[self_regularization_loss, local_adversarial_loss])

refiner_model_path = None
discriminator_model_path = None

# Set up the pipeline to feed new images to both modelsqui
datagen = image.ImageDataGenerator(
    preprocessing_function=applications.xception.preprocess_input,
    dim_ordering='tf')

flow_from_directory_params = {'target_size': (img_height, img_width),
                              'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                              'class_mode': None,
                              'batch_size': batch_size}
flow_params = {'batch_size': batch_size}

synthetic_generator = datagen.flow(
    X = syn_image_stack,
    seed = 1,
    **flow_params
)

real_generator = datagen.flow(
    X = real_image_stack,
    seed = 1,
    **flow_params
)

def get_image_batch(generator):
    """keras generators may generate an incomplete batch for the last batch"""
    img_batch = generator.next()
    if len(img_batch) != batch_size:
        img_batch = generator.next()

    assert len(img_batch) == batch_size

    return img_batch

# the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
# assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)
# batch_out = get_image_batch(synthetic_generator)
# assert batch_out.shape == (batch_size, img_height, img_width, img_channels), \
#     "Image Dimensions do not match, {}!={}".format(batch_out.shape, (batch_size, img_height, img_width, img_channels))

# Pretraining
if not refiner_model_path:
    # we first train the R network with just self-regularization loss for 1,000 steps
    print('pre-training the refiner network...')
    gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

    for i in range(pre_steps):
        synthetic_image_batch = get_image_batch(synthetic_generator)
        gen_loss = np.add(refiner_model.train_on_batch(synthetic_image_batch, synthetic_image_batch), gen_loss)

        # log every `log_interval` steps
        if not i % log_interval:
            figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(i)
            print('Saving batch of refined images during pre-training at step: {}.'.format(i))

            synthetic_image_batch = get_image_batch(synthetic_generator)
            refined_s_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
            print('Refined Synthetic_Image_Batch')
            print(np.min(refined_s_image_batch))
            print(np.mean(refined_s_image_batch))
            print(np.max(refined_s_image_batch))

            print('Synthetic_Image_Batch')
            print(np.min(synthetic_image_batch))
            print(np.mean(synthetic_image_batch))
            print(np.max(synthetic_image_batch))

            plot_batch(
                np.concatenate((synthetic_image_batch, refiner_model.predict_on_batch(synthetic_image_batch))),
                os.path.join(cache_dir, figure_name),
                label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

            real_image_batch = get_image_batch(real_generator)
            print('Real_Image_Batch')
            print(np.min(real_image_batch))
            print(np.mean(real_image_batch))
            print(np.max(real_image_batch))

            # plot_batch(
            #     np.concatenate((synthetic_image_batch, real_image_batch)),
            #     os.path.join(cache_dir, 'real_image_batch_pre_train_step_{}.png'.format(i)),
            #     label_batch=['Synthetic'] * batch_size + ['Real'] * batch_size)

    refiner_model.save(os.path.join(cache_dir, 'refiner_model_pre_trained.h5'))
    print('Refiner model self regularization loss: {}.'.format(gen_loss / pre_steps))
else:
    refiner_model.load_weights(refiner_model_path)

if not discriminator_model_path:
    # and D for 200 steps (one mini-batch for refined images, another for real)
    print('pre-training the discriminator network...')
    disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

    for _ in range(pre_steps):
        real_image_batch = get_image_batch(real_generator)
        doutr = discriminator_model.predict_on_batch(real_image_batch)
        dlnewr = discriminator_model.train_on_batch(real_image_batch, y_real)
        disc_loss = np.add(dlnewr, disc_loss)

        synthetic_image_batch = get_image_batch(synthetic_generator)
        refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
        douts = discriminator_model.predict_on_batch(refined_image_batch)
        dlnews = discriminator_model.train_on_batch(refined_image_batch, y_refined)
        disc_loss = np.add(dlnews, disc_loss)
        import ipdb; ipdb.set_trace()

    discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))
    print('Discriminator model loss: {}.'.format(disc_loss / (pre_steps * 2)))
else:
    discriminator_model.load_weights(discriminator_model_path)

# Full training
# TODO: what is an appropriate size for the image history buffer?
image_history_buffer = ImageHistoryBuffer((0, img_height, img_width, img_channels), batch_size * 100, batch_size)

combined_loss = np.zeros(shape=len(combined_model.metrics_names))
disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

# see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
for i in range(nb_steps):
    # print('Step: {} of {}.'.format(i, nb_steps))
    # train the refiner
    for _ in range(k_g * 2):
        # sample a mini-batch of synthetic images
        synthetic_image_batch = get_image_batch(synthetic_generator)

        # update by taking an SGD step on mini-batch loss LR
        combined_loss = np.add(combined_model.train_on_batch(synthetic_image_batch,
                                                             [synthetic_image_batch, y_real]), combined_loss)

    for _ in range(k_d):
        # sample a mini-batch of synthetic and real images
        synthetic_image_batch = get_image_batch(synthetic_generator)
        real_image_batch = get_image_batch(real_generator)

        # refine the synthetic images w/ the current refiner
        refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

        # use a history of refined images
        half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
        image_history_buffer.add_to_image_history_buffer(refined_image_batch)

        if len(half_batch_from_image_history):
            refined_image_batch[:batch_size // 2] = half_batch_from_image_history

        # update by taking an SGD step on mini-batch loss LD
        disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
        disc_loss_refined = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined),
                                   disc_loss_refined)

    if not i % log_interval:
        # plot batch of refined images w/ current refiner
        figure_name = 'refined_image_batch_step_{}.png'.format(i)
        print('Saving batch of refined images at adversarial step: {}.'.format(i))

        synthetic_image_batch = get_image_batch(synthetic_generator)
        refined_s_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
        print('Refined Synthetic_Image_Batch')
        print(np.min(refined_s_image_batch))
        print(np.mean(refined_s_image_batch))
        print(np.max(refined_s_image_batch))
        plot_batch(
            np.concatenate((synthetic_image_batch, refiner_model.predict_on_batch(synthetic_image_batch))),
            os.path.join(cache_dir, figure_name),
            label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

        real_image_batch = get_image_batch(real_generator)
        print('Real_Image_Batch')
        print(np.min(real_image_batch))
        print(np.mean(real_image_batch))
        print(np.max(real_image_batch))
        # plot_batch(
        #     np.concatenate((synthetic_image_batch, real_image_batch)),
        #     os.path.join(cache_dir, 'real_image_batch_pre_train_step_{}.png'.format(i)),
        #     label_batch=['Synthetic'] * batch_size + ['Real'] * batch_size)

        # log loss summary
        print('Refiner model loss: {}.'.format(combined_loss / (log_interval * k_g * 2)))
        print('Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * k_d * 2)))
        print('Discriminator model loss refined: {}.'.format(disc_loss_refined / (log_interval * k_d * 2)))

        combined_loss = np.zeros(shape=len(combined_model.metrics_names))
        disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
        disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

        # save model checkpoints
        model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_step_{}.h5')
        refiner_model.save(model_checkpoint_base_name.format('refiner', i))
        discriminator_model.save(model_checkpoint_base_name.format('discriminator', i))