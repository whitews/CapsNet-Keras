"""
Keras implementation of CapsNet in Hinton's paper:
    Dynamic Routing Between Capsules.

The current version maybe only works for TensorFlow backend. Actually it
will be straightforward to re-write to TF code. Adopting to other backends
should be easy, but I have not tested this.

Author: Xifeng Guo
E-mail: `guoxifeng1990@163.com`
Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as k
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import matplotlib.pyplot as plt
from capsnet_keras.utils import combine_images
from PIL import Image
from capsnet_keras.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from capsnet_keras.utils import plot_log

k.set_image_data_format('channels_last')


def build_caps_net(input_shape, n_class, n_routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param n_routings: number of routing iterations
    :return: 2 Keras models, the 1st used for training, the 2nd for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(
        filters=256,
        kernel_size=9,
        strides=1,
        padding='valid',
        activation='relu',
        name='conv1'
    )(x)

    # Layer 2: Conv2D layer with `squash` activation,
    # then reshape to [None, num_capsule, dim_capsule]
    primary_caps = PrimaryCap(
        conv1,
        dim_capsule=8,
        n_channels=32,
        kernel_size=9,
        strides=2,
        padding='valid'
    )

    # Layer 3: Capsule layer. Routing algorithm works here.
    digit_caps = CapsuleLayer(
        num_capsule=n_class,
        dim_capsule=16, routings=n_routings,
        name='digit_caps'
    )(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule
    # with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary.
    out_caps = Length(name='capsnet')(digit_caps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))

    # The true label is used to mask the output of capsule layer. For training
    masked_by_y = Mask()([digit_caps, y])

    # Mask using the capsule with maximal length. For prediction
    masked = Mask()(digit_caps)

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digit_caps = layers.Add()([digit_caps, noise])
    masked_noised_y = Mask()([noised_digit_caps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`,
    this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    loss = y_true * k.square(k.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * k.square(k.maximum(0., y_pred - 0.1))

    return k.mean(k.sum(loss, 1))


def train(
        model,
        data,
        save_dir,
        batch_size,
        epochs,
        lr,
        lr_decay,
        lam_recon,
        shift_fraction,
        debug=False):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data,
                 like `((x_train, y_train), (x_test, y_test))`
    :param save_dir: directory where results are saved
    :param batch_size: batch size
    :param epochs: number of epochs
    :param lr: learning rate
    :param lr_decay: learning rate decay
    :param lam_recon: ?
    :param shift_fraction: number of pixels to shift
    :param debug: turn on debugging
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(save_dir + '/log.csv')
    tb = callbacks.TensorBoard(
        log_dir=save_dir + '/tensorboard-logs',
        batch_size=batch_size,
        histogram_freq=int(debug)
    )
    checkpoint = callbacks.ModelCheckpoint(
        save_dir + '/weights-{epoch:02d}.h5',
        monitor='val_capsnet_acc',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    lr_scheduler = callbacks.LearningRateScheduler(
        schedule=lambda epoch: lr * (lr_decay ** epoch)
    )

    # compile the model
    model.compile(
        optimizer=optimizers.Adam(lr=lr),
        loss=[margin_loss, 'mse'],
        loss_weights=[1., lam_recon],
        metrics={'capsnet': 'accuracy'}
    )

    # Begin: Training with data augmentation
    def train_generator(x, y, tg_batch_size, tg_shift_fraction=0.):
        # shift up to 2 pixel for MNIST
        train_datagen = ImageDataGenerator(
            width_shift_range=tg_shift_fraction,
            height_shift_range=tg_shift_fraction
        )
        generator = train_datagen.flow(x, y, batch_size=tg_batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation.
    # If shift_fraction=0., also no augmentation.
    model.fit_generator(
        generator=train_generator(
            x_train,
            y_train,
            batch_size,
            shift_fraction
        ),
        steps_per_epoch=int(y_train.shape[0] / batch_size),
        epochs=epochs,
        validation_data=[[x_test, y_test], [y_test, x_test]],
        callbacks=[log, tb, checkpoint, lr_scheduler]
    )
    # End: Training with data augmentation

    model.save_weights(save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % save_dir)

    plot_log(save_dir + '/log.csv', show=True)

    return model


def test(model, data, save_dir):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print(
        'Test acc:',
        np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    )

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(save_dir + "/real_and_recon.png")
    # TODO: show images for misclassified data
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, digit, save_dir):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(save_dir + '/manipulate-%d.png' % digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (save_dir, digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)
