"""
Keras implementation of CapsNet in Hinton's paper:
    Dynamic Routing Between Capsules.

The current version maybe only works for TensorFlow backend. Actually it
will be straightforward to re-write to TF code. Adopting to other backends
should be easy, but I have not tested this.

Usage:
       python capsulenet-multi-gpu.py
       python capsulenet-multi-gpu.py --gpus 2
       ... ...

Result:
    About 55 seconds per epoch on two GTX1080Ti GPU cards

Author: Xifeng Guo
E-mail: `guoxifeng1990@163.com`
Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

from keras import optimizers
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from capsnet_keras.utils import plot_log

k.set_image_data_format('channels_last')

from capsnet_keras.capsulenet import margin_loss


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data,
                 like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(
        log_dir=args.save_dir + '/tensorboard-logs',
        batch_size=args.batch_size,
        histogram_freq=int(args.debug)
    )
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.lr * (0.9 ** epoch)
    )

    # compile the model
    model.compile(
        optimizer=optimizers.Adam(lr=args.lr),
        loss=[margin_loss, 'mse'],
        loss_weights=[1., args.lam_recon]
    )

    # Begin: Training with data augmentation
    def train_generator(x, y, batch_size, shift_fraction=0.):
        # shift up to 2 pixel for MNIST
        train_datagen = ImageDataGenerator(
            width_shift_range=shift_fraction,
            height_shift_range=shift_fraction
        )
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation.
    # If shift_fraction=0., also no augmentation.
    model.fit_generator(
        generator=train_generator(
            x_train,
            y_train,
            args.batch_size,
            args.shift_fraction
        ),
        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
        epochs=args.epochs,
        validation_data=[[x_test, y_test], [y_test, x_test]],
        callbacks=[log, tb, lr_decay]
    )
    # End: Training with data augmentation

    plot_log(args.save_dir + '/log.csv', show=True)

    return model
