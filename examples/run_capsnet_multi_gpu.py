import numpy as np
import tensorflow as tf
import os
import argparse
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model
from capsnet_keras import capsulenet_multi_gpu as capsulenet_gpu, capsulenet

if __name__ == "__main__":
    # set hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=300, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing data set")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--gpus', default=2, type=int)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = capsulenet.load_mnist()

    # define model
    with tf.device('/cpu:0'):
        model, eval_model, manipulate_model = capsulenet.build_caps_net(
            input_shape=x_train.shape[1:],
            n_class=len(np.unique(np.argmax(y_train, 1))),
            n_routings=args.routings
        )
    model.summary()
    plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        # define muti-gpu model
        multi_model = multi_gpu_model(model, gpus=args.gpus)
        capsulenet_gpu.train(
            model=multi_model,
            data=((x_train, y_train), (x_test, y_test)),
            args=args
        )
        model.save_weights(args.save_dir + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
        capsulenet.test(model=eval_model, data=(x_test, y_test), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights provided, using random initialized weights.')

        capsulenet.manipulate_latent(manipulate_model, (x_test, y_test), args)
        capsulenet.test(model=eval_model, data=(x_test, y_test), args=args)
