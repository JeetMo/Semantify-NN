import faulthandler;

faulthandler.enable()
import numpy as np
# import tensorflow as tf
import os, sys, random, time, math, argparse

from utils.setup_mnist import MNIST
from utils.setup_cifar import CIFAR
from utils.setup_gtsrb import GTSRB
import utils.save_nlayer_weights as nl
from utils.utils import generate_data

from utils.attacks import grid_attack


def handle_parser(parser):
    parser.add_argument('--model',
                        default="mnist",
                        choices=["cifar", "mnist", "gtsrb"],
                        help='model to be used')
    parser.add_argument('--eps',
                        default=0.5,
                        type=float,
                        help="epsilon for verification")
    parser.add_argument('--delta',
                        default=0.05,
                        type=float,
                        help="step size for grid")
    parser.add_argument('--hidden',
                        default=1024,
                        type=int,
                        help="number of hidden units")
    parser.add_argument('--numlayer',
                        default=2,
                        type=int,
                        help='number of layers in the model')
    parser.add_argument('--numimage',
                        default=2,
                        type=int,
                        help='number of images to run')
    parser.add_argument('--modelfile',
                        default=None,
                        type=str,
                        help='pretrained model name')
    parser.add_argument('--startimage',
                        default=0,
                        type=int,
                        help='start image')
    parser.add_argument('--attack',
                        default="lighten",
                        choices=["lighten", "saturate", "hue", "bandc", "rotate"],
                        help='threat model to be used')
    parser.add_argument('--LP',
                        action="store_true",
                        help='use LP to get bounds for final output')
    parser.add_argument('--LPFULL',
                        action="store_true",
                        help='use FULL LP to get bounds for output')
    parser.add_argument('--quad',
                        action="store_true",
                        help='use quadratic bound to imporve 2nd layer output')
    parser.add_argument('--warmup',
                        action="store_true",
                        help='warm up before the first iteration')
    parser.add_argument('--modeltype',
                        default="vanilla",
                        choices=["lighten", "saturate", "hue", "vanilla", "dropout", "distill", "adv_retrain"],
                        help="select model type")
    parser.add_argument('--targettype',
                        default="top2",
                        choices=["untargeted", "least", "top2", "random"],
                        help='untargeted minimum distortion')
    parser.add_argument('--steps',
                        default=15,
                        type=int,
                        help='how many steps to binary search')
    parser.add_argument('--activation',
                        default="relu",
                        choices=["relu", "tanh", "sigmoid", "arctan", "elu", "hard_sigmoid", "softplus"])
    parser.add_argument('--test_minUB',
                        action="store_true",
                        help='test the idea of minimize UB of g(x) in Fast-Lin')
    parser.add_argument('--test_estLocalLips',
                        action="store_true",
                        help='test the idea of estimating local lipschitz constant using Fast-Lin')
    parser.add_argument('--test_probnd',
                        default="none",
                        choices=["gaussian_iid", "gaussian_corr", "uniform", "none"],
                        help="select input distribution")
    parser.add_argument('--test_weightpert',
                        action="store_true",
                        help="perturb weight matrices")
    return parser


import matplotlib.pyplot as plt


def gen_image(arr):
    two_d = arr.astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt


if __name__ == "__main__":

    #### parser ####
    parser = argparse.ArgumentParser(description='compute activation bound for CIFAR and MNIST')
    parser = handle_parser(parser)
    args = parser.parse_args()

    nhidden = args.hidden
    # quadratic bound only works for ReLU
    assert ((not args.quad) or args.activation == "relu")
    # for all activations we can use general framework

    targeted = True
    if args.targettype == "least":
        target_type = 0b0100
    elif args.targettype == "top2":
        target_type = 0b0001
    elif args.targettype == "random":
        target_type = 0b0010
    elif args.targettype == "untargeted":
        target_type = 0b10000
        targeted = False

    if args.modeltype == "vanilla":
        suffix = ""
    else:
        suffix = "_" + args.modeltype

    # try models/mnist_3layer_relu_1024
    activation = args.activation

    if args.modelfile is None:
        modelfile = "models_training/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + str(nhidden) + suffix
        if not os.path.isfile(modelfile):
            # if not found, try models/mnist_3layer_relu_1024_1024
            modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + str(nhidden) + suffix
            # if still not found, try models/mnist_3layer_relu
            if not os.path.isfile(modelfile):
                modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + suffix
                # if still not found, try models/mnist_3layer_relu_1024_best
                if not os.path.isfile(modelfile):
                    modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + str(
                        nhidden) + suffix + "_best"
                    if not os.path.isfile(modelfile):
                        raise (RuntimeError("cannot find model file"))
    else:
        modelfile = args.modelfile


    if args.model == "mnist":
        data = MNIST()
        model = nl.NLayerModel([nhidden] * (args.numlayer - 1), modelfile, activation=activation)
    elif args.model == "cifar":
        data = CIFAR()
        model = nl.NLayerModel([nhidden] * (args.numlayer - 1), modelfile, image_size=32, image_channel=3,activation=activation)
    elif args.model == "gtsrb":
        data = GTSRB()
        model = nl.NLayerModel([nhidden] * (args.numlayer - 1), modelfile, image_size=28, image_channel=3, num_labels=43, activation=activation)
    else:
        raise (RuntimeError("unknown model: " + args.model))

    print("Evaluating", modelfile)
    sys.stdout.flush()

    random.seed(1215)
    np.random.seed(1215)

    """
    Generate data
    """
    inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=args.numimage, targeted=targeted,
                                                                     random_and_least_likely=True,
                                                                     target_type=target_type,
                                                                     predictor=model.model.predict,
                                                                     start=args.startimage)
    # get the logit layer predictions
    preds = model.model.predict(inputs)

    Nsamp = 0
    r_sum = 0.0
    r_gx_sum = 0.0

    """
    Start computing robustness bound
    """
    print("starting robustness verification on {} images!".format(len(inputs)))
    sys.stdout.flush()
    sys.stderr.flush()
    total_time_start = time.time()
    total_verifiable = 0

    # compute worst case bound: no need to pass in sess, model and data
    # just need to pass in the weights, true label, norm, x0, prediction of x0, number of layer and eps
    avg = 0.0
    for i in range(len(inputs)):
        verifiable = False
        Nsamp += 1
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        start = time.time()
        n = 2*args.eps/args.delta
        eps = args.eps

        images = grid_attack(inputs[i], -args.eps, args.eps, args.delta, method=args.attack)
        print("images shape = {}".format(images.shape))
        print("inputs[i] shape = {}".format(inputs[i].shape))
        print("labels shape = {}".format(data.train_labels.shape))
        time.sleep(3)
        predictions = model.model.predict(images[1:])
        indices = np.where(predictions[:, target_label] > predictions[:, predict_label])[0]

        if len(indices) == 0:
            verifiable = True
            value = args.eps
        else:
            value = np.min(np.abs(indices - (n/2)))*args.delta
        print(i, value)
        avg += value
        print("[L1] Test 2: eps = {:.3f}, runtime = {:.2f}".format(
                value, time.time() - start))

        if verifiable:
            total_verifiable += 1

        sys.stdout.flush()
        sys.stderr.flush()
    print(
        "[L2] Test 2: verification percentage = {:.5f}, avg_eps = {:.3f}, runtime = {:.2f}".format(
            100 * total_verifiable / Nsamp,
            avg / Nsamp, time.time() - total_time_start))

    sys.stdout.flush()
    sys.stderr.flush()
