import faulthandler;

faulthandler.enable()
import numpy as np
import os, sys, random, time, math, argparse

from utils.setup_mnist import MNIST
from utils.setup_cifar import CIFAR
from utils.setup_gtsrb import GTSRB
import utils.save_nlayer_weights as nl
from utils.utils import generate_data

from algo_Semantic import Semantic


def handle_parser(parser):
    parser.add_argument('--model',
                        default="cifar",
                        choices=["cifar", "mnist", "gtsrb"],
                        help='model to be used')
    parser.add_argument('--eps',
                        default=0.05,
                        type=float,
                        help="epsilon for verification")
    parser.add_argument('--hidden',
                        default=2048,
                        type=int,
                        help="number of hidden neurons per layer")
    parser.add_argument('--delta',
                        default=0.001,
                        type=float,
                        help='size of divisions')
    parser.add_argument('--numlayer',
                        default=5,
                        type=int,
                        help='number of layers in the model')
    parser.add_argument('--numimage',
                        default=2,
                        type=int,
                        help='number of images to run')
    parser.add_argument('--startimage',
                        default=0,
                        type=int,
                        help='start image')
    parser.add_argument('--hsl',
                        default="lighten",
                        choices=["lighten", "saturate", "hue", "bandc"],
                        help='model to be used')
    parser.add_argument('--norm',
                        default="i",
                        type=str,
                        choices=["i", "1", "2"],
                        help='perturbation norm: "i": Linf, "1": L1, "2": L2')
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

    modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + str(nhidden) + suffix
    if not os.path.isfile(modelfile):
        # if not found, try models/mnist_3layer_relu_1024_1024
        modelfile += ("_" + str(nhidden)) * (args.numlayer - 2) + suffix
        # if still not found, try models/mnist_3layer_relu
        if not os.path.isfile(modelfile):
            modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + suffix
            # if still not found, try models/mnist_3layer_relu_1024_best
            if not os.path.isfile(modelfile):
                modelfile = "models/" + args.model + "_" + str(args.numlayer) + "layer_" + activation + "_" + str(
                    nhidden) + suffix + "_best"
                if not os.path.isfile(modelfile):
                    raise (RuntimeError("cannot find model file"))

    if args.LP or args.LPFULL:
        # use gurobi solver
        import gurobipy as grb

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    if args.model == "mnist":
        data = MNIST()
        model = nl.NLayerModel([nhidden] * (args.numlayer - 1), modelfile, activation=activation)
    elif args.model == "cifar":
        data = CIFAR()
        model = nl.NLayerModel([nhidden] * (args.numlayer - 1), modelfile, image_size=32, image_channel=3,
                               activation=activation)
    elif args.model == "gtsrb":
        data = GTSRB()
        model = nl.NLayerModel([nhidden] * (args.numlayer - 1), modelfile, image_size=28, image_channel=3,
                                activation=activation, num_labels = 43)
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

    # compute worst case bound: no need to pass in sess, model and data
    # just need to pass in the weights, true label, norm, x0, prediction of x0, number of layer and eps
    Semantic_BND = Semantic(model)

    total_verifiable = 0
    lower, upper = 0.0, 0.0
    delta = args.delta
    for i in range(len(inputs)):
        Nsamp += 1
        p = args.norm  # p = "1", "2", or "i"
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        start = time.time()
        eps = args.eps
        
        start_1 = time.time()
        # run CROWN
        robustness, max_cert = Semantic_BND.certify_eps_explicit(predict_label, target_label, eps, inputs[i],
                                                                    args.hsl, p, delta = delta)

        print("verified", time.time() - start_1, robustness)

        if robustness >= 0:
            total_verifiable += 1
            verifiable = True
        else:
            verifiable = False

        print("[L1] model = {}, seq = {}, id = {}, true_class = {}, target_class = {}, info = {}, "
              "verifiable = {}, lower_bound = {}, upper_bound = {}, time = {:.4f}, total_time = {:.4f}"
              .format(modelfile, i, true_ids[i], predict_label, target_label, img_info[i],
                      verifiable, max_cert[0], max_cert[1], time.time() - start, time.time() - start))

        lower += max_cert[0]
        upper += max_cert[1]
        sys.stdout.flush()
        sys.stderr.flush()

    print("[L0] model = {}, info = {}, numimage = {},  lower_bound_avg = {}, uper_bound_avg = {}, total verifiable = {:.2f}%, time = {:.4f}, total_time = {:.4f}".format(modelfile, img_info[i], Nsamp, lower/Nsamp, upper/Nsamp, 100 * total_verifiable / Nsamp, time.time() - start, time.time() - total_time_start))

    sys.stdout.flush()
    sys.stderr.flush()