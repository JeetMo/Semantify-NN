#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:16:14 2019

@author: smartlily
"""

from numba import jit, njit
import numpy as np

from algo_IBP import IBP
import utils.general_fit as fit
from threat_models.threat_lighten import get_first_layers as lighten_layers
from threat_models.threat_saturate import get_first_layers as saturate_layers
from threat_models.threat_hue import get_first_layers as hue_layers
from threat_models.threat_bandc import get_first_layers as bandc_layers

import matplotlib.pyplot as plt

class CROWN(IBP):
    def __init__(self, model):
        super().__init__(model)

    # cannot unify the bounds calculation functions due to limitations of numba
    @staticmethod
    @jit(nopython=True)
    def get_relu_bounds(UBs, LBs, neuron_states, bounds_ul):
        ub_pn = fit.relu_ub_pn
        lb_pn = fit.relu_lb_pn
        ub_p = fit.relu_ub_p
        lb_p = fit.relu_lb_p
        ub_n = fit.relu_ub_n
        lb_n = fit.relu_lb_n
        # step 1: get indices of three classes of neurons
        # index: 0->slope for ub, 1->intercept for ub, 
        #        2->slope for lb, 3->intercept for lb
        upper_k = bounds_ul[0]
        upper_b = bounds_ul[1]
        lower_k = bounds_ul[2]
        lower_b = bounds_ul[3]
        idx_p = np.nonzero(neuron_states == 1)[0]
        idx_n = np.nonzero(neuron_states == -1)[0]
        idx_pn = np.nonzero(neuron_states == 0)[0]
        upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
        lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
        upper_k[idx_p], upper_b[idx_p] = ub_p(UBs[idx_p], LBs[idx_p])
        lower_k[idx_p], lower_b[idx_p] = lb_p(UBs[idx_p], LBs[idx_p])
        upper_k[idx_n], upper_b[idx_n] = ub_n(UBs[idx_n], LBs[idx_n])
        lower_k[idx_n], lower_b[idx_n] = lb_n(UBs[idx_n], LBs[idx_n])
        return upper_k, upper_b, lower_k, lower_b

    @staticmethod
    @jit(nopython=True)
    def get_tanh_bounds(UBs, LBs, neuron_states, bounds_ul):
        ub_pn = lambda u, l: fit.general_ub_pn(u, l, fit.act_tanh, fit.act_tanh_d)
        lb_pn = lambda u, l: fit.general_lb_pn(u, l, fit.act_tanh, fit.act_tanh_d)
        ub_p = lambda u, l: fit.general_ub_p(u, l, fit.act_tanh, fit.act_tanh_d)
        lb_p = lambda u, l: fit.general_lb_p(u, l, fit.act_tanh, fit.act_tanh_d)
        ub_n = lambda u, l: fit.general_ub_n(u, l, fit.act_tanh, fit.act_tanh_d)
        lb_n = lambda u, l: fit.general_lb_n(u, l, fit.act_tanh, fit.act_tanh_d)
        # step 1: get indices of three classes of neurons
        upper_k = bounds_ul[0]
        upper_b = bounds_ul[1]
        lower_k = bounds_ul[2]
        lower_b = bounds_ul[3]
        idx_p = np.nonzero(neuron_states == 1)[0]
        idx_n = np.nonzero(neuron_states == -1)[0]
        idx_pn = np.nonzero(neuron_states == 0)[0]
        upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
        lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
        upper_k[idx_p], upper_b[idx_p] = ub_p(UBs[idx_p], LBs[idx_p])
        lower_k[idx_p], lower_b[idx_p] = lb_p(UBs[idx_p], LBs[idx_p])
        upper_k[idx_n], upper_b[idx_n] = ub_n(UBs[idx_n], LBs[idx_n])
        lower_k[idx_n], lower_b[idx_n] = lb_n(UBs[idx_n], LBs[idx_n])
        return upper_k, upper_b, lower_k, lower_b

    @staticmethod
    @jit(nopython=True)
    def get_sigmoid_bounds(UBs, LBs, neuron_states, bounds_ul):
        ub_pn = lambda u, l: fit.general_ub_pn(u, l, fit.act_sigmoid, fit.act_sigmoid_d)
        lb_pn = lambda u, l: fit.general_lb_pn(u, l, fit.act_sigmoid, fit.act_sigmoid_d)
        ub_p = lambda u, l: fit.general_ub_p(u, l, fit.act_sigmoid, fit.act_sigmoid_d)
        lb_p = lambda u, l: fit.general_lb_p(u, l, fit.act_sigmoid, fit.act_sigmoid_d)
        ub_n = lambda u, l: fit.general_ub_n(u, l, fit.act_sigmoid, fit.act_sigmoid_d)
        lb_n = lambda u, l: fit.general_lb_n(u, l, fit.act_sigmoid, fit.act_sigmoid_d)
        # step 1: get indices of three classes of neurons
        upper_k = bounds_ul[0]
        upper_b = bounds_ul[1]
        lower_k = bounds_ul[2]
        lower_b = bounds_ul[3]
        idx_p = np.nonzero(neuron_states == 1)[0]
        idx_n = np.nonzero(neuron_states == -1)[0]
        idx_pn = np.nonzero(neuron_states == 0)[0]
        upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
        lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
        upper_k[idx_p], upper_b[idx_p] = ub_p(UBs[idx_p], LBs[idx_p])
        lower_k[idx_p], lower_b[idx_p] = lb_p(UBs[idx_p], LBs[idx_p])
        upper_k[idx_n], upper_b[idx_n] = ub_n(UBs[idx_n], LBs[idx_n])
        lower_k[idx_n], lower_b[idx_n] = lb_n(UBs[idx_n], LBs[idx_n])
        return upper_k, upper_b, lower_k, lower_b

    @staticmethod
    @jit(nopython=True)
    def get_arctan_bounds(UBs, LBs, neuron_states, bounds_ul):
        ub_pn = lambda u, l: fit.general_ub_pn(u, l, fit.act_arctan, fit.act_arctan_d)
        lb_pn = lambda u, l: fit.general_lb_pn(u, l, fit.act_arctan, fit.act_arctan_d)
        ub_p = lambda u, l: fit.general_ub_p(u, l, fit.act_arctan, fit.act_arctan_d)
        lb_p = lambda u, l: fit.general_lb_p(u, l, fit.act_arctan, fit.act_arctan_d)
        ub_n = lambda u, l: fit.general_ub_n(u, l, fit.act_arctan, fit.act_arctan_d)
        lb_n = lambda u, l: fit.general_lb_n(u, l, fit.act_arctan, fit.act_arctan_d)
        # step 1: get indices of three classes of neurons
        upper_k = bounds_ul[0]
        upper_b = bounds_ul[1]
        lower_k = bounds_ul[2]
        lower_b = bounds_ul[3]
        idx_p = np.nonzero(neuron_states == 1)[0]
        idx_n = np.nonzero(neuron_states == -1)[0]
        idx_pn = np.nonzero(neuron_states == 0)[0]
        upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
        lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
        upper_k[idx_p], upper_b[idx_p] = ub_p(UBs[idx_p], LBs[idx_p])
        lower_k[idx_p], lower_b[idx_p] = lb_p(UBs[idx_p], LBs[idx_p])
        upper_k[idx_n], upper_b[idx_n] = ub_n(UBs[idx_n], LBs[idx_n])
        lower_k[idx_n], lower_b[idx_n] = lb_n(UBs[idx_n], LBs[idx_n])
        return upper_k, upper_b, lower_k, lower_b


    # initialize diags:
    @staticmethod
    def init_layer_bound(Ws):
        nlayer = len(Ws)
        # preallocate all A matrices
        diags = [None] * nlayer
        # diags[0] is an identity matrix
        diags[0] = np.ones(Ws[0].shape[1], dtype=np.float32)
        for i in range(1, nlayer):
            diags[i] = np.empty(Ws[i].shape[1], dtype=np.float32)
        return diags

    @staticmethod
    def init_layer_bound_general(Ws):
        nlayer = len(Ws)
        # preallocate all upper and lower bound slopes and intercepts
        bounds_ul = [None] * nlayer
        # first k is identity
        bounds_ul[0] = np.ones((4, Ws[0].shape[1]), dtype=np.float32)
        for i in range(1, nlayer):
            bounds_ul[i] = np.empty((4, Ws[i].shape[1]), dtype=np.float32)
        return bounds_ul

    @staticmethod
    def assign_neuron_state(UB, LB):
        assert UB.shape == LB.shape, "UB shape = {} != LB shape {}" % (UB.shape, LB.shape)
        states = np.zeros(shape=UB.shape, dtype=np.int8)

        ## if UB and LB are passed ReLU
        # states -= (UB == 0)
        # states += (LB > 0)

        ## for raw Pre-ReLU bounds
        # if UB <= 0, it's inactivated, and state = -1
        states -= (UB <= 0)
        # if LB > 0, it's activated, and state = +1
        states += (LB > 0)
        # else, uncertain neuron states = 0

        return states

    # CROWN's get_layer_bound
    @staticmethod
    @jit(nopython=True)
    def get_crown_layer_bound(Ws, bs, UBs, LBs, neuron_state, nlayer, bounds_ul, x0, eps, p, e=0):
        assert nlayer >= 2
        assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1) == len(bounds_ul)
        assert p == "i" or p == "1" or p == "2" or p == "0"

        if p == "i":
            q_np = 1
        elif p == "2":
            q_np = 2
        elif p == "1":
            q_np = np.inf
        elif p == "0":
            q_np = 0
        else:
            q_np = np.inf

        ## moved get_bounds to outside
        ##get_bounds(UBs[nlayer-1], LBs[nlayer-1], neuron_state[nlayer - 2], bounds_ul[nlayer-1])

        # step 3: update matrix A (merged into one loop)
        # step 4: adding all constants (merged into one loop)
        constants_ub = np.copy(bs[-1])  # the last bias
        constants_lb = np.copy(bs[-1])  # the last bias

        # step 5: bounding l_n term for each layer
        UB_final = np.zeros_like(constants_ub)
        LB_final = np.zeros_like(constants_lb)
        # first A is W_{nlayer} D_{nlayer}
        # A_UB = Ws[nlayer-1] * diags[nlayer-1]
        A_UB = np.copy(Ws[nlayer - 1])
        # A_LB = Ws[nlayer-1] * diags[nlayer-1]
        A_LB = np.copy(Ws[nlayer - 1])
        for i in range(nlayer - 1, 0, -1):
            # create intercepts array for this layer
            l_ub = np.empty_like(LBs[i])
            l_lb = np.empty_like(LBs[i])
            diags_ub = np.empty_like(bounds_ul[i][0, :])
            diags_lb = np.empty_like(bounds_ul[i][0, :])
            upper_k = bounds_ul[i][0]
            upper_b = bounds_ul[i][1]
            lower_k = bounds_ul[i][2]
            lower_b = bounds_ul[i][3]
            """
            if not np.isfinite(upper_k).all():
                print("upper_k nan detected", i)
                return UB_final, LB_final
            if not np.isfinite(upper_b).all():
                print("upper_b nan detected", i)
                return UB_final, LB_final
            if not np.isfinite(lower_k).all():
                print("lower_k nan detected", i)
                return UB_final, LB_final
            if not np.isfinite(lower_b).all():
                print("lower_b nan detected", i)
                print(lower_b)
                loc = np.argwhere(np.isinf(lower_b))
                print(loc)
                u = UBs[i][loc]
                l = LBs[i][loc]
                print(u, l)
                print(general_lb_p(u, l, act_tanh, act_tanh_d))
                print(general_ub_p(u, l, act_tanh, act_tanh_d))
                print(lower_b[loc])
                return UB_final, LB_final
            """
            # bound the term A[i] * l_[i], for each element
            for j in range(A_UB.shape[0]):
                # index for positive entries in A for upper bound
                idx_pos_ub = np.nonzero(A_UB[j] > 0)[0]
                # index for negative entries in A for upper bound
                idx_neg_ub = np.nonzero(A_UB[j] <= 0)[0]
                # index for positive entries in A for lower bound
                idx_pos_lb = np.nonzero(A_LB[j] > 0)[0]
                # index for negative entries in A for lower bound
                idx_neg_lb = np.nonzero(A_LB[j] <= 0)[0]
                # for upper bound, set the neurons with positive entries in A to upper bound
                diags_ub[idx_pos_ub] = upper_k[idx_pos_ub]
                l_ub[idx_pos_ub] = upper_b[idx_pos_ub]
                # for upper bound, set the neurons with negative entries in A to lower bound
                diags_ub[idx_neg_ub] = lower_k[idx_neg_ub]
                l_ub[idx_neg_ub] = lower_b[idx_neg_ub]
                # for lower bound, set the neurons with negative entries in A to upper bound
                diags_lb[idx_neg_lb] = upper_k[idx_neg_lb]
                l_lb[idx_neg_lb] = upper_b[idx_neg_lb]
                # for lower bound, set the neurons with positve entries in A to lower bound
                diags_lb[idx_pos_lb] = lower_k[idx_pos_lb]
                l_lb[idx_pos_lb] = lower_b[idx_pos_lb]
                """
                if not np.isfinite(A_UB[j]).all():
                    print("A_UB[j] nan detected", i, j)
                    return UB_final, LB_final
                if not np.isfinite(A_LB[j]).all():
                    print("A_LB[j] nan detected", i, j)
                    return UB_final, LB_final
                if not np.isfinite(l_ub).all():
                    print("l_ub nan detected", i, j)
                    return UB_final, LB_final
                if not np.isfinite(l_lb).all():
                    print("l_lb nan detected", i, j)
                    return UB_final, LB_final
                """
                # compute the relavent terms
                UB_final[j] += np.dot(A_UB[j], l_ub)
                LB_final[j] += np.dot(A_LB[j], l_lb)
                # update the j-th row of A with diagonal matrice
                A_UB[j] = A_UB[j] * diags_ub
                # update A with diagonal matrice
                A_LB[j] = A_LB[j] * diags_lb
                """
                if not np.isfinite(UB_final).all():
                    print("UB_final nan detected", i, j)
                    return UB_final, LB_final
                if not np.isfinite(LB_final).all():
                    print("LB_final nan detected", i, j)
                    return UB_final, LB_final
                """
            # constants of previous layers
            constants_ub += np.dot(A_UB, bs[i - 1])
            constants_lb += np.dot(A_LB, bs[i - 1])
            """
            if not np.isfinite(constants_ub).all():
                print("constants_ub nan detected", i, j)
                return UB_final, LB_final
            if not np.isfinite(constants_lb).all():
                print("constants_lb nan detected", i, j)
                return UB_final, LB_final
            """
            # compute A for next loop
            # diags matrices is multiplied above
            A_UB = np.dot(A_UB, Ws[i - 1])
            A_LB = np.dot(A_LB, Ws[i - 1])

        # after the loop is done we get A0
        UB_final += constants_ub
        LB_final += constants_lb

        # step 6: bounding A0 * x
        Ax0_UB = np.dot(A_UB, x0)
        Ax0_LB = np.dot(A_LB, x0)

        if q_np == 1:
            dualnorm_ub = np.dot(np.abs(A_UB), eps)
            dualnorm_lb = np.dot(np.abs(A_LB), eps)

            UB_final += Ax0_UB + dualnorm_ub
            LB_final += Ax0_LB - dualnorm_lb

        else:

            for j in range(A_UB.shape[0]):
                dualnorm_Aj_ub = np.linalg.norm(np.multiply(eps, A_UB[j]), q_np)
                dualnorm_Aj_lb = np.linalg.norm(np.multiply(eps, A_LB[j]), q_np)
                # else:
                #     Aus = np.split(A_LB[j], eps.shape[0])
                #     Als = np.split(A_LB[j], eps.shape[0])
                #     dualnorm_Aj_ub = 0
                #     dualnorm_Aj_lb = 0
                #     for al, au, e in zip(Aus, Als, eps):
                #         e = int(e)
                #         dualnorm_Aj_ub += -1 * np.sum(np.partition(-1 * au[au > 0].flatten(), e)[:e])
                #         dualnorm_Aj_lb += np.sum(np.partition(al[al < 0].flatten(), e)[:e])

                    # dualnorm_Aj_ub = -1*np.sum(np.partition(-1*A_UB[j][A_UB[j] > 0].flatten(), e)[:e])
                    # dualnorm_Aj_lb = -1*np.sum(np.partition(-1*A_LB[j][A_LB[j] > 0].flatten(), e)[:e])

                UB_final[j] += (Ax0_UB[j] + dualnorm_Aj_ub)
                LB_final[j] += (Ax0_LB[j] - dualnorm_Aj_lb)

        # constant_gap: to be revised after deadline
        constant_gap = 0
        # probnd: to be revised after deadline
        probnd = 0

        # final constants A_final_UB, A_final_LB, B_final_UB, B_final_LB
        A_final_UB = np.copy(A_UB)
        A_final_LB = np.copy(A_LB)
        B_final_UB = np.copy(UB_final)
        B_final_LB = np.copy(LB_final)

        # use tuples instead of list in order to use numba
        As = (A_final_UB, A_final_LB)
        Bs = (B_final_UB, B_final_LB)
        # append tuples
        # coefficients = coefficients + (A_final_UB,A_final_LB,B_final_UB,B_final_LB)

        return UB_final, LB_final, constant_gap, probnd, As, Bs

    # CROWN's certify_eps      
    def certify_eps(self, predict_class, target_class, eps, x0, p="i", activation="relu"):
        # predict_class, target_class = scalar num
        x0 = x0.flatten().astype(np.float32)
        eps = eps.flatten().astype(np.float32)
        # contains numlayer arrays, each corresponding to a pre-ReLU bound
        preReLU_UB = []
        preReLU_LB = []
        # save neuron states
        neuron_states = []

        # initialize diags: in fast-lin, it's called diags; in crown, it's called bounds_ul
        bounds_ul = CROWN.init_layer_bound_general(self.weights)

        if activation == "relu":
            get_bounds = CROWN.get_relu_bounds
        elif activation == "tanh":
            get_bounds = CROWN.get_tanh_bounds
        elif activation == "sigmoid":
            get_bounds = CROWN.get_sigmoid_bounds
        elif activation == "arctan":
            get_bounds = CROWN.get_arctan_bounds
        else:
            raise ValueError('activation %s is not supported!' % activation)

        for num in range(self.numlayer):

            W = self.weights[num]
            b = self.biases[num]

            ## compute pre-ReLU bounds for neurons 
            # 1st layer
            if num == 0:
                # get the raw bound from IBP
                # 1st layer IBP bounds only depend on input constraints |x-x0| <= eps and not UB_prev, LB_prev
                UB, LB = super().get_layer_bound(W, b, UB_prev=None, LB_prev=None, is_first=True, x0=x0, eps=eps, p=p)

            else:
                # need to modify last layer weights and bias with g0_trick
                if num == self.numlayer - 1:  # last layer
                    # with g0_trick, use W = W[c]-W[j] (dim = l# of last hidden layer nodes *1) and b = b[c]-b[j] (dim = 1)
                    W, b = super().g0_trick(W, b, predict_class, target_class)

                Ws = self.weights[:num] + [W]
                bs = self.biases[:num] + [b]

                # get pre-ReLU bounds
                # note: [x0-x0] this term is just for extending the 1st dim and for numba to work (because we can't pass None type to numpy)
                UB, LB, _, _, As, Bs = CROWN.get_crown_layer_bound(tuple(Ws), tuple(bs),
                                                                   tuple([x0 - x0] + preReLU_UB),
                                                                   tuple([x0 - x0] + preReLU_LB),
                                                                   tuple(neuron_states),
                                                                   num + 1, tuple(bounds_ul[:num + 1]),
                                                                   x0, eps, p, e=int(np.amax(eps)))

            ## compute neuron states based on pre-ReLU bounds
            if num < self.numlayer - 1:
                # save those pre-ReLU bounds
                preReLU_UB.append(UB)
                preReLU_LB.append(LB)

                # computing neuron states using preReLU bounds
                neuron_states.append(CROWN.assign_neuron_state(preReLU_UB[-1], preReLU_LB[-1]))
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,",
                      sum(neuron_states[-1] == +1), "neurons always activated")

                # update next layer's bounds_ul[num+1] with current neuron states and preReLU bounds: num = 0 to numlayer-2 
                # note: this function will modify bounds_ul[num] due to python's pass by reference
                # compute slopes and intercepts for upper and lower bounds
                # only need to create upper/lower bounds' slope and intercept for this layer,
                # slopes and intercepts for previous layers have been stored
                # index: 0->slope for ub, 1->intercept for ub, 
                #        2->slope for lb, 3->intercept for lb
                get_bounds(preReLU_UB[-1], preReLU_LB[-1], neuron_states[-1], bounds_ul[num + 1])

                # Print bounds results
        ##print("diags = {}".format(diags))
        print("epsilon = {:.5f}".format(np.amax(eps)))
        print("    {:.2f} < f_c - f_j < {:.2f}".format(LB[0], UB[0]))
        gap_gx = LB[0]
        # print("[L1] inside CROWN, As length = {}, Bs length = {}".format(len(As), len(Bs)))

        return gap_gx, As, Bs
