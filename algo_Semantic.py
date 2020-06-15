import numpy as np

from algo_CROWN import CROWN

from threat_models.threat_lighten import get_first_layers as lighten_layers
from threat_models.threat_saturate import get_first_layers as saturate_layers
from threat_models.threat_hue import get_first_layers as hue_layers
from threat_models.threat_bandc import get_first_layers as bandc_layers

from utils.get_epsilon import get_eps_2 as get_eps


class Semantic(CROWN):
    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def get_layer_bound_implicit(W, b, UB_prev, LB_prev, is_first, x0, eps):

        UB_new = np.empty_like(b)
        LB_new = np.empty_like(b)

        if is_first:  # first layer
            Ax0 = np.matmul(W, x0)

            for j in range(W.shape[0]):

                dualnorm_Aj = np.sum(np.abs(np.multiply(W[j], eps)), axis=1)

                UB_new[j] = np.max(Ax0[j] + dualnorm_Aj) + b[j]
                LB_new[j] = np.min(Ax0[j] - dualnorm_Aj) + b[j]

        else:  # 2nd layer or more
            UB_hat = self.ReLU(UB_prev)
            LB_hat = self.ReLU(LB_prev)
            W_abs = np.abs(W)

            # not sure why, but in numba, W_abs is float32 and 0.5*(UB_hat-LB_hat) is float64
            # while in numpy, W_abs and UB_hat are both float32
            B_sum = np.float32(0.5) * (UB_hat + LB_hat)
            B_diff = np.float32(0.5) * (UB_hat - LB_hat)

            term_1st = np.dot(W_abs, B_diff)
            term_2nd = np.dot(W, B_sum) + b

            # term_1st = np.dot(W_abs,np.float32(0.5)*(UB_hat-LB_hat))
            # term_2nd = np.dot(W_Nk,np.float32(0.5)*(UB_hat+LB_hat))+b_Nk

            UB_new = term_1st + term_2nd
            LB_new = -term_1st + term_2nd

        return UB_new, LB_new

    @staticmethod
    # @jit(nopython=True)
    def get_semantic_layer_bound_implicit(Ws, bs, UBs, LBs, neuron_state, nlayer, bounds_ul, x0, eps):

        constants_ub = np.copy(bs[-1])
        constants_lb = np.copy(bs[-1])

        UB_final = np.zeros_like(constants_ub)
        LB_final = np.zeros_like(constants_lb)

        A_UB = np.copy(Ws[nlayer - 1])
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

                UB_final[j] += np.dot(A_UB[j], l_ub)
                LB_final[j] += np.dot(A_LB[j], l_lb)
                # update the j-th row of A with diagonal matrice
                A_UB[j] = A_UB[j] * diags_ub
                # update A with diagonal matrice
                A_LB[j] = A_LB[j] * diags_lb

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

        dualnorm_Aj_ub = np.dot(np.abs(A_UB),eps)
        dualnorm_Aj_lb = np.dot(np.abs(A_LB),eps)

        for j in range(A_UB.shape[0]):

            UB_final[j] += np.max(Ax0_UB[j] + dualnorm_Aj_ub[j])
            LB_final[j] += np.min(Ax0_LB[j] - dualnorm_Aj_lb[j])


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

        return UB_final, LB_final, constant_gap, probnd, As, Bs

    def certify_eps_explicit(self, predict_class, target_class, eps, x0, hsl, p="i", activation="relu", delta = None):
        
        if hsl == "lighten":
            w_new, b_new = lighten_layers(x0, self.weights[0], self.biases[0])
        elif hsl == "saturate":
            w_new, b_new = saturate_layers(x0, self.weights[0], self.biases[0])
        elif hsl == "hue":
            w_new, b_new = hue_layers(x0, self.weights[0], self.biases[0])
        elif hsl == "bandc":
            w_new, b_new = bandc_layers(x0, self.weights[0], self.biases[0])
        else:
            raise ValueError

        eps_val = delta
        div = int(np.floor((eps / eps_val) + 0.00001))
        min_val = 100.0
        max_cert = (-eps, eps)

        for j in range(1, 2*div+1):
            offset = (2 * (j % 2) - 1) * np.floor(j / 2) * eps_val
            x0 = (offset + (eps_val/2.0))*np.ones((1,)).astype(np.float32)

            preReLU_UB = []
            preReLU_LB = []
            neuron_states = []
            eps = (eps_val/2.0) * np.ones_like(x0).astype(np.float32)

            if hsl == "hue":
                weights = w_new + self.weights[1:]
                biases = b_new + self.biases[1:]
            elif hsl == "bandc":
                if cont:
                    offset += 1
                    eps = np.array([(eps_val / 2.0), eps2]).astype(np.float32)
                    x0 = np.array([(offset + (eps_val / 2.0)), 0.0]).astype(np.float32).reshape((2,))
                else:
                    eps = np.array([eps2, (eps_val / 2.0)]).astype(np.float32)
                    x0 = np.array([(0.1, offset + (eps_val / 2.0))]).astype(np.float32).reshape((2,))

                weights = w_new + self.weights[1:]
                biases = b_new + self.biases[1:]
            else:
                if offset >= 0:
                    weights = w_new['pos'] + self.weights[1:]
                    biases = b_new['pos'] + self.biases[1:]
                else:
                    weights = w_new['neg'] + self.weights[1:]
                    biases = b_new['neg'] + self.biases[1:]

            numlayer = self.numlayer + 1

            bounds_ul = super().init_layer_bound_general(weights)
            print("before bounds_ul[1][0] = {}".format(bounds_ul[1][0]))

            if activation == "relu":
                get_bounds = super().get_relu_bounds
            elif activation == "tanh":
                get_bounds = super().get_tanh_bounds
            elif activation == "sigmoid":
                get_bounds = super().get_sigmoid_bounds
            elif activation == "arctan":
                get_bounds = super().get_arctan_bounds
            else:
                raise ValueError('activation %s is not supported!' % activation)

            for num in range(numlayer):

                W = weights[num]
                b = biases[num]

                if num == 0:
                    UB, LB = super().get_layer_bound(W, b, UB_prev=None, LB_prev=None, is_first=True, x0=x0, eps=eps, p=p)
                else:
                    if num == numlayer - 1:
                        W, b = super().g0_trick(W, b, predict_class, target_class)

                    Ws = weights[:num] + [W]
                    bs = biases[:num] + [b]

                    UB, LB, _, _, As, Bs = super().get_crown_layer_bound(tuple(Ws), tuple(bs),
                                                                       tuple([x0 - x0] + preReLU_UB),
                                                                       tuple([x0 - x0] + preReLU_LB),
                                                                       tuple(neuron_states),
                                                                       num + 1, tuple(bounds_ul[:num + 1]),
                                                                       x0, eps, p)

                if num < numlayer - 1:
                    preReLU_UB.append(UB)
                    preReLU_LB.append(LB)

                    neuron_states.append(super().assign_neuron_state(preReLU_UB[-1], preReLU_LB[-1]))
                    print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,",
                          sum(neuron_states[-1] == +1), "neurons always activated")

                    get_bounds(preReLU_UB[-1], preReLU_LB[-1], neuron_states[-1], bounds_ul[num + 1])

            print("epsilon = {:.5f}".format(np.amax(eps)))
            print("    {:.2f} < f_c - f_j < {:.2f}".format(LB[0], UB[0]))
            gap_gx = LB[0]
            min_val = min(min_val, gap_gx)

            if min_val < 0:
                prevs = [(2*((j-i)%2)-1)*np.floor((j-i)/2)*eps_val for i in range(1, 3) if (j-i) >= 0]
                if len(prevs) > 0:
                    max_cert = (min(prevs), max(prevs) + eps_val)
                else:
                    max_cert = (0.0, 0.0)
                break

        gap_gx = min_val

        return gap_gx, max_cert

    def certify_eps_implicit(self, predict_class, target_class, lower_lim, upper_lim, x0, hsl="rotate", divisions=1, activation="relu"):
        
        if hsl == "rotate":
            eps, offsets = get_eps(x0 + 0.5, lower_lim, upper_lim, div=divisions)
        else:
            raise ValueError
        
        x0 = x0.flatten().astype(np.float32)
        x = (x0.reshape((x0.shape[0], 1)) + offsets).astype(np.float32)

        eps = eps.astype(np.float32)

        # contains numlayer arrays, each corresponding to a pre-ReLU bound
        preReLU_UB = []
        preReLU_LB = []
        # save neuron states
        neuron_states = []

        # initialize diags: in fast-lin, it's called diags; in crown, it's called bounds_ul
        bounds_ul = super().init_layer_bound_general(self.weights)
        
        if activation == "relu":
            get_bounds = super().get_relu_bounds
        elif activation == "tanh":
            get_bounds = super().get_tanh_bounds
        elif activation == "sigmoid":
            get_bounds = super().get_sigmoid_bounds
        elif activation == "arctan":
            get_bounds = super().get_arctan_bounds
        else:
            raise ValueError('activation %s is not supported!' % activation)

        eps = eps.astype(np.float32)
        for num in range(self.numlayer):

            W = self.weights[num]
            b = self.biases[num]

            if num == 0:
                UB, LB = Semantic.get_layer_bound_implicit(W, b, UB_prev=None, LB_prev=None, is_first=True, x0=x, eps=eps.T)
            else:
                # need to modify last layer weights and bias with g0_trick
                if num == self.numlayer - 1:  # last layer
                    W, b = super().g0_trick(W, b, predict_class, target_class)

                Ws = self.weights[:num] + [W]
                bs = self.biases[:num] + [b]

                # get pre-ReLU bounds
                UB, LB, _, _, As, Bs = Semantic.get_semantic_layer_bound_implicit(tuple(Ws), tuple(bs),
                                                                   tuple([x0 - x0] + preReLU_UB),
                                                                   tuple([x0 - x0] + preReLU_LB),
                                                                   tuple(neuron_states),
                                                                   num + 1, tuple(bounds_ul[:num + 1]),
                                                                   x, eps)

            if num < self.numlayer - 1:
                # save those pre-ReLU bounds
                preReLU_UB.append(UB)
                preReLU_LB.append(LB)

                neuron_states.append(super().assign_neuron_state(preReLU_UB[-1], preReLU_LB[-1]))
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,",
                      sum(neuron_states[-1] == +1), "neurons always activated")

                get_bounds(preReLU_UB[-1], preReLU_LB[-1], neuron_states[-1], bounds_ul[num + 1])

        print("epsilon = {:.5f}".format(np.amax(eps)))
        print("    {:.2f} < f_c - f_j < {:.2f}".format(LB[0], UB[0]))
        gap_gx = LB[0]

        return gap_gx