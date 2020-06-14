# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
from numba import jit
import numpy as np

# interval bound propagation
class IBP:
    def __init__(self, model):
        
        # setup self.weights, self.biases, and self.numlayer
        self.get_weights_list(model)
    
    def get_weights_list(self,model):     
        weights = []
        bias = []
        
        U = model.U    
        for i, Ui in enumerate(U):
            # save hidden layer weights, layer by layer
            # middle layer weights: Ui
            [weight_Ui, bias_Ui] = Ui.get_weights()
            print("Hidden layer {} weight shape: {}".format(i, weight_Ui.shape))        
            weights.append(np.ascontiguousarray(np.transpose(weight_Ui)))
            bias.append(np.ascontiguousarray(np.transpose(bias_Ui)))
            print("Hidden layer {} bias shape: {}".format(i,bias_Ui.shape))
    
        # last layer weights: W
        [W, bias_W] = model.W.get_weights()    
        weights.append(np.ascontiguousarray(np.transpose(W)))
        bias.append(np.ascontiguousarray(np.transpose(bias_W)))
        print("Last layer weight shape: {}".format(W.shape))
        print("Last layer bias shape: {}".format(bias_W.shape))
        
        self.weights = weights
        self.biases = bias
        assert len(weights) == len(bias)
        self.numlayer = len(weights)
        
    
    @staticmethod    
    def ReLU(x): 
        return np.maximum(x, 0)
    
    def get_layer_bound(self,W,b,UB_prev,LB_prev,is_first,x0,eps,p):

        UB_new = np.empty_like(b)
        LB_new = np.empty_like(b)
    
        if is_first: # first layer
            Ax0 = np.dot(W,x0)

            for j in range(W.shape[0]):

                if p == "0":
                    Ws = np.split(W[j], eps.shape[0])
                    dualnorm_Aj_u = 0
                    dualnorm_Aj_l = 0
                    for w, e in zip(Ws, eps):
                        e = int(e)
                        dualnorm_Aj_u += -1*np.sum(np.partition(-1*w[w > 0].flatten(), e)[:e])
                        dualnorm_Aj_l += np.sum(np.partition(w[w < 0].flatten(), e)[:e])
                    UB_new[j] = Ax0[j] + dualnorm_Aj_u + b[j]
                    LB_new[j] = Ax0[j] + dualnorm_Aj_l + b[j]

                else:
                    if p == "i":  # p == "i", q = 1
                        dualnorm_Aj = np.sum(np.abs(np.multiply(eps, W[j])))
                    elif p == "1":  # p = 1, q = i
                        dualnorm_Aj = np.max(np.abs(np.multiply(eps, W[j])))
                    elif p == "2":  # p = 2, q = 2
                        dualnorm_Aj = np.linalg.norm(np.multiply(eps, W[j]))

                    UB_new[j] = Ax0[j]+dualnorm_Aj+b[j]
                    LB_new[j] = Ax0[j]-dualnorm_Aj+b[j]
    
        else: # 2nd layer or more
            UB_hat = self.ReLU(UB_prev)
            LB_hat = self.ReLU(LB_prev)
            W_abs = np.abs(W)
            #print("dtype W_abs = {}, UB_hat = {}, LB_hat = {}".format(W_abs.dtype,UB_hat.dtype,LB_hat.dtype))
            
            # not sure why, but in numba, W_abs is float32 and 0.5*(UB_hat-LB_hat) is float64 
            # while in numpy, W_abs and UB_hat are both float32
            B_sum = np.float32(0.5)*(UB_hat+LB_hat)
            B_diff = np.float32(0.5)*(UB_hat-LB_hat)
            
            term_1st = np.dot(W_abs,B_diff)
            term_2nd = np.dot(W,B_sum)+b
            
            #term_1st = np.dot(W_abs,np.float32(0.5)*(UB_hat-LB_hat))
            #term_2nd = np.dot(W_Nk,np.float32(0.5)*(UB_hat+LB_hat))+b_Nk
            
            UB_new = term_1st + term_2nd
            LB_new = -term_1st + term_2nd
    
        return UB_new, LB_new

    @staticmethod
    def g0_trick(W,b,predict_class,target_class):
        
        # untargetted
        if target_class == -1:
            ind = np.ones(len(W), bool)
            ind[predict_class] = False
            W_last = W[predict_class] - W[ind]
            b_last = b[predict_class] - b[ind] 
        else:
            W_last = np.expand_dims(W[predict_class] - W[target_class], axis=0)
            b_last = np.expand_dims(b[predict_class] - b[target_class], axis=0)                          
        return W_last,b_last

    def certify_eps(self, predict_class, target_class, eps, x0, p="i", activation="relu"):
        # predict_class, target_class = scalar num
        x0 = x0.flatten().astype(np.float32)
        if type(eps) is not np.ndarray:
            eps = eps*np.ones_like(x0)
        else:
            offset = (eps[0, :] + eps[1, :]) /2
            x0 = x0 + offset
            eps = eps[1, :] - offset

        UB_N0 = x0 + eps
        LB_N0 = x0 - eps
        
        UBs = []
        LBs = []
        UBs.append(UB_N0)
        LBs.append(LB_N0)
        neuron_states = []
        
        for num in range(self.numlayer):
            W = self.weights[num] 
            b = self.biases[num]
            
            # middle layer
            if num < self.numlayer-1:
                if num == 0:
                    UB, LB = self.get_layer_bound(W,b,UBs[num],LBs[num],True,x0,eps,p)
                else:
                    UB, LB = self.get_layer_bound(W,b,UBs[num],LBs[num],False,x0,eps,p)
                
                print("num = {}, UB = {}, LB = {}".format(num,UB[:3],LB[:3]))
                
                neuron_states.append(np.zeros(shape=b.shape, dtype=np.int8))
                # neurons never activated set to -1
                neuron_states[-1] -= UB == 0
                # neurons always activated set to +1
                neuron_states[-1] += LB > 0
                # other neurons could be activated or inactivated
                print("layer", num, sum(neuron_states[-1] == -1), "neurons never activated,", 
                                    sum(neuron_states[-1] == +1), "neurons always activated")
                        
                UBs.append(UB)
                LBs.append(LB)
            
            # last layer
            elif num == self.numlayer-1:     
                W_last, b_last = self.g0_trick(W,b,predict_class,target_class)
                UB, LB = self.get_layer_bound(W_last,b_last,UB,LB,False,x0,eps,p)
                print("num = {}, UB = {}, LB = {}".format(num,UB[:3],LB[:3]))
        
        # Print bounds results
        print("epsilon = {:.5f}".format(np.amax(eps)))
        print("    {:.2f} < f_c - f_j < {:.2f}".format(LB[0], UB[0]))
        gap_gx = LB[0]
        
        return gap_gx

    def certify_largest_eps(self, predict_class, target_class, eps, x0, p="i", activation="relu"):
       
        # initial find eps range
        eps_LB = None
        eps_UB = None
        while eps_LB == None or eps_UB == None:
            gap = self.certify_eps(predict_class, target_class, eps, x0, p=p, activation=activation)
            print("searching initial eps bounds: eps_LB = {}, eps_UB = {}, eps = {:.4f}".format(eps_LB, eps_UB, eps))

            if type(gap) == tuple: # due to return As and Bs in CROWN
                gap = gap[0]
            
            if gap > 0:
                # increase eps
                eps_LB = eps
                eps = eps*10
            else:
                # decrease eps
                eps_UB = eps
                eps = eps/10

       # now we have rough value on eps_LB and eps_UB
        assert eps_UB >= eps_LB, "eps_UB(%.3f) > eps_LB(%.3f)" % (eps_UB, eps_LB)
        
        step = 1
        steps_max = 15

        while (eps_UB-eps_LB)/(eps_UB+eps_LB) > 0.0001 and step < steps_max:

            if step > steps_max:
                print("reach maximum binary search step")
                break

            gap = self.certify_eps(predict_class, target_class, eps, x0, p=p, activation=activation)
            
            if type(gap) == tuple: # due to return As and Bs in CROWN
                gap = gap[0]

            print("[L2] step = {}: searching formal eps bounds: eps_LB = {:.4f}, eps_UB = {:.4f}, eps = {:.4f}".format(step,eps_LB, eps_UB, eps))
            
            if gap > 0:
                # increase eps
                eps_LB = eps
            else:
                # decrease eps
                eps_UB = eps
            
            eps = (eps_UB+eps_LB)/2
            step +=1
            
        return eps_LB





