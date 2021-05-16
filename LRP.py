import numpy as np
import pandas as pd
from numpy import newaxis as na
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,Activation,Input
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

def linear_LRP(input_arr,w,b,preactivation,R_upper,eps=0.01):
    """
    N = サンプル数（バッチ数）
    M = この層のニューロン数
    D = 下の層のニューロン数
    input.shape = (N,D)
    w.shape = (D,M)
    b.shape = (M,)
    R_upper.shape = (N,M)
    pre_activation = np.dot(input,w) + b    # shape(N,M)
    """
    
    sign_preact = np.where(preactivation>=0,1,-1)  #shape(N,M)
    
    numer = w*input_arr[:,:,np.newaxis]      # shape(N,D,M)
    denom    = preactivation[:,np.newaxis,:] + (eps*sign_preact[:,np.newaxis,:])   # shape (N,1, M)
    
    message = (numer/denom)*R_upper[:,np.newaxis,:]
    
    R_lower = np.sum(message,axis=2)
    
    return R_lower  # shape(N,D)


def RNN_LRP(input_arr,w,u,b,R_upper,output):
    '''
    input.shape = (N,T,dim)
    output.shape = (N,T,units)
    '''
    
    R_in = np.zeros(input_arr.shape)
    for i in np.arange(1,input_arr.shape[1])[::-1]: #[3,2,1]
        preactivation = np.dot(input_arr[:,i,:],w)+np.dot(output[:,i-1,:],u)+b
        
        R_in[:,i,:] = linear_LRP(input_arr[:,i,:],w,b,preactivation,R_upper)
       # from IPython.core.debugger import Pdb; Pdb().set_trace()
        R_upper = linear_LRP(output[:,i-1,:],u,b,preactivation,R_upper)

    preactivation = np.dot(input_arr[:,0,:],w)+b
    R_in[:,0,:] = linear_LRP(input_arr[:,0,:],w,b,preactivation,R_upper)
    
    return R_in


def relevance_score(model,x_test):
    layers = [l for l in model.layers if isinstance(l,(Dense,SimpleRNN))]

    layers_name = [l.name for l in layers]
    layers_weights = [l.get_weights() for l in layers]

    outputs = [model.get_layer(l_name).output for l_name in layers_name]
    inputs = [model.get_layer(l_name).input for l_name in layers_name]

    get_output_model = Model(inputs=model.input,outputs=outputs)
    get_input_model = Model(inputs=model.input,outputs=inputs)
                                 
    layer_outputs = get_output_model.predict(x_test)  #!!!! X_test
    layer_inputs = get_input_model.predict(x_test)
    R_upper = layer_outputs[-1]

    for layer,i in zip(layers[::-1],np.arange(0,len(layers))[::-1]):
        if isinstance(layer,Dense):
            input_arr = layer_inputs[i]
            w = layers_weights[i][0]
            b = layers_weights[i][1]
            preactivation = np.dot(input_arr,w)+b
            R_lower = linear_LRP(input_arr,w,b,preactivation,R_upper)
            R_upper = R_lower
            
            continue

        if isinstance(layer,SimpleRNN):
            input_arr = layer_inputs[i]    
            w = layers_weights[i][0]
            u = layers_weights[i][1]
            b = layers_weights[i][2]
            outputs = layer_outputs[i]
            R_in = RNN_LRP(input_arr,w,u,b,R_upper,outputs)

    return R_in