import torch
import tensorflow as tf
import torchvision.models as models
import numpy as np
from pprint import pprint
import collections

# extracting and converting original IFCNN .pth weights to custom weight dict for keras

# custom weight dict
weights = {
    "conv1" : {
        "weight" : None ,
        "bias"   : None
    } ,
    "conv2" : {
        "conv" : {
            "weight" : None ,
            "bias"   : None 
        } ,
        "bn" : {
            "weight" : None ,
            "bias"   : None ,
            "running_mean" : None ,
            "running_var"  : None 
        }
    } ,
    "conv3" : {
        "conv" : {
            "weight" : None ,
            "bias"   : None
        } ,
        "bn" : {
            "weight" : None ,
            "bias"   : None ,
            "running_mean" : None ,
            "running_var"  : None
        }
    } ,
    "conv4" : {
        "weight" : None ,
        "bias"   : None
    }
}

def reformatWeightForIFCNN(pth , channel_convert=True) :
    stat = torch.load(
        pth , 
        map_location=torch.device('cpu')
    )

    if channel_convert == True :
        chan_ix = -1
        for layer , value in stat.items() :
            if np.size(value.shape) == 4 :
                stat[layer] = np.transpose(
                        value ,
                        axes=(2 , 3 , 1 , 0)
                    )
    else :
        chan_ix = 1
    
    # define the fields

    weights["conv1"]["weight"] = np.array(stat["conv1.weight"])
    weights["conv1"]["bias"]   = np.zeros(shape=stat["conv1.weight"].shape[chan_ix] , dtype=np.float32)
    #---
    weights["conv2"]["conv"]["weight"] = np.array(stat["conv2.conv.weight"])
    weights["conv2"]["conv"]["bias"]   = np.zeros(shape=stat["conv2.conv.weight"].shape[chan_ix] , dtype=np.float32)
    weights["conv2"]["bn"]["weight"]   = np.array(stat["conv2.bn.weight"])
    weights["conv2"]["bn"]["bias"]     = np.array(stat["conv2.bn.bias"])
    weights["conv2"]["bn"]["running_mean"]     = np.array(stat["conv2.bn.running_mean"])
    weights["conv2"]["bn"]["running_variance"] = np.array(stat["conv2.bn.running_var"])
    #---
    weights["conv3"]["conv"]["weight"] = np.array(stat["conv3.conv.weight"])
    weights["conv3"]["conv"]["bias"]   = np.zeros(shape=stat["conv3.conv.weight"].shape[chan_ix] , dtype=np.float32)
    weights["conv3"]["bn"]["weight"]   = np.array(stat["conv3.bn.weight"])
    weights["conv3"]["bn"]["bias"]     = np.array(stat["conv3.bn.bias"])
    weights["conv3"]["bn"]["running_mean"]     = np.array(stat["conv3.bn.running_mean"])
    weights["conv3"]["bn"]["running_variance"] = np.array(stat["conv3.bn.running_var"])
    #---
    weights["conv4"]["weight"] = np.array(stat["conv4.weight"])
    weights["conv4"]["bias"]   = np.array(stat["conv4.bias"])



    return weights



