import torch
import tensorflow as tf
import torchvision.models as models
import numpy as np
from pprint import pprint
import collections

class kerasWeightsConstruct :
    def __init__(self , statDict : collections.OrderedDict):
        self.stat = statDict
        self._sampleLayerNames()

    def kerasWeightConstructor(self) :
        for layer , value in self.stat.items() :
            if np.size(value.shape) == 4 :
                self.stat[layer] = np.transpose(
                    value ,
                    axes=(2 , 3 , 1 , 0)
                )
        return self.stat

    def _sampleLayerNames(self) :
        names = [
            "conv1_weight" , 
            "conv2_weight" ,
            "conv3_weight" ,
            "conv4_weight" ,
            "conv4_bias" ,
            "conv2_bn_weight" ,
            "conv2_bn_bias" ,
            "conv2_bn_running_mean" ,
            "conv2_bn_running_var" ,
            "conv3_bn_weight" ,
            "conv3_bn_bias" ,
            "conv3_bn_running_mean" ,
            "conv3_bn_running_var"
        ]
        print("example dict layer names : ")
        pprint(names)   




#--- run test  
model = torch.load(
    "fusionWeights/IFCNN-MAX.pth" , 
    map_location=torch.device('cpu')
)

convert = kerasWeightsConstruct(model)
weights = convert.kerasWeightConstructor()
for layer , value in weights.items() :
    print(layer , " :" , value.shape)

