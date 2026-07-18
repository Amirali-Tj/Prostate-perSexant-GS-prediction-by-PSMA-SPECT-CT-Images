import collections
import tensorflow as tf
import keras
from keras.layers import Conv2D , Conv3D , ReLU , BatchNormalization , ZeroPadding2D , Average , Add , Maximum , Concatenate
from _ifcnn_weight import weights as wFormat
from _ifcnn_weight import reformatWeightForIFCNN
from pprint import pprint

# implemet the IFCNN model in keras and load weights on the model

class ConvBlock(keras.layers.Layer) :
    def __init__(self , name) :
        super().__init__(name=name)
        self.Conv2D    = Conv2D(filters=64 , kernel_size=3 , padding="same" , strides=1)
        self.batchnorm = BatchNormalization()
        self.Relu      = ReLU()
    def call(self , tensor , training=None):
        x = self.Conv2D(tensor)
        x = self.batchnorm(x , training=training)
        x = self.Relu(x)
        
        return x
    
class imageFusionCNN :
    def __init__(self , fuseType="MEAN" , data_format="channel_last") :
        self.type   = fuseType
        self.format = data_format

        # model layers
        if self.format == "channel_last" :
            self.inpt_1      = keras.Input(shape=(224 , 224 , 3))
            self.inpt_2      = keras.Input(shape=(224 , 224 , 3))
        elif self.format == "channel_first" :
            self.inpt_1      = keras.Input(shape=(3 , 224 , 224))
            self.inpt_2      = keras.Input(shape=(3 , 224 , 224))
        self.ZeroPad2D = ZeroPadding2D(padding=((3 , 3) , (3 , 3)))
        self.Conv1     = Conv2D(filters=64, kernel_size=7 , strides=1 , name="conv1")
        self.Conv2     = ConvBlock(name="conv2")
        self.Conv3     = ConvBlock(name="conv3")
        self.Conv4     = Conv2D(filters=3, kernel_size=1 , name="conv4")
        self.mean      = Average() 
        self.sum       = Add()
        self.max       = Maximum()

    def build(self) :
        # image 1 feature extraction
        x1 = self.ZeroPad2D(self.inpt_1)
        x1 = self.Conv1(x1)
        x1 = self.Conv2(x1)

        # image 2 feature extraction
        x2 = self.ZeroPad2D(self.inpt_2)
        x2 = self.Conv1(x2)
        x2 = self.Conv2(x2)

        # feature Fusion
        if self.type == "MEAN" :
            fuseX = self.mean([x1 , x2])
        elif self.type == "SUM" :
            fuseX = self.sum([x1 , x2])
        elif self.type == "MAX" :
            fuseX = self.max([x1 , x2])

        # feature reconstruction
        fuseX = self.Conv3(fuseX)
        fuseX = self.Conv4(fuseX)
        
        model = keras.Model(
            inputs  = [self.inpt_1 , self.inpt_2] ,
            outputs = fuseX
        )
        self._model = model

    def load_weights(self , weights : dict) :
        try :
            for layer in self._model.layers :
                if layer.name == "conv1" :
                    layer.set_weights(
                        [
                            weights["conv1"]["weight"] ,
                            weights["conv1"]["bias"]
                        ]
                    )
                elif layer.name == "conv2" : 
                    layer.Conv2D.set_weights(
                        [
                            weights["conv2"]["conv"]["weight"] , 
                            weights["conv2"]["conv"]["bias"]
                        ]
                    )
                    layer.batchnorm.set_weights(
                        [
                            weights["conv2"]["bn"]["weight"] ,
                            weights["conv2"]["bn"]["bias"] ,
                            weights["conv2"]["bn"]["running_mean"] ,
                            weights["conv2"]["bn"]["running_variance"]

                        ]
                    )
                elif layer.name == "conv3" : 
                    layer.Conv2D.set_weights(
                        [
                            weights["conv3"]["conv"]["weight"] , 
                            weights["conv3"]["conv"]["bias"]
                        ]
                    )
                    layer.batchnorm.set_weights(
                        [
                            weights["conv3"]["bn"]["weight"] ,
                            weights["conv3"]["bn"]["bias"] ,
                            weights["conv3"]["bn"]["running_mean"] ,
                            weights["conv3"]["bn"]["running_variance"]
                        ]
                    )
                elif layer.name == "conv4" :
                    layer.set_weights(
                        [
                            weights["conv4"]["weight"] ,
                            weights["conv4"]["bias"]
                        ]
                    )
        except KeyError :
                print("Warning your weights are not in true format , they didn't SET")
                print("use following format : ")
                pprint(wFormat)
            
    
    def get_model(self) :
        return self._model

class fusionLayer(keras.layers.Layer) : # will be inject to model
    def __init__(self , name , kernel_size , data_format="channels_last"):
        super().__init__(name=name)
        self.Conv3D = Conv3D(filter=3 , kernel_size=kernel_size , padding="same" , data_format=data_format)
        self.Cocat  = Concatenate(axis=-1)
    def __call__(self , listofTensor , training=None) :
        x = self.Cocat(listofTensor)
        x = self.Conv3D(x)
        return x
    


#-----
#w = reformatWeightForIFCNN(pth="fusionWeights/IFCNN-SUM.pth")

#fuse = imageFusionCNN(fuseType="MAX" , data_format="channel_last")
#fuse.build()
#fuse.load_weights(w)
#model = fuse.get_model()
#model.summary()