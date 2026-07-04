import collections
import tensorflow as tf
import keras
from keras.layers import Conv2D , ReLU , BatchNormalization , ZeroPadding2D , Average , Add , Maximum
from _ifcnn_weight import w
from pprint import pprint

# implemet the IFCNN model in keras and load weights on the model

class ConvBlock(keras.layers.Layer) :
    def __init__(self , name) :
        super().__init__(name=name)
        self.Conv2D    = Conv2D(filters=64 , kernel_size=3 , padding="same" , strides=1)
        self.Relu      = ReLU()
        self.batchnorm = BatchNormalization()
    def call(self , tensor):
        x = self.Conv2D(tensor)
        x = self.batchnorm(x)
        x = self.Relu(x)
        
        return x
    
class imageFusionCNN :
    def __init__(self , fuseType="MEAN" , data_format="channel_last") :
        self.type   = fuseType
        self.format = data_format
    
    def build(self) :
        # model layers
        if self.format == "channel_last" :
            inpt_1      = keras.Input(shape=(224 , 224 , 3))
            inpt_2      = keras.Input(shape=(224 , 224 , 3))
        elif self.format == "channel_first" :
            inpt_1      = keras.Input(shape=(3 , 224 , 224))
            inpt_2      = keras.Input(shape=(3 , 224 , 224))
        ZeroPad2D = ZeroPadding2D(padding=((3 , 3) , (3 , 3)))
        Conv1     = Conv2D(filters=64, kernel_size=7 , strides=1 , name="conv1")
        Conv2     = ConvBlock(name="conv2")
        Conv3     = ConvBlock(name="conv3")
        Conv4     = Conv2D(filters=3, kernel_size=1 , name="conv4")
        mean      = Average() 
        sum       = Add()
        max       = Maximum()

        def forward() :
            # image 1 feature extraction
            x1 = ZeroPad2D(inpt_1)
            x1 = Conv1(x1)
            x1 = Conv2(x1)

            # image 2 feature extraction
            x2 = ZeroPad2D(inpt_2)
            x2 = Conv1(x2)
            x2 = Conv2(x2)

            # feature Fusion
            if self.type == "MEAN" :
                fuseX = mean([x1 , x2])
            elif self.type == "SUM" :
                fuseX = sum([x1 , x2])
            elif self.type == "MAX" :
                fuseX = max([x1 , x2])

            # feature reconstruction
            fuseX = Conv3(fuseX)
            fuseX = Conv4(fuseX)
            return fuseX
        
        model = keras.Model(
            inputs  = [inpt_1 , inpt_2] ,
            outputs = forward()
        )
        self._model = model

    def load_weights(self , weights : dict) :
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
            
    
    def get_model(self) :
        return self._model

#-----
#fuse = imageFusionCNN(fuseType="SUM")
#fuse.build()
#fuse.load_weights(w)

#model = fuse.get_model()
#model.summary()