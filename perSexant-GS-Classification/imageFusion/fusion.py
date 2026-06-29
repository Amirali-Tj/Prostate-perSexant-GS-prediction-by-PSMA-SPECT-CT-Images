import tensorflow as tf
import keras

# implemet the IFCNN model in keras and load weights on the model

class ConvBlock(keras.Model) :
    def __init__(self) :
        super().__init__()
        self.Conv2D    = keras.layers.Conv2D(filters=64 , padding="same" , strides=1) # padding 3
        self.Relu      = keras.layers.ReLU()
        self.batchnorm = keras.layers.BatchNormalization()
    def call(self , tensor):
        x = self.Conv2D(tensor)
        x = self.batchnorm(x)
        x = self.Relu(x)
        
        return x
        

        