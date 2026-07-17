import tensorflow as tf
import keras

class dotAttention(keras.layers.Layer) : # input is same on 3D volume
    def __init__(self , name , type=None):
        super().__init__(name=name)
        self.type = type
        self.matmul   = keras.ops.matmul
        self.softmax  = keras.layers.Softmax(axis=-1)
        self.multiply = keras.layers.Multiply()
        self.sum      = keras.ops.sum 
        self.mean     = keras.ops.mean 
     
    def build(self, input_shape):
        self.attWeight = self.add_weight(
            shape=(input_shape[1] , 1) ,
            initializer='glorot_uniform' ,
            trainable=True ,
            name="dotAtt"
        )
    def call(self, tesnor , training=None) :
        attenstionScore     = self.matmul(tesnor , self.attWeight)
        attenstionScoreNorm = self.softmax(attenstionScore)
        x = self.multiply([tesnor , attenstionScoreNorm])

        if self.type == "SUM" :
            x = self.sum(x , axis=1 , keepdims=False)
            return x
        elif self.type == "MEAN" :
            x = self.mean(x , axis=1 , keepdims=False)
            return x
        else :
            return x

class BilinearCNN : 
    pass