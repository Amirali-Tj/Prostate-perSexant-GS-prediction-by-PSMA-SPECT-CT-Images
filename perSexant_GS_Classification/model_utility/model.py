import tensorflow as tf
import keras
import backbone as bbn

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



class BilinearCnnWithAttention() :
    def __init__(self, backbone="ResNet50") :
        self.input     = keras.Input(shape=(64 , 64 , 3)) 
        self.matmul    = keras.ops.matmul
        self.Attention = dotAttention(name="dotProduct" , type="SUM") # type mention as argument
        self.ConvertToFirst  = keras.layers.Permute(dims=(3 , 1 , 2)) 
        self.permute         = keras.layers.Permute(dims=(2 , 1))
        self.divide          = keras.ops.divide

        if backbone == "ResNet50" :
            self.backbone      = bbn.ResNet50
            outputShape        = self.backbone.layers[-1].output.shape
            self.featurElement = outputShape[1]*outputShape[2] # For 2D Network
            self.reshape  = keras.layers.Reshape(target_shape=(outputShape[3] , self.featurElement)) # For 2D Network
    
    def build(self) : 
        # forward
        x  = self.backbone(self.input)
        x  = self.ConvertToFirst(x)
        xR = self.reshape(x)
        xT = self.permute(xR)
        x  = self.matmul(xR , xT)
        x  = self.divide(x , self.featurElement) 
        x  = self.Attention(x)

        # build model
        model = keras.Model(
            inputs  = self.input ,
            outputs = x
        )

        return model
    

model   = BilinearCnnWithAttention()
myModel = model.build()
myModel.summary()

