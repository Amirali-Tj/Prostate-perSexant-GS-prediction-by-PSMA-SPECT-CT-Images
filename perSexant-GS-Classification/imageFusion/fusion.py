import tensorflow as tf
import keras
from keras.layers import Conv2D , ReLU , BatchNormalization , ZeroPadding2D
from _ifcnn_weight import channel_last_weights

# implemet the IFCNN model in keras and load weights on the model

class ConvBlock(keras.layers.Layer) :
    def __init__(self) :
        super().__init__()
        self.Conv2D    = Conv2D(filters=64 , kernel_size=7 , padding="same" , strides=1) # padding 3
        self.Relu      = ReLU()
        self.batchnorm = BatchNormalization()
    def call(self , tensor):
        x = self.Conv2D(tensor)
        x = self.batchnorm(x)
        x = self.Relu(x)
        
        return x
    
class imageFusionCNN :
    def __init__(self , fuseType=None , data_format="channel_last") :
        self.type   = fuseType
        self.format = data_format
    
    def build(self) :
        # model layers
        if self.format == "channel_last" :
            inpt      = keras.Input(shape=(224 , 224 , 3))
        elif self.format == "channel_first" :
            inpt      = keras.Input(shape=(3 , 224 , 224))
        
        ZeroPad2D = ZeroPadding2D(padding=((3 , 3) , (3 , 3)))
        Conv1     = Conv2D(filters=64, kernel_size=7 , strides=1 , name="conv1")

        def forward() :
            x = ZeroPad2D(inpt)
            x = Conv1(x)
            return x
        
        model = keras.Model(
            inputs  = inpt ,
            outputs = forward()
        )
        self._model = model

    def load_weights(self , weights) :
        for layer in self._model.layers :
            if layer.name in list(weights.keys()) :
                layer.set_weights(weights[layer.name])
    
    def get_model(self) :
        return self._model


fuse = imageFusionCNN()
fuse.build()
fuse.load_weights(channel_last_weights)
model = fuse.get_model()
model.summary()