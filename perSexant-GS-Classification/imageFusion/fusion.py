import tensorflow as tf
import keras
from keras.layers import Conv2D , ReLU , BatchNormalization , ZeroPadding2D , Average
#from _ifcnn_weight import channel_last_weights

# implemet the IFCNN model in keras and load weights on the model

class ConvBlock(keras.layers.Layer) :
    def __init__(self) :
        super().__init__()
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
        Conv2     = ConvBlock()
        Conv3     = ConvBlock()
        Conv4     = Conv2D(filters=3, kernel_size=1 , name="conv4")
        average   = Average() 

        def forward() :
            # image 1
            x1 = ZeroPad2D(inpt_1)
            x1 = Conv1(x1)
            x1 = Conv2(x1)

            # image 2
            x2 = ZeroPad2D(inpt_2)
            x2 = Conv1(x2)
            x2 = Conv2(x2)

            # merge 
            mrgX = average([x1 , x2])
            mrgX = Conv3(mrgX)
            mrgX = Conv4(mrgX)
            return mrgX
        
        model = keras.Model(
            inputs  = [inpt_1 , inpt_2] ,
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
model = fuse.get_model()
model.summary()