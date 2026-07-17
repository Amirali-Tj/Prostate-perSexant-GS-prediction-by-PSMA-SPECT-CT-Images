import tensorflow as tf



class multiWindowStacking :
    def __init__(self  , *ranges):
        self.ranges = tf.convert_to_tensor(list(ranges) , dtype=tf.float64)
    def WindowStacking(self , img_arr , label_arr) : # graph compatibale
        def cond(rng , i , img_arr , stackedWindows) :
            return i < tf.shape(rng)[0]
        def body(rng , i , img_arr , stackedWindows) :
            wl = rng[i][0]
            ww = rng[i][1]
            minHU = wl - (ww/2)
            maxHU = wl + (ww/2)
            windowChannel  = tf.expand_dims(tf.clip_by_value(img_arr , minHU , maxHU) , axis=0)
            stackedWindows = tf.concat([stackedWindows , windowChannel] , axis=0)
            i+=1
            return rng , i , img_arr , stackedWindows
        _ , _ , _ , windows = tf.while_loop(
            cond ,
            body ,
            loop_vars=[
                self.ranges ,
                0 , 
                img_arr ,
                tf.expand_dims(tf.zeros(tf.shape(img_arr) , dtype=tf.float64) , axis=0)
            ] , 
            shape_invariants = [
                self.ranges.get_shape(),
                tf.TensorShape(()) ,
                img_arr.get_shape() , 
                tf.TensorShape([None , None , None , None])
            ]
        )
        #---
        img_arr   = windows[1: , : , : , :]
        label_arr = tf.tile(tf.expand_dims(label_arr , axis=0) , [tf.shape(self.ranges)[0] , 1 , 1 , 1])
        #---
        return img_arr , label_arr


class SpectWindowing : # add specific spect windows
    pass