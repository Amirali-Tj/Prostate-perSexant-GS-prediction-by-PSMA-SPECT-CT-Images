import nrrd
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy import ndimage

def read_img(imgPath , labelTensor) :
    pass

def nrrd2Nifti(nrrdFile : str , output: str) :
    data , header = nrrd.read(nrrdFile)
    space = header["space"]
    directions = header["space directions"]
    origin = np.absolute(header["space origin"]) 

    if "left" in space :
        dir_coefi = -1
    elif "right" in space : 
        dir_coefi = 1
    if "posterior" in space :
        dir_coefj = -1
    elif "anterior" in space : 
        dir_coefj = 1
    if "inferior" in space :
        dir_coefk = -1
    elif "superior" in space : 
        dir_coefk = 1
    
    ijk_coeff   = np.array([dir_coefi , dir_coefj , dir_coefk])
    orgin_coeff = np.array([-dir_coefi , -dir_coefj , -dir_coefk])
    niftiDir    = directions*ijk_coeff
    niftiOrigin = np.expand_dims(origin*orgin_coeff , axis=1)
    niftiAffine = np.concatenate(
        [
            niftiDir , 
            niftiOrigin
        ] , 
        axis=1
    )
    niftiAffineFix = np.concatenate([niftiAffine , np.array([[0.0 , 0.0 , 0.0 , 1.0]])] , axis=0)
    nifiImg = nib.Nifti1Image(
        data , 
        affine=niftiAffineFix
    )

    nib.save(nifiImg , output)

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

class randomGeo : # should be wrapped with py func
    def __init__(self , p):
        self.p = p
    def rot(self , img_arr , label_arr , * , imgOrder , lblOrder , imgCval , lblCval) :
        angle = tf.random.uniform(shape=() , minval=0 , maxval=360 , dtype=tf.int32)
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                ndimage.rotate(
                    img_arr ,
                    angle ,
                    axes=(0 , 1) ,
                    reshape=False,
                    order=imgOrder ,
                    mode='constant' ,
                    cval=imgCval
                ) , 
                ndimage.rotate(
                    label_arr ,
                    angle ,
                    axes=(0 , 1) ,
                    reshape=False,
                    order=lblOrder ,
                    mode='constant' ,
                    cval=lblCval
                )
            ) , 
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr
    
    def flipX(self , img_arr , label_arr) : 
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                tf.reverse(
                    img_arr ,
                    axis=[1]
                ) ,
                tf.reverse(
                    label_arr ,
                    axis=[1]
                )
            ) ,
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr
    
    def flipY(self , img_arr , label_arr) : 
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                tf.reverse(
                    img_arr ,
                    axis=[2]
                ) ,
                tf.reverse(
                    label_arr ,
                    axis=[2]
                )
            ) ,
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr
    
    def flipZ(self , img_arr , label_arr) : 
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                tf.reverse(
                    img_arr ,
                    axis=[0]
                ) ,
                tf.reverse(
                    label_arr ,
                    axis=[0]
                )
            ) ,
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr


def channelize(img , label) :
    img   = tf.expand_dims(img   , axis=0)
    label = tf.expand_dims(label , axis=0)
    return img , label


def convert_to_channel_last(img , label) :
    img = tf.transpose(
        img , 
        perm = [0 , 2 , 3 , 4 , 1]
    )

    label = tf.transpose(
        label ,
        perm = [0 , 2 , 3 , 4 , 1]
    )
    return img , label

class tile :
    def __init__(self , tile_dim):
        self.tile_dim = tile_dim
    def tile(self , img , label) :
        img   = tf.tile(img , self.tile_dim)
        #label = tf.tile(label , self.tile_dim)
        return img , label
   
def cast32(img , label) :
    img   = tf.cast(img   , dtype=tf.float32)
    label = tf.cast(label , dtype=tf.float32)

    return img , label

def cast16(img , label) :
    img   = tf.cast(img   , dtype=tf.float16)
    label = tf.cast(label , dtype=tf.float16)
    
    return img , label


class setShape :
    def __init__(self , imgShape  , labelShape):
        self.imgShape   = imgShape
        self.labelShape = labelShape
    def set(self , img , label) :
        img.set_shape(self.imgShape)
        label.set_shape(self.labelShape)
        
        return img , label
def normalize(img , label) :
    max   = tf.math.reduce_max(
        img , 
        axis=[1 , 2 , 3 , 4] ,
        keepdims=True
    )
    min   = tf.math.reduce_min(
        img ,
        axis=[1 , 2 , 3 , 4] ,
        keepdims=True
    )
    n_img = (img - min)/(max - min)
    return n_img , label

class histogramWindowDefine() :
    pass