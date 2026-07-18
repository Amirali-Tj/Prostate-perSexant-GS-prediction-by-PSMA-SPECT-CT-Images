import keras


# simple backbone in 2d space
ResNet50 = keras.applications.ResNet50(
    include_top=False ,
    weights="imagenet" ,
    input_shape=(64 , 64 , 3)
)

