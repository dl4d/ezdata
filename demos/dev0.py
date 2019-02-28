import sys, os
sys.path.append(os.path.abspath('..\\ezdata'))

import keras

# ---------------------- [       EZ DATA      ] -----------------------------

from ezdata.images import ezdata_images
from ezdata.images import ezdata_images_trainer
from ezdata.utils  import preprocess, gen_trainval

ez = ezdata_images()

parameters = {
    "path"  : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "name"  : "Bacteria",
}
ez.import_classification(parameters)

ez.to_keras(resize=(32,32))

ez.gen_test(size=0.2)

X_scaled,scalerX = preprocess(ez.X, type="minmax")
y_categ, scalerY = preprocess(ez.y, type="categorical")

X_train,X_valid,y_train,y_valid = gen_trainval(X_scaled,y_categ,size = 0.2)

ez.assign(train = (X_train,y_train),valid = (X_valid,y_valid),scaler = (scalerX,scalerY))

# ---------------------- [   KERAS + EZ NETWORK   ] --------------------------------

from keras.models import Input,Model
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense


inputs = ez.input_network()
#  -- Keras network --
x = Conv2D(6, kernel_size = (5, 5), strides=(1,1), padding="valid",input_shape=(32, 32, 1)) (inputs)
x = Activation("relu") (x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
x = Conv2D(16, kernel_size = (5, 5), strides=(1,1), padding="valid") (x)
x = Activation("relu") (x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
x = Flatten() (x)
x = Dense(120) (x)
x = Dense(84) (x)
# --
outputs = ez.output_network(x)
network = Model(inputs=inputs,outputs=outputs)

# --------------------- [   KERAS + EZ OPTIMIZER   ] ------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : "categorical_crossentropy",
    "metrics": ["accuracy"]
}

ez.compile(network = network,optimizer=optimizer)

# --------------------- [   KERAS + EZ TRAINING   ] ------------------------------
parameters = {
    "epochs":10
}
ez.train(parameters=parameters)
# --------------------- [   KERAS + EZ EVALUATION   ] ------------------------------

ez.evaluate()


# --------------------- [   EZ SAVE   ] ------------------------------
ez.save("model0")
