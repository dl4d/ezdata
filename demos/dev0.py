import sys, os
sys.path.append(os.path.abspath('..\\ezdata'))

from ezdata.images import ezdata_images

ez = ezdata_images()

parameters = {
    "path"  : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "name"  : "Bacteria",
}

ez.import_classification(parameters)

ez.to_keras(resize=(32,32))

X_test,y_test = ez.gen_test(size=0.2)

X_train,y_train,X_valid,y_valid = ez.gen_train_val(size=0.2)
