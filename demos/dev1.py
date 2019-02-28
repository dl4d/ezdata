import sys, os
sys.path.append(os.path.abspath('..\\ezdata'))


from ezdata.images import ezdata_images
from ezdata.utils  import ez_load

ez = ez_load("model0")

ez.evaluate()

ez.train()
