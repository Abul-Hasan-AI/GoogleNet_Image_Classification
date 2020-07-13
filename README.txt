# - General Software dependencies - #
# The code was build on and reqires the following version of python libraries
1. python 3.6
2. tensorflow-gpu == 1.14.0
3. keras == 2.2.4 with tensorflow backend and "image_data_format": "channels_last"
4. imagio
5. numpy
6. matplotlib

# - Instructions on running the files - #
1. GoogleNet.py:
 -It contains code for googlenet. It depends on lrn.py and pool_helper.py.
 -change value of 'dataset' variable on line 37 to change the dataset(between mnist and cifar10).
 -the variable 'runtype' on line 38 can be changed to switch between training and predicting.(Note: Model needs to be trained atleast once before predicting)

 2. VGG_16.py:
  - It contains code for VGG 16 layers. 
  - change the variable 'dataset' variable on line 14 to change the dataset(between mnist and cifar10).
