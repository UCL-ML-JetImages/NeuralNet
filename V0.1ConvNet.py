
# NOTE:
  #/Library/Python/2.7/site-packages/keras/utils/np_utils.py:23: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
#Y[i, y[i]] = 1.



from __future__ import print_function

import root_numpy
import ROOT
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import pydot
from keras.utils.visualize_util import plot


batch_size = 116
nb_classes = 2
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 25, 25
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

#print whole array
np.set_printoptions(threshold='nan')


backgroundFilename = "outputjetBackground.root"
signalFilename = "outputWSignals.root"

sig = ROOT.TFile.Open(signalFilename)
bkg = ROOT.TFile.Open(backgroundFilename)

Nbkg = len(bkg.GetListOfKeys())
Nsig = len(sig.GetListOfKeys())

bkgArray = []

sigArray = []

for i in range(Nbkg):
  histoname = "histo"+str(i)
  #print("Getting Background Histogram:", histoname)
  aHisto = bkg.Get(histoname)
  aHisto.Print() #This prevents all the array elements being 0.

  anArrayfromHisto = root_numpy.hist2array(aHisto)
  anArrayfromHisto = np.rot90(anArrayfromHisto)

  bkgArray.append(anArrayfromHisto)


for i in range(Nsig):
  histoname = "histo"+str(i)
  #print("Getting Signal Histogram:", histoname)
  aHisto = sig.Get(histoname)
  aHisto.Print() #This prevents all the array elements being 0.
  anArrayfromHisto = root_numpy.hist2array(aHisto)
  anArrayfromHisto = np.rot90(anArrayfromHisto)
  #append histo array only if sum != 0
  if np.sum(anArrayfromHisto) != 0.0:
  	sigArray.append(anArrayfromHisto)



#print(len(bkgArray))
#print(len(sigArray))



bkgTwoThirds = 9*len(bkgArray)/10
sigTwoThirds = 9*len(sigArray)/10

bkgTrain, bkgTest = bkgArray[:bkgTwoThirds], bkgArray[bkgTwoThirds:]
sigTrain, sigTest = sigArray[:sigTwoThirds], sigArray[sigTwoThirds:]

bkgTrainVal = np.zeros(len(bkgTrain))
bkgTestVal = np.zeros(len(bkgTest))

sigTrainVal = np.ones(len(sigTrain))
sigTestVal = np.ones(len(sigTest))

TrainList = bkgTrain + sigTrain
TestList = bkgTest + sigTest

TrainVals = np.concatenate((bkgTrainVal, sigTrainVal),axis=0)
TestVals = np.concatenate((bkgTestVal, sigTestVal),axis=0)




#print(TrainVals[160])


TrainList = np.array(TrainList)
TestList = np.array(TestList)



#input_shape = (TrainList.shape[0], TrainList.shape[1] , TrainList.shape[2]) #N.B formatted for tensorflow, theano requires alternate format
input_shape = ( TrainList.shape[1] , TrainList.shape[2], 1) #N.B formatted for tensorflow, theano requires alternate format
TrainList = TrainList.reshape(TrainList.shape[0], img_rows, img_cols, 1)
TestList = TestList.reshape(TestList.shape[0], img_rows, img_cols, 1)


#print(TrainList.shape)
#print(input_shape)

# convert class vectors to binary class matrices 
TrainVals = np_utils.to_categorical(TrainVals, nb_classes)
TestVals = np_utils.to_categorical(TestVals, nb_classes)



model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
#print(model.output_shape)
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))


model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))

model.add(Flatten())


model.add(Dense(232))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(TrainList, TrainVals, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(TestList, TestVals))
score = model.evaluate(TestList, TestVals, verbose=0)









# Exception: You are passing a target array of shape (232, 1) while using as loss `categorical_crossentrop
# y`. `categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, clas
# ses). If your targets are integer classes, you can convert them to the expected format via:
# ```
# from keras.utils.np_utils import to_categorical
# y_binary = to_categorical(y_int)
# ```

# Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)



# print('Test score:', score[0])
# print('Test accuracy:', score[1])










# plot(model, to_file='model.png')







#  bkgArray = np.append(bkgArray, anArrayfromHisto)
# bkgArray.itemset(i,anArrayfromHisto)




# TODO truncation issue







# plt.imshow(TestList[3], cmap = "Greys")

# plt.show()


# print("---------PRINTING ARRAY FROM HISTO ------------")
# print(anArrayfromHisto)
# print(type(anArrayfromHisto))
# print(type(anArrayfromHisto[1,1]))

