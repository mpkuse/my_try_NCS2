# Adopted from Keras Examples
# Also see https://github.com/keras-team/keras/tree/master/examples

import numpy as np
import tensorflow as tf
import code
import time
import cv2
import math
#import matplotlib.pyplot as plt
#from viz_utils import labels_to_logits
def labels_to_logits( labels, n_classes=None ):
    if n_classes is None:
        n_classes = len( np.unique( labels ) )

    logits = np.zeros( (labels.shape[0], n_classes) )

    for i in range( len(labels) ):
        logits[i, labels[i] ] = 1.
    return logits

#---------------------------------------------------------------------------
# Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train: 60K x 28 x 28
# y_train: 60K x 1
# x_test : 10K x 28 x 28
# y_test : 10K x 1


#----------------------------------------------------------------------------
# Model
model = tf.keras.Sequential()
model.add( tf.keras.layers.Dense(512, activation='relu', input_shape=(784,) ) )
model.add( tf.keras.layers.Dropout(0.2) )
model.add( tf.keras.layers.Dense(512, activation='relu') )
model.add( tf.keras.layers.Dropout(0.2) )
model.add( tf.keras.layers.Dense(10, activation='softmax') )

model.summary()


#-----------------------------------------------------------------------------
# Compile
# optimizer = tf.keras.optimizers.Adam(lr=1e-5)
optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#----------------------------------------------------------------------------
# Iterations
if True:

    model.fit(x=x_train.reshape( 60000, 28*28),
              y=labels_to_logits(y_train),
              epochs=20, batch_size=128, verbose=2)

    model.save( 'mnist_mlp.h5' )


#---------------------------------------------------------------------------
# Load pretrained model
if False:
    model.load_weights( 'mnist_mlp.h5' )


#---------------------------------------------------------------------------
# Evaluate
score = model.evaluate( x_test.reshape( 10000, 28*28 ), labels_to_logits(y_test), verbose=1 )
print( 'Test Loss: ', score[0] )
print( 'Accuracy : ', score[1] )


#---------------------------------------------------------------------------
# Predict
for _ in range(30):
    r = np.random.randint( x_test.shape[0] )
    pred_outs = model.predict( x_test[r,:,:].reshape( 1, 28*28) )
    print( 'r=', r )
    print( 'predicted = ', pred_outs.argmax(), )
    print( 'ground truth = ', y_test[r], )
    print( '' )
    cv2.imshow( 'test image', x_test[r,:,:].astype('uint8') )
    cv2.waitKey(0)
