# Loads the IR file (.xml) and does inference on MYRIAD device
# This needs python3
import numpy as np
import cv2
import tensorflow as tf


GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
DEVICE = "MYRIAD"

def display_info(input_shape, output_shape, ir):
    print()
    print(YELLOW + 'Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Plugin:       ' + NOCOLOR + 'Myriad')
    print('   - ' + YELLOW + 'IR File:     ' + NOCOLOR, ir)
    print('   - ' + YELLOW + 'Input Shape: ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shape:' + NOCOLOR, output_shape)
    #print('   - ' + YELLOW + 'Labels File: ' + NOCOLOR, labels)
    #print('   - ' + YELLOW + 'Mean File:   ' + NOCOLOR, mean)
    #print('   - ' + YELLOW + 'Image File:   ' + NOCOLOR, image)

try:
    from openvino.inference_engine import IENetwork, IECore
except:
    print('\nPlease make sure your OpenVINO environment variables are set by sourcing the `setupvars.sh` script found in <your OpenVINO install location>/bin/ folder.\n')
    exit(1)



# Data for prediction
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train: 60K x 28 x 28
# y_train: 60K x 1
# x_test : 10K x 28 x 28
# y_test : 10K x 1


# Select the myriad plugin and IRs to be used
ir = './output_model.xml'
ie = IECore()
net = IENetwork(model = ir, weights = ir[:-3] + 'bin')


# Set up the input and output blobs
input_blob = next(iter(net.inputs))
output_blob = next(iter(net.outputs))
input_shape = net.inputs[input_blob].shape
output_shape = net.outputs[output_blob].shape
display_info(input_shape, output_shape, ir )


# Load the network and get the network shape information
exec_net = ie.load_network(network = net, device_name = DEVICE)
#n, c = input_shape


# Predict
import time
st = time.time()
n_iterations = 5000

for _ in range(n_iterations):
    r = np.random.randint( x_test.shape[0] )
    # pred_outs = model.predict( x_test[r,:,:].reshape( 1, 28*28) )
    #res = exec_net.infer({input_blob: x_test[r,:,:].reshape( 1, 28*28)}) #for MLP
    res = exec_net.infer({input_blob: x_test[r,:,:].reshape( 1, 1,28,28)}) #for CNN

    output_logits = res[output_blob][0]
    # print( 'r=', r )
    # print( 'res[output_blob] = ', output_logits.argmax() )
    # print( 'ground truth = ', y_test[r], )
    # print( '' )
    # cv2.imshow( 'test image', x_test[r,:,:].astype('uint8') )
    # cv2.waitKey(0)
print( '%d iterations took %4.4f ms' %(n_iterations,  1000. * ( time.time() - st ) ) )
