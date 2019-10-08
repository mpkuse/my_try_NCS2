import numpy as np
import tensorflow as tf
import code
import time
import cv2
import math


#import TerminalColors
#tcol = TerminalColors.bcolors()
def write_kerasmodel_as_tensorflow_pb( model, LOG_DIR, output_model_name='output_model.pb' ):
    """ Takes as input a keras.models.Model() and writes out
        Tensorflow proto-binary.
    """
    print( '## [write_kerasmodel_as_tensorflow_pb] Start' )

    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    tf.keras.backend.set_learning_phase(0)
    sess = tf.keras.backend.get_session()



    # Make const
    print( 'Make Computation Graph as Constant and Prune unnecessary stuff from it' )
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        [node.op.name for node in model.outputs])
    constant_graph = tf.graph_util.remove_training_nodes(constant_graph)


    #--- convert Switch --> Identity
    # I am doing this because TensorRT cannot process Switch operations.
    # # https://github.com/tensorflow/tensorflow/issues/8404#issuecomment-297469468
    # for node in constant_graph.node:
    #     if node.op == "Switch":
    #         node.op = "Identity"
    #         del node.input[1]
    # # END

    # Write .pb
    # output_model_name = 'output_model.pb'
    print( "##", 'Write ', output_model_name )
    print( 'model.outputs=', [node.op.name for node in model.outputs] )
    graph_io.write_graph(constant_graph, LOG_DIR, output_model_name,
                     as_text=False)
    print( '## [write_kerasmodel_as_tensorflow_pb] Done' )


    # Write .pbtxt (for viz only)
    output_model_pbtxt_name = output_model_name+'.pbtxt' #'output_model.pbtxt'
    print( '## Write ', output_model_pbtxt_name )
    tf.train.write_graph(constant_graph, LOG_DIR,
                      output_model_pbtxt_name, as_text=True)

    # Write model.summary to file (to get info on input and output shapes)
    output_modelsummary_fname = LOG_DIR+'/'+output_model_name + '.modelsummary.log'
    print(  '## Write ', output_modelsummary_fname )
    with open(output_modelsummary_fname,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))



#---------------------------------------------------------------------------
# Load Model (h5)
model = tf.keras.models.load_model( 'mnist_cnn.h5' )
model.summary()

#--------------------------------------------------------------------------
# Save as Frozen Tensorflow
write_kerasmodel_as_tensorflow_pb( model, './' )

#---------------------------------------------------------------------------
# Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train: 60K x 28 x 28
# y_train: 60K x 1
# x_test : 10K x 28 x 28
# y_test : 10K x 1


#---------------------------------------------------------------------------
# Predict
import time
st = time.time()
n_iterations = 10000
for _ in range(n_iterations):
    r = np.random.randint( x_test.shape[0] )
    pred_outs = model.predict( x_test[r,:,:].reshape( 1, 28,28,1) )
    # print( 'r=', r )
    # print( 'predicted = ', pred_outs.argmax(), )
    # print( 'ground truth = ', y_test[r], )
    # print( '' )
    # cv2.imshow( 'test image', x_test[r,:,:].astype('uint8') )
    # cv2.waitKey(0)
print( '%d iterations took %4.4f ms' %(n_iterations,  1000. * ( time.time() - st ) ) )
