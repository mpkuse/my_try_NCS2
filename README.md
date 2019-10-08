# Bare Bones Example on running Keras model on NCS2 (Intel Compute Stick 2)

Note: All these need python3 to run. Also ensure tensorflow (needed for training and convert keras files to .pb)
is installed on python3.

## Train a model
```
python3 mnist_mlp.py
```

## Keras Model to Frozen Tensorflow Graph
This will load the saved .h5 (keras model), convert to frozen tensorflow graph
```
python3 mnist_predict.py
```

## Frozen Graph (.pb) to Intel's (IR)
```
source ~/intel/openvino/bin/setupvars.sh
mo_tf.py --input_model output_model.pb --input_shape "(1,784)"
```

## Execute IR On Device (NCS2 aka MYRIAD)
```
python3 run_on_ncs2.py
```
