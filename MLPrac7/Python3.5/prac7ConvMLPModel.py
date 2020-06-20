from math import sqrt
import tensorflow as tf
import os
from SupportCode import Helpers

from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
__author__ = "Llewyn Salt"

def prac7ConvMLPModel(model='MLP',MLPTop={},convTop={},optimiser={},
  act=tf.nn.relu,max_steps=100):
  # Import data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  # Create Inputs x is MNIST image and y_labels is the label
#   tf.python.framework.ops.reset_default_graph()
  ops.reset_default_graph()
  sess = tf.compat.v1.InteractiveSession()
  optimise = Helpers.optimiserParams(optimiser)
  if optimise==None:
    print("Invalid Optimiser")
    return
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_labels = tf.placeholder(tf.float32, [None, 10], name='y-input')
  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)
  #Generate hidden layers
  layers={}
  if model=='convNet':
    topology = Helpers.convParams(convTop)
    FCLayerSize=topology.pop('FCLayerSize')
    for i in range(topology.pop('convPoolLayers')):
      if i==0:
        layers[str(i)] = Helpers.convLayer(image_shaped_input,"convPoolLayer"+str(i),i,**topology,act=act)
      else:
        layers[str(i)] = Helpers.convLayer(layers[str(i-1)],"convPoolLayer"+str(i),i,**topology,act=act)
    FC1 = Helpers.conv2FCLayer(layers[str(i)],FCLayerSize,"FC1")
    y = Helpers.FCLayer(FC1, FCLayerSize, 10, 'output_layer', act=tf.identity)
  elif model =='MLP':
    hiddenDims = MLPTop.setdefault("hiddenDims",[500])
    for i in range(len(hiddenDims)):
      if i==0:
        layers[str(i)] = Helpers.FCLayer(x, 784, hiddenDims[i],"hidden_layer_"+str(i),act=act)
      else:
        layers[str(i)] = Helpers.FCLayer(layers[str(i-1)],hiddenDims[i-1],hiddenDims[i],"hidden_layer_"+str(i),act=act)
    y = Helpers.FCLayer(layers[str(i)], hiddenDims[i], 10, 'output_layer', act=tf.identity)
  else:
    print("MLP or convNet - nothing else is valid")
    return
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)
  with tf.name_scope('train'):
    train_step = optimise.minimize(cross_entropy)
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  trainPath,testPath = Helpers.getSaveDir(model)
  train_writer = tf.summary.FileWriter(trainPath, sess.graph)
  test_writer = tf.summary.FileWriter(testPath)
  tf.global_variables_initializer().run()
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = mnist.train.next_batch(100)
    else:
      xs, ys = mnist.test.images, mnist.test.labels
    return {x: xs, y_labels: ys}

  for i in range(max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print(('Accuracy at step %s: %s' % (i, acc)))
    else:  # Record train set summaries, and train
      if i % 25 == 24:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print(('Adding run metadata for', i))
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()
  print(("Accuracy on test set: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_labels: mnist.test.labels})))
  sess.close()
  Helpers.openTensorBoard(trainPath,testPath)
  