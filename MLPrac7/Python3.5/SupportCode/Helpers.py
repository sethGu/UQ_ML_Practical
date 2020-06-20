
__author__ = "Llewyn Salt"

import os
import webbrowser
import subprocess
import signal
import platform
import re
from math import sqrt
import tensorflow as tf
import time

def convParams(topology):
  """
  Sets the convNet defaults
  """
  topology.setdefault('convPoolLayers',2)
  topology.setdefault('filterSize',5)
  topology.setdefault('convStride',1)
  topology.setdefault('numFilters',32)
  topology.setdefault('poolK',2)
  topology.setdefault('poolStride',2)
  topology.setdefault('FCLayerSize',1024)
  return topology

def optimiserParams(optDic):
  """
  Sets the default optimisation parameters
  """
  validOptimisers =["GradientDescent","Adam","RMSProp",
  "Momentum","Adagrad"]
  optimisation = {}
  opt=optDic.setdefault('optMethod',"GradientDescent")
  if opt not in validOptimisers:
    return None
  optimisation['learning_rate']=optDic.setdefault('learning_rate',0.001)
  if opt == "GradientDescent":
    optimiser=tf.optimizers.SGD(**optimisation)
  elif opt == "Momentum":
    optimisation["momentum"]=optDic.setdefault('momentum',0.9)
    optimiser = tf.train.MomentumOptimizer(**optimisation)
  elif opt=="Adagrad":
    optimisation["initial_accumulator_value"]=optDic.setdefault('initial_accumulator_value',0.1)
    optimiser = tf.train.AdagradOptimizer(**optimisation)
  elif opt=="RMSProp":
    optimisation["momentum"]=optDic.setdefault('momentum',0.0)
    optimisation["decay"]=optDic.setdefault('decay',0.9)
    optimisation["centered"]=optDic.setdefault('centered',False)
    optimiser = tf.train.RMSPropOptimizer(**optimisation)
  elif opt=="Adam":
    optimisation["beta1"]=optDic.setdefault('beta1',0.9)
    optimisation["beta2"]=optDic.setdefault('beta2',0.999)
    optimiser = tf.train.AdamOptimizer(**optimisation)
  return optimiser



def put_kernels_on_grid (kernel, pad = 1):
    #This function was obtained from https://gist.github.com/kukuruza/03731dc494603ceab0c5
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    # this organisation ensures that the channel value is 1 (Making it 3 could be useful but no idea)
    # this modifies the original gist code which did not enforce channel = 1 

    #x7 = tf.transpose(x6, (2, 0, 1, 3))
    return tf.transpose(x6, (2, 0, 1, 3)) 
    # scaling to [0, 255] for the tensor board.
    #return  tf.image.convert_image_dtype(x7, dtype = tf.uint8) 

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def dropoutLayer(inputs):
  with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(inputs, keep_prob)
    return dropped

def conv2d(x, W, stride=[1,1,1,1]):
  return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

def maxPool(x,K=[1,2,2,1],stride=[1,2,2,1]):
  return tf.nn.max_pool(x, ksize=K,strides=stride, padding='SAME')

def convLayer(x,layerName,layerNum,filterSize=5,convStride=1,numFilters=32,
  poolK=2,poolStride=2, act=tf.nn.relu):
  cStride=[1,convStride,convStride,1]
  pK=[1,poolK,poolK,1]
  pStride=[1,poolStride,poolStride,1]
  numChannels = (layerNum+1)*numFilters
  with tf.name_scope(layerName):
    with tf.name_scope("weights"):
      WConv = weight_variable([filterSize,filterSize,
        int(x.shape[3]),numChannels])
      variable_summaries(WConv)
      with tf.variable_scope('Weight_Visualisation'):
        grid = put_kernels_on_grid(WConv)
        tf.summary.image(layerName+'/features', grid, 1)
    with tf.name_scope("biases"):
      bConv = bias_variable([numChannels])
      variable_summaries(bConv)
    hConv = act(conv2d(x,WConv,cStride)+bConv)
    hPool = maxPool(hConv,pK,pStride)
  return hPool

def conv2FCLayer(x,outputDim,layerName, act=tf.nn.relu):
  with tf.name_scope(layerName):
    with tf.name_scope('weights'):
      W = weight_variable([int(x.shape[1])*int(x.shape[2])*int(x.shape[3]),outputDim])
      variable_summaries(W)
      with tf.variable_scope('Weight_Visualisation'):
        # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
        x_min = tf.reduce_min(W)
        x_max = tf.reduce_max(W)
        weights_0_to_1 = (W - x_min) / (x_max - x_min)
        weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
        # to tf.image_summary format [batch_size, height, width, channels]
        weights_transposed = tf.reshape(weights_0_to_255_uint8, [-1, int(weights_0_to_255_uint8.shape[0]), 
          int(weights_0_to_255_uint8.shape[1]), 1])
        # this will display random 3 filters from the 64 in conv1
        tf.summary.image('Weight_Visualisation', weights_transposed,10)
    with tf.name_scope('biases'):
      b = bias_variable([outputDim])
      variable_summaries(b)
    xFlat = tf.reshape(x,[-1, int(x.shape[1])*int(x.shape[2])*int(x.shape[3])])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(xFlat,W)+b
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
  return activations

def FCLayer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
      # Visualize conv1 features
      with tf.variable_scope('heatmap'):
        # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
        x_min = tf.reduce_min(weights)
        x_max = tf.reduce_max(weights)
        weights_0_to_1 = (weights - x_min) / (x_max - x_min)
        weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
        # to tf.image_summary format [batch_size, height, width, channels]
        weights_transposed = tf.reshape(weights_0_to_255_uint8, [-1, int(weights_0_to_255_uint8.shape[0]), 
          int(weights_0_to_255_uint8.shape[1]), 1])
        # this will display random 3 filters from the 64 in conv1
        tf.summary.image('heatmap', weights_transposed,10)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def killProcessesOnPorts(portTrain,portTest):
    ports=[str(portTrain),str(portTest)]
    if "Windows" in platform.system():
        popen = subprocess.Popen(['netstat', '-a','-n','-o'],
                           shell=False,
                           stdout=subprocess.PIPE)
    else:
        popen = subprocess.Popen(['netstat', '-lpn'],
                         shell=False,
                         stdout=subprocess.PIPE)
    (data, err) = popen.communicate()
    data = data.decode("utf-8")
    
    if "Windows" in platform.system():
        for line in data.split('\n'):
            line = line.strip()
            for port in ports:
                if '127.0.0.1:' + port in line and "0.0.0.0:" in line:
                    pid = line.split()[-1]
                    subprocess.Popen(['Taskkill', '/PID', pid, '/F'])
    else:
        pattern = "^tcp.*((?:{0})).* (?P<pid>[0-9]*)/.*$"
        pattern = pattern.format(')|(?:'.join(ports))
        prog = re.compile(pattern)
        for line in data.split('\n'):
            match = re.match(prog, line)
            if match:
                pid = match.group('pid')
                subprocess.Popen(['kill', '-9', pid])

def openTensorBoard(trainPath,testPath,portTrain=8001,portTest=8002):
  urlTrain = 'http://localhost:'+str(portTrain)
  urlTest = 'http://localhost:'+str(portTest)
  
  killProcessesOnPorts(portTrain,portTest)
  proc = subprocess.Popen(['tensorboard', '--logdir=' + trainPath,'--host=localhost', '--port=' + str(portTrain)])
  proc2 = subprocess.Popen(['tensorboard', '--logdir=' + testPath,'--host=localhost', '--port=' + str(portTest)])
  time.sleep(4)
  webbrowser.open(urlTrain)
  webbrowser.open(urlTest)

def getSaveDir(model):
  directory = os.path.dirname(os.path.abspath(__file__))
  if "Windows" in platform.system():
    directory="/".join(directory.split("\\"))
  directory=directory.rsplit("/",1)[0] + "/tensorflowResults/"
  if not os.path.exists(directory):
    os.makedirs(directory)
  dirList = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
  dirName = model+"train"
  maxNum = -1
  for d in dirList:
    if dirName in d:
      num = list(map(int, re.findall('\d+', d)))
      if len(num)==1:
          if num[0]>maxNum:
              maxNum=num[0]
  trainDir = directory + dirName + str(maxNum+1) +'/'
  testDir = directory + model+"test" + str(maxNum+1) + '/'
  return trainDir, testDir

def openTensorBoardAtIndex(model,ind,portTrain=8001,portTest=8002):
  directory = os.path.dirname(os.path.abspath(__file__))
  if "Windows" in platform.system():
    directory="/".join(directory.split("\\"))
  directory=directory.rsplit("/",1)[0] + "/tensorflowResults/"
  if not os.path.exists(directory):
    print("You need to run at least a model")
    return
  dirList = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
  dirName = model+"train"
  foundDir = False
  for d in dirList:
    if dirName in d:
      num = list(map(int, re.findall('\d+', d)))
      if len(num)==1:
          if num[0]==ind:
              foundDir=True
  if not foundDir:
    print("That index does not exist")
    return
  else:
    trainDir = directory + dirName + str(ind) +'/'
    testDir = directory + model+"test" + str(ind) + '/'
    openTensorBoard(trainDir,testDir,portTrain,portTest)