# Twitter
Twitterに関するもの

import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import os
import glob
import scipy
from scipy import io
import time
from tensorflow.contrib import learn
import random

###############################################################################
# Constants for the image input and output.
###############################################################################

# Output folder for the images.
OUTPUT_DIR = 'output/'
# Style image to use.
STYLE_IMAGE = 'images/guernica.jpg'
# Content image to use.
CONTENT_IMAGE = 'images/hongkong.jpg'
# Image dimensions constants. 
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400
COLOR_CHANNELS = 3

###############################################################################
# Algorithm constants
###############################################################################
# Noise ratio. Percentage of weight of the noise for intermixing with the
# content image.
NOISE_RATIO = 0.6

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def build_model(input_img,IMAGE_WIDTH,IMAGE_HEIGHT,CHANNEL):
  def conv_layer(layer_name, layer_input, W):
    conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
    # if args.verbose: print('--{} | shape={} | weights_shape={}'.format(layer_name, 
    #   conv.get_shape(), W.get_shape()))
    return conv

  def relu_layer(layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    # if args.verbose: 
    #   print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), 
    #     b.get_shape()))
    return relu

  def pool_layer(layer_name, layer_input):
    # if args.pooling_type == 'avg':
    pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], 
      strides=[1, 2, 2, 1], padding='SAME')
    # elif args.pooling_type == 'max':
      # pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], 
      #   strides=[1, 2, 2, 1], padding='SAME')
    # if args.verbose: 
    #   print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
    return pool

  def get_weights(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    W = tf.constant(weights)
    return W

  def get_bias(vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    return b

  # if args.verbose: print('\nBUILDING VGG-19 NETWORK')
  net = {}
  # _, h, w, d     = input_img.shape
  
  # if args.verbose: print('loading model weights...')
  vgg_rawnet     = scipy.io.loadmat(VGG_MODEL)
  vgg_layers     = vgg_rawnet['layers'][0]
  # if args.verbose: print('constructing layers...')
  net['input']   = tf.Variable(np.zeros((1,IMAGE_WIDTH,IMAGE_HEIGHT,CHANNEL)), dtype=np.float32)

  # if args.verbose: print('LAYER GROUP 1')
  # net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
  # net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))

  # net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
  # net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))
  
  # net['pool1']   = pool_layer('pool1', net['relu1_2'])

  # # if args.verbose: print('LAYER GROUP 2')  
  # net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
  # net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))
  
  # net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
  # net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))
  
  # net['pool2']   = pool_layer('pool2', net['relu2_2'])
  
  # # if args.verbose: print('LAYER GROUP 3')
  # net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
  # net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))

  net['conv3_2'] = conv_layer('conv3_2', net['input'], W=get_weights(vgg_layers, 12))
  net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))

  net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
  net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))

  net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
  net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))

  net['pool3']   = pool_layer('pool3', net['relu3_4'])

  # if args.verbose: print('LAYER GROUP 4')
  net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
  net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))

  # net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
  # net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))

  # net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
  # net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))

  # net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
  # net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))

  # net['pool4']   = pool_layer('pool4', net['relu4_4'])

  # if args.verbose: print('LAYER GROUP 5')
  # net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
  # net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))

  # net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
  # net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))

  # net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
  # net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))

  # net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
  # net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))

  # net['pool5']   = pool_layer('pool5', net['relu5_4'])

  return net




def createImage(tensor):
    newImage = np.zeros((len(tensor[0]),len(tensor[0][0]),3))
    for z in range(len(tensor[0][0][0])):
        for y in range(len(tensor[0])):
          for x in range(len(tensor[0][y])):
            newImage[y][x][0] = tensor[0][y][x][z]
            newImage[y][x][1] = tensor[0][y][x][z]
            newImage[y][x][2] = tensor[0][y][x][z]
        newImage = np.asarray(newImage).astype('float64')
        newImage += 127.0#MEAN_VALUES[0]
        # newImage = newImage[:, :, ::-1]
        newImage = np.clip(newImage, 0, 255).astype('uint8')
    print(newImage.shape)
    return newImage

def getPhi_L_MtoN(M,N,patch):
    Phi = np.zeros((len(M),len(M[0]),1))
    for My in range(len(M)):
        for Mx in range(len(M[My])):
            Mpositions = getPatchPosition(M,Mx,My,patch)
            MFxVector=[]
            for i in range(len(Mpositions)):
                MFxVector.append(getNormalizedFx(M, Mpositions[i][0], Mpositions[i][1]))

            minimum = 100000
            minimum_positions = []
            for Ny in range(len(N)):
                for Nx in range(len(N[Ny])):
                    Npositions = getPatchPosition(N,Nx,Ny,patch)
                    total = 0

                    for i in range(len(Npositions)):
                        NFx = getNormalizedFx(N,Npositions[i][0],Npositions[i][1])
                        total += abs(MFxVector[i] - NFx) ** 2

                    if minimum > total:
                        minimum = total
                        minimum_positions = [Nx,Ny]
            Phi[My][Mx] = minimum_positions
    return Phi

def getPhi_Random(image):
    print(len(image))
    print(len(image[0]))
    print(len(image[0][0]))
    ret = []
    for y in range(len(image)):
        for x in range(len(image[y])):
            ret.append(image[x][y])
    random.shuffle(ret)
    ret = np.asarray(ret)
    randomRet = np.reshape(ret,(len(image),len(image[0]),3))
    return randomRet


def warp(image,Phi):
    ret = np.zeros_like(image)
    for y in range(len(Phi)):
        for x in range(len(Phi[y])):
            newX = Phi[y][x][0]
            newY = Phi[y][x][1]
            ret[newY][newX] = image[y][x]
    return ret




def batch_norm_wrapper(inputs, is_training, decay = 0.999):
        epsilon = 1e-5
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        rank = len(inputs.get_shape())
        axes = []  # nn:[0], conv:[0,1,2]
        for i in range(rank - 1):
            axes.append(i)
        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,axes)
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def minimize_with_lbfgs(sess, net, optimizer, init_img, goal_img):
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  # sess.run(net['y_'].assign(goal_img))
  optimizer.minimize(sess)

def minimize_with_adam(sess, net, optimizer, init_img, goal_img, loss,max_iterations,print_iterations):
  train_op = optimizer.minimize(loss)
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  # sess.run(net['y_'].assign(goal_img))
  iterations = 0
  while (iterations < max_iterations):
    sess.run(train_op)
    if iterations % print_iterations==0:
      curr_loss = loss.eval()
      print("At iterate {}\tf=  {:.5E}".format(iterations, curr_loss))
      cv2.imwrite("./output/"+str(iterations)+".jpg",createImage(sess.run(net['conv4_1'])))

    iterations += 1


def get_optimizer(loss):
  # print_iterations = args.print_iterations if args.verbose else 0
  if args.optimizer == 'lbfgs':
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      loss, method='L-BFGS-B',
      options={'maxiter': args.max_iterations
                  ,'disp': 1000
                  })
  elif args.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
  return optimizer



def getLoss(sess, vggModel, layer, generate_image, goal_image):
    print(generate_image.shape)
    print(goal_image.shape)
    sess.run(vggModel['input'].assign(generate_image))
    cnnOutput = sess.run(vggModel['conv4_1'])
    # cnnOutput = getMono(createImage(cnnOutput))
    cnnOutput = tf.convert_to_tensor(np.asarray(cnnOutput,dtype='float32'),tf.float32)
    # temp = tf.argmax(vggModel['conv5_1'],3)
    # temp = tf.cast(temp,tf.float32)
    cnnOutput = vggModel['conv4_1'] * tf.cast(cnnOutput,tf.float32)
    # goal_image = tf.convert_to_tensor(goal_image,tf.float32) 
    # return tf.nn.l2_loss(cnnOutput - goal_image)
    # return style_layer_loss(vggModel['y_'],cnnOutput)
    sub = tf.subtract(cnnOutput,vggModel['y_'])
    abso= tf.norm(sub)
    return tf.reduce_sum(tf.pow(abso,2))


def sum_total_variation_losses(sess, net, input_img):
  b, h, w, d = input_img.shape
  x = net['input']
  tv_y_size = b * (h-1) * w * d
  tv_x_size = b * h * (w-1) * d
  loss_y = tf.nn.l2_loss(x[:,1:,:,:] - x[:,:-1,:,:]) 
  loss_y /= tv_y_size
  loss_x = tf.nn.l2_loss(x[:,:,1:,:] - x[:,:,:-1,:]) 
  loss_x /= tv_x_size
  loss = 2 * (loss_y + loss_x)
  loss = tf.cast(loss, tf.float32)
  return loss

def style_layer_loss(a, x):
  _, h, w, d = a.get_shape()
  M = h.value * w.value
  N = d.value
  A = gram_matrix(a, M, N)
  G = gram_matrix(x, M, N)
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
  return loss

def gram_matrix(x, area, depth):
  F = tf.reshape(x, (area, depth))
  G = tf.matmul(tf.transpose(F), F)
  return G

def get_optimizer(loss, select_optimizer, learning_rate, max_iterations, print_iterations):
  if select_optimizer == 'lbfgs':
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      loss, method='L-BFGS-B',
      options={'maxiter': max_iterations
                  # ,'disp': print_iterations
                  })
  elif select_optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  return optimizer



def deconv(sess, init_img, goal_image, channel, select_optimizer, max_iterations):
    graph = deconvGrapf(init_img, goal_image, channel)
    tf.global_variables_initializer().run()

         
    # output_img = sess.run(x_image)
    L_total=getLoss(sess, graph, 'conv5_1', init_img, goal_image)
    optimizer=get_optimizer(L_total,select_optimizer, 1e-2, max_iterations,1000)
    if select_optimizer == 'adam':
        minimize_with_adam(sess, graph, optimizer, init_img, L_total,max_iterations)
    elif select_optimizer == 'lbfgs':
        minimize_with_lbfgs(sess, graph, optimizer, init_img)

    output_img = sess.run(graph['input'])
    print(output_img.shape)
    # output_img = createImage(output_img)
    # print(output_img.shape)
    output_img = np.clip(output_img,0,255).astype('uint8')
    cv2.imshow("",output_img[0])
    cv2.waitKey(0)
# def newDeconvGraph():

def newDeconv(sess, init_img, goal_img, vggModel, select_optimizer, max_iterations):
    # vggModel['y_'] = tf.Variable(tf.constant(goal_img))
    vggModel['y_'] = tf.constant(goal_img)

    L_total=getLoss(sess, vggModel, 'conv4_1', init_img, goal_img)
    # L_total = sum_total_variation_losses(sess, vggModel, init_img)
    optimizer=get_optimizer(L_total,select_optimizer, 1e-0, max_iterations,1000)
    if select_optimizer == 'adam':
        minimize_with_adam(sess, vggModel, optimizer, init_img, goal_img, L_total,max_iterations,1000)
    elif select_optimizer == 'lbfgs':
        minimize_with_lbfgs(sess, vggModel, optimizer, init_img, goal_img)

    output_img = sess.run(vggModel['conv4_1'])
    in_img = sess.run(vggModel['conv3_2'])

    # print(output_img.shape)
    # output_img = createImage(output_img)
    # print(output_img.shape)
    output_img = createImage(output_img)# + MEAN_VALUES
    # output_img = np.clip(output_img,0,255).astype('uint8')
    cv2.imshow("L4",output_img)
    cv2.waitKey(0)
    cv2.imshow("L3_2",createImage(in_img))
    cv2.waitKey(0)
    cv2.imshow("input",createImage(sess.run(vggModel['input'])))
    cv2.waitKey(0)

def generateNoiseImage(height, width, channel):
    randomByteArray = bytearray(os.urandom(height*width*channel)) #画素数文の乱数発生
    return np.asarray(randomByteArray).reshape((height, width, channel))

def getMono(img):
    # gray_img=None
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(gray_img.shape)
    # ret = np.zeros((len(img),len(img[0])))

    # for y in range(len(ret)):
    #     for x in range(len(ret[y])):
    #         ret[y][x] = gray_img[y][x][0]
    # return ret 



sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# model = load_vgg_model(VGG_MODEL,len(PhiAB[0]),len(PhiAB[0][0]),512)

# img = cv2.imread('avater.jpg')
# img = np.asarray(img).astype('float64')

# img -= MEAN_VALUES[0]
# img = img[:, :, ::-1].astype('float64')


# # img = img.transpose((2, 0, 1))
# img = np.expand_dims(img, axis=0)

# sess.run(model['input'].assign(np.asarray(img)))
# A=[]
# A.append(createImage(sess.run(model['conv1_1'])))
# A.append(createImage(sess.run(model['conv2_1'])))
# A.append(createImage(sess.run(model['conv3_1'])))
# A.append(createImage(sess.run(model['conv4_1'])))
# A.append(createImage(sess.run(model['conv5_1'])))

# PhiAB =getPhi_Random(A[5-1])
# noise = generateNoiseImage(IMAGE_HEIGHT,IMAGE_WIDTH,3)
# cv2.imshow("",noise)
# cv2.waitKey(0)
# PhiAB = getMono(PhiAB)
# noise = getMono(noise)
# cv2.imshow("",warp(A[5-1],PhiAB))
# cv2.waitKey(0)
# sess, generate_image, goal_image, channel, select_optimizer, max_iterations
# deconv(sess, np.asarray([noise],dtype=np.float32), np.asarray([PhiAB],dtype=np.float32),3,'adam', 20000)
# deconv(noise, PhiAB, 3, 20000)
PhiAB = getMono(cv2.imread("L4.jpg"))

goal = np.zeros((len(PhiAB),len(PhiAB[0]),512))
for y in range(len(PhiAB)):
    for x in range(len(PhiAB[y])):
        for z in range(256):
            goal[y][x][z] = PhiAB[y][x]


noise = generateNoiseImage(int(len(PhiAB)*2),int(len(PhiAB[0])*2),256)
noise = np.asarray(noise,dtype=np.float32)
noise -= 127.0#MEAN_VALUES[0]
model = build_model(noise,int(len(PhiAB)*2),int(len(PhiAB[0])*2),256)

print(noise.shape)
print(goal.shape)

# graph = TrueDeconvGrapf(np.asarray([noise],dtype=np.float32), np.asarray([PhiAB],dtype=np.float32), 3)
tf.global_variables_initializer().run()


newDeconv(sess, np.asarray([noise],dtype=np.float32), np.asarray([goal],dtype=np.float32), model, 'adam', 10001)
