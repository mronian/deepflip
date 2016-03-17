import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt

import skimage.transform
import sklearn.cross_validation
import pickle
import os
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX


np.random.seed(42)

CLASSES = ['LabelMe_Original', '90LabelMe_Original', '180LabelMe_Original','270LabelMe_Original']
LABELS = {cls: i for i, cls in enumerate(CLASSES)}

def build_model():
	net = {}
	net['input'] = InputLayer((None, 3, 224, 224))
	net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
	net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
	net['pool1'] = PoolLayer(net['conv1_2'], 2)
	net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
	net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
	net['pool2'] = PoolLayer(net['conv2_2'], 2)
	net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
	net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
	net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
	net['pool3'] = PoolLayer(net['conv3_3'], 2)
	net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
	net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
	net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
	net['pool4'] = PoolLayer(net['conv4_3'], 2)
	net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
	net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
	net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
	net['pool5'] = PoolLayer(net['conv5_3'], 2)
	net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
	net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
	net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
	net['prob'] = NonlinearityLayer(net['fc8'], softmax)
	return net

d = pickle.load(open('ModelZoo/vgg16.pkl'))

net = build_model()
lasagne.layers.set_all_param_values(net['prob'], d['param values'])


IMAGE_MEAN = d['mean value'][:, np.newaxis, np.newaxis]

def prep_image(fn, ext='jpg'):
	im = plt.imread(fn, ext)
	#h, w, _ = im.shape
	#if h < w:
	#	im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
	#else:
	#	im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
	h, w, _ = im.shape
	im = im[h//2-112:h//2+112, w//2-112:w//2+112]
	#rawim = np.copy(im).astype('uint8')
	im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
	im = im[:3]
	im = im[::-1, :, :]
	im = im - IMAGE_MEAN
	return floatX(im[np.newaxis])

# Load and preprocess the entire dataset into numpy arrays
X = []
y = []

for cls in CLASSES:
	for fn in os.listdir('./Datasets/deepflip/{}'.format(cls)):
		if np.random.rand()>0.5:
			continue
		im = prep_image('./Datasets/deepflip/{}/{}'.format(cls, fn))
		X.append(im)
		y.append(LABELS[cls])
		
X = np.concatenate(X)
y = np.array(y).astype('int32')
print X.shape

# Split into train, validation and test sets
train_ix, test_ix = sklearn.cross_validation.train_test_split(range(len(y)), random_state=42)
train_ix, val_ix = sklearn.cross_validation.train_test_split(train_ix, random_state=42)

X_tr = X[train_ix]
y_tr = y[train_ix]

X_val = X[val_ix]
y_val = y[val_ix]

X_te = X[test_ix]
y_te = y[test_ix]
# We'll connect our output classifier to the last fully connected layer of the network
output_layer = DenseLayer(net['fc7'], num_units=len(CLASSES), nonlinearity=softmax)

# Define loss function and metrics, and get an updates dictionary
X_sym = T.tensor4()
y_sym = T.ivector()

prediction = lasagne.layers.get_output(output_layer, X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym),
                      dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.0001, momentum=0.9)

# Compile functions for training, validation and prediction
train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
val_fn = theano.function([X_sym, y_sym], [loss, acc])
pred_fn = theano.function([X_sym], prediction)

# generator splitting an iterable into chunks of maximum length N
def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == N:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# We need a fairly small batch size to fit a large network like this in GPU memory
BATCH_SIZE = 16

def train_batch():
    ix = range(len(y_tr))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return train_fn(X_tr[ix], y_tr[ix])

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])

for epoch in range(25):
    for batch in range(25):
        loss = train_batch()

    ix = range(len(y_val))
    np.random.shuffle(ix)

    loss_tot = 0.
    acc_tot = 0.
    for chunk in batches(ix, BATCH_SIZE):
        loss, acc = val_fn(X_val[chunk], y_val[chunk])
        loss_tot += loss * len(chunk)
        acc_tot += acc * len(chunk)

    loss_tot /= len(ix)
    acc_tot /= len(ix)
    print(epoch, loss_tot, acc_tot * 100)

def deprocess(im):
    im = im[::-1, :, :]
    im = np.swapaxes(np.swapaxes(im, 0, 1), 1, 2)
    im = (im - im.min())
    im = im / im.max()
    return im

p_y = pred_fn(X_val[:25]).argmax(-1)

for i in range(0, 25):
    true = y_val[i]
    pred = p_y[i]
    print str(true) + "->" + str(pred)
