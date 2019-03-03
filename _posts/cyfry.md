---
layout: default
title: Recognition of digit string in Keras
author: PT
---

# Recognition of digit string in Keras

Hi! Recently I'm struggling with integrating the Plain Old(tm) WFST with CTC predictions, I wanted
to deliver a simpler project in the meantime - that is, to recognize spoken digits in Polish. Solutions
to the problem were conceived somewhere in the ancient era, but I'll use dead simple code for Keras.

The greatest problem with ASR is the fact, that the sequences have varying length. That's not good
for fixed size architectures like ANN, even in RNN, where the time dimension may vary - those 
architectures output single output for each timestep - which obviously isn't the case in speech:
phonemes undergo state changes, their boundaries blur and speech rate varies. A method of 
contracting the multiple outputs to one coherent sequence of symbols is needed.

### CTC 

CTC is a loss function, that's defined in a non-straightforward way. It's based on sum of probabilities
that given label sequence can be extracted from output of the network. Since we have multiple possible
combinations of symbols, that can merge to a given labelling, an algorithm is used ...

### Data

I've downloaded a [small corpus from Clarin project][1]. There are already chosen train/dev/test sets,
and I shamelessly import all the files from respective folders. Usually, MFCC is used as input features
for speech recognition and I comply to that custom. 

```
from keras.models import Model
from keras.layers import GRU, Conv1D, Dropout, LeakyReLU, Dense, Input, Lambda
from keras.optimizers import Adam

import editdistance
import keras
import librosa
import numpy as np
import os

DIGITSPATH = os.path.expanduser("~/Downloads/cyfry/digits_train")
VALIDDATA = os.path.expanduser("~/Downloads/cyfry/digits_valid")
NPHONES = 10
NFEATS = 39

def load_data(path):
    X, y = [], []
    for fname in (os.path.join(path, x[:-4]) for x in os.listdir(path)
            if x.endswith('.raw')):
        print(fname)
        with open(fname + '.raw') as f:
            recording = np.fromfile(f, dtype=np.int16)
            recording = extract_features(recording)
        with open(fname + '.txt') as f:
        	transcript = [x for x in f.read() if x in '0123456789']
        X.append(recording.T)
        y.append(np.array([int(x) for x in transcript]))
    return X, y

def extract_features(x):
    x = x.astype(np.float32)
    x /= 2**15
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    ddelta = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, ddelta])
```

I went the lazy way and used fairly well-developed `librosa` package. Alongside with `pyAudioAnalysis`
those are my choices to go when it comes to advanced audio processing. The downside is our lack of
fine-grained control of the window size, but for now, it's *k*. Delta features are estimated
derivatives of said quantities and are usually used alongside MFCC for better accuracy.

### Model

I use CNN-RNN approach - I apply one-dimensional convolutions in time to the signal and use the convolved
features as sequence input of GRU layers. I use convolutions with padding preserving the shape, to ensure
all sequences fit, but probably this is an overcaution - didn't try with diminishing time dimension though.
LeakyReLU and Dropuout sprinkled in, the code looks like this:

```
def mk_model(max_label_length):
    feature_input = Input(shape = (None, NFEATS))
    layer_1 = Conv1D(48, 7, padding = 'same')(feature_input)
    layer_2 = LeakyReLU(0.01)(layer_1)
    layer_3 = Dropout(0.25)(layer_2)
    layer_4 = Conv1D(64, 5, padding = 'same')(layer_3)
    layer_5 = LeakyReLU(0.01)(layer_4)
    layer_6 = Dropout(0.25)(layer_5)
    layer_7 = Conv1D(96, 3, padding = 'same')(layer_6)
    layer_8 = LeakyReLU(0.01)(layer_7)
    layer_9 = Dropout(0.25)(layer_8)
    layer_10 = GRU(64, return_sequences = True)(layer_9)
    layer_11 = Dropout(0.25)(layer_10)
    layer_12 = GRU(48, return_sequences = True)(layer_11)
    layer_13 = Dropout(0.25)(layer_12)
    layer_14 = GRU(32, return_sequences = True)(layer_13)
    layer_15 = GRU(NPHONES + 1, return_sequences = True, activation = 'softmax')(layer_14)
```

Ok, I'm using plain Model instead of Sequential - why? CTC in Keras is fairly tricky subject,
because it isn't available as other loss functions a fellow researcher uses every day. CTC is rather
defined as a layer, using native Tensorflow constructs, the loss used to optimize the weights is computed
as a layer and injected directly as a loss value. The code providing a good base for the issue
is present in [Keras examples][2]. Copy-paste-adapting the code into the project:

```
def ctc_loss_function(arguments):
    y_pred, y_true, input_length, label_length = arguments
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def mk_model(max_label_length):
	...
    label_input = Input(shape = (max_label_length,))
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_lambda = Lambda(ctc_loss_function, output_shape=(1,), name='ctc')([layer_15, label_input, input_length, label_length])
    model = Model([feature_input, label_input, input_length, label_length], [loss_lambda])
    predictive = Model(feature_input, layer_15)
    return model, predictive
```

### Training and validation

Recordings have varying lengths. Keras recurrent layers can operate on input of arbitrary lengths,
but the lengths must agree to form a well-shaped tensor. I write a short function that return both
appropriate tensors as well as lengths used to train CTC model.

```
def make_data(X, y):
    X_lengths = np.array([x.shape[0] for x in X])
    maxlen = max([x.shape[0] for x in X])
    X = [np.pad(item, ((0, maxlen - item.shape[0]), (0, 0)), 'constant') for item in X]
    y_lengths = np.array([x.shape[0] for x in y])
    maxlen = max([x.shape[0] for x in y])
    y = [np.pad(item, (0, maxlen - item.shape[0]), 'constant', constant_values = NPHONES) for item in y]
    return np.stack(X), np.stack(y), X_lengths, y_lengths
```

Training procedure is rather straightforward. As I mentioned before, our loss function doesn't
really calculate a thing, pushing output of the CTC-loss producing layer back to the network. The loss
isn't modified at this layer, leading to the change solely on the network proper.

One would probably like to plot loss overtime, alongside valiation data, but the model runs for quite
a long time, so I decide to skip it for convenience (and <strike>processor</strike> GPU time)

```
def train(model, trainX, trainy, trainXl, trainyl, epochs = 500):
    # build predictive model
    optimizer = Adam(0.001)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    # encode the labels numerically and
    return model.fit([trainX, trainy, trainXl, trainyl], np.zeros(trainX.shape[0]), epochs = epochs)
```

Validation code is there to demonstrate predictions of the model

```
def validate(predictions, valid_length, groundtruth, target_length):
    predictions = keras.backend.ctc_decode(predictions, valid_length, False, 1000)
    predictions = predictions[0][0].eval(session=keras.backend.get_session())
    DERs = []
    for index in range(predictions.shape[0]):
        dist = float(editdistance.eval(
            [x for x in predictions[index, :] if x != -1],
            [x for x in groundtruth[index, :] if x != NPHONES]))
        DER = dist / target_length[index]
        DERs.append((DER, target_length[index]))
    return DERs
```

### Evaluation

Having defined all the right functions, I append the main program code:

```
if __name__=='__main__':
    data = make_data(*load_data(DIGITSPATH))
    trn, predict = mk_model(data[1].shape[1])
    train(trn, *data, epochs=500) # at 300 it makes sensible predictions
    valid_data = make_data(*load_data(VALIDDATA))
    predictions = predict.predict(valid_data[0])
    DERs = validate(predictions, valid_data[2], valid_data[1], valid_data[3])
    print sum([x[0] * x[1] for x in DERs]) / sum([x[1] for x in DERs])

```

I launch the whole snippet as a script via `%load` IPython magic. This way I may inspect the
objects created in the experiment before I decide to throw it out the window. Moreover, this
allows me to further tune the network, if results are deemed unsatisfactory. After 750 epochs I get:

```
0.124637681159
```

I admit, this is quite high WER for that simple dataset. Part of the reason may be that I 
completely ignored data about phonemes, part because I migth be over- or underfit.
YMMV, especially due to the fact, that for a long time the network is grinding in place with the loss.
Fortunately, I attained a model that's maybe not pretty accurate, but at least demonstrates
correctness of the proof of concept. The interesting bits come out, when you look at the actual output
of the network (without decoding) in comparison to the groundtruth:

```
In [15]: m.predict(vd[0]).argmax(axis=2)[0]
Out[15]: 
array([10, 10, 10, 10, 10, 10, 10, 10,  8,  8,  8,  8, 10, 10, 10, 10, 10,
       10, 10,  4,  4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        8,  8,  8,  8, 10, 10, 10, 10, 10, 10, 10, 10, 10,  5,  5,  5, 10,
       10, 10, 10, 10, 10,  1,  1,  1,  1, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10,  8,  8,  8,  8, 10, 10, 10, 10, 10, 10, 10, 10,  0,  0,  0,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  9,  9,  9, 10, 10,
       10, 10, 10, 10,  7,  7,  7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        7,  7,  7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  9,
        9,  9, 10, 10, 10, 10, 10,  3,  3,  3, 10, 10, 10, 10, 10, 10, 10,
       10,  7,  7,  7,  7,  7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  3,
        3,  3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  8,  8,  8, 10, 10,
       10, 10, 10, 10, 10, 10,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10,  2,
       10, 10, 10, 10, 10, 10,  7,  7, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10,  5,
        5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

In [16]: vd[1][0]
Out[16]: array([8, 4, 8, 5, 1, 8, 0, 9, 7, 7, 9, 3, 7, 3, 8, 0, 2, 7, 9, 5])
```

As you may see, this example was solved almost correctly. The output of validate function
shows us which cases have weak WER - it turns out that shorter examples are more prone to errors,
probably due to the difference in the length of examples.

```
In [22]: validate(m.predict(vd[0]), vd[2], vd[1], vd[3])
Out[22]: 
[(0.05, 20),
 (0.05, 20),
 (0.2, 5),
 (0.0, 5),
 (0.15, 20),
 (0.2, 5),
 (0.2, 5),
 (0.0, 20),
 (0.1, 20),
 (0.4, 5),
 (0.2, 5),
 (0.2, 5),
 (0.2, 5),
 (0.0, 5),
 (0.0, 5),
 (0.0, 5),
 (0.0, 5),
 (0.2, 20),
 (0.2, 5),
 (0.2, 5),
 (0.2, 5),
 (0.0, 20),
 (0.4, 5),
 (0.0, 5),
 (0.2, 5),
 (0.0, 5),
 (0.4, 5),
 (0.1, 20),
 (0.2, 5),
 (0.6, 5),
 (0.15, 20),
 (0.0, 5),
 (0.1, 20),
 (0.0, 5),
 (0.0, 5),
 (0.0, 5),
 (0.2, 5),
 (0.2, 5),
 (0.6, 5)]

```

Another important thing may be that the features are not normalized, which causes both longer training
time and higher sensitivity to some dimensions. Enough said, our prototype work quite well. I wonder
how much can be achieved with such a simple model.

[1]: https://clarin-pl.eu/dspace/handle/11321/317
[2]: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py