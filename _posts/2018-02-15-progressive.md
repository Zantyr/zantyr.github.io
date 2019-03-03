---
layout: default
title: Progressive neural networks
author: PT
---

In this text, I present an introduction to progressive neural networks, which is an interesting multi-task architecture; I also introduce an example implementation in Keras.

Multi-task learning is fairly hot topic recently. Indeed, using multiple datasets, especially when they're prepared for different tasks, is alluring in the world of data scarcity (annotated, it is).
Additionally, there are other benefits of finding a common internal representation - increased efficiency, introducing domain knowledge. It seems to be more natural as well, as humans don't learn each
task separately, being rather bombarded by data conveying multitude of information at once. Our learning is most often based on previous experience attained on previous tasks.

Multi-task, continual and training learning have a common problem to struggle with, so called _catastrophic forgetting_. Basically, anytime you update a network that was trained on any other dataset, you
risk degrading the previously learnt representation. Model trained, for example, on ImageNet, adapted for a specific task, lose accuracy to discern original images even when original decoder is attached.
A solution to this would be to not modify the original representation, which is done by

Rather interesting approach is proposed by [Rusu et al.][1], dubbed _progressive neural networks_. It is a column-based approach: each task has a separate sequential architecture. They are trained separately
in order from most well-formed task to the hardest one. After finishing one model a next column is attached, so each layer of the new column takes as inputs all layers a lower level, both from the target task
and all previously trained. The previous columns are frozen, effectively training only the column in question, leaving other layers intact. This allows for using previous representations without modifying it.

While you may fairly well use PNNs where there are multiple tasks and the data is not readily available, the obvious drawback is that the models tend to grow larger with each task, as Nth column requires
inputs from majority, if not all, layers from previous columns. Concatenation of inputs at each layer produces fairly large weight tensors.

## Keras functional API

Most commonly, a NN model is thought of as a sequence of layers stacked each upon another. While it's true even for advanced models, skip connections as in ResNet or branching in MTL models require more
flexible architecture. Generally, Keras is capable of training any structure that can be expressed as a DAG with completely defined inputs and outputs. It is done using functional API.

This API is called functional due to the fact that a Layer object is actually a callable. Applied to a Tensor (or a list of) this callable produces another Tensor representing said layer.
A model is produced by chaining the calls to layers and then passing input and output Tensors to a Model constructor. For example, let's consider this toy sequential model:

```
from keras.models import Sequential
from import Dense

model = Sequential()
model.add(Dense(10, input_shape = (15, ), activation = 'tanh'))
model.add(Dense(10, activation = 'tanh'))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
```

The same model may be implemented in functional API as this:

```
from keras.models import Model
from import Dense, Input

input_layer = Input(shape = (15, ))
dense_1 = Dense(10, activation = 'tanh')(input_layer)
dense_2 = Dense(10, activation = 'tanh')(dense_1)
dense_3 = Dense(2, activation = 'softmax')(dense_2)
model = Model([input_layer], [dense_3])
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
```

Notice how the layers are constructed and connecter _before_ actual model is created. This may seem counterintuitive, but definition of arbitrary graph instead of linear adding is required to attempt modelling
more detailed networks.

## Progressive NN implementation

Let's construct a simple model for two different versions of CIFAR dataset. Those datasets have identical input size - which is a requirement to utilize the same structure for both tasks. I'll build a simple classifier
and fit it first. The structure was based on [this example from Keras repository][2].

```
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Input, LeakyReLU
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import rmsprop, Adam

input_layer = Input(shape = (32, 32, 3))
c_10_conv_1 = Convolution2D(32, 3)(input_layer)
c_10_lrelu_1 = LeakyReLU(0.01)(c_10_conv_1)
c_10_conv_2 = Convolution2D(32, 3)(c_10_lrelu_1)
c_10_lrelu_2 = LeakyReLU(0.01)(c_10_conv_2)
c_10_pooling_1 = MaxPooling2D(2)(c_10_lrelu_2)
c_10_dropout_1 = Dropout(0.25)(c_10_pooling_1)

c_10_conv_3 = Convolution2D(64, 3)(c_10_dropout_1)
c_10_lrelu_3 = LeakyReLU(0.01)(c_10_conv_3)
c_10_conv_4 = Convolution2D(64, 3)(c_10_lrelu_3)
c_10_lrelu_4 = LeakyReLU(0.01)(c_10_conv_4)
c_10_pooling_2 = MaxPooling2D(2)(c_10_lrelu_4)
c_10_dropout_2 = Dropout(0.25)(c_10_pooling_2)

c_10_flatten = Flatten()(c_10_dropout_2)
c_10_dense_1 = Dense(512)(c_10_flatten)
c_10_lrelu_5 = LeakyReLU(0.01)(c_10_dense_1)
c_10_dropout_3 = Dropout(0.5)(c_10_lrelu_5)
c_10_dense_2 = Dense(10, activation='softmax')(c_10_dropout_3)
c_10_model = Model([input_layer], [c_10_dense_2])

optimizer = Adam(0.0001, decay = 1e-6)
c_10_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['acc'])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
hist1 = c_10_model.fit(x_train, to_categorical(y_train), epochs = 50,
              validation_data=(x_test, to_categorical(y_test)))
```

The history objects are useful to monitor the metrics on both validation and training sets. Putting
them to matplotlib may show us how performant are our models over time.

![Losses on cifar-10](progressive-1.png)

We could really try to squeeze out a little bit more from this model, but for this article I won't
try to get the best accuracies possible. Losses look sane.

Now, let's build a more complex model. This time we'll use CIFAR-100 dataset. Let's freeze all layers in the previous column and link a new hierarchy of layers.

```
for x in c_10_model.layers:
    x.trainable = False

from keras.layers import concatenate
from keras.datasets import cifar100

c_100_conv_1 = Convolution2D(32, 3)(input_layer)
c_100_lrelu_1 = LeakyReLU(0.01)(c_100_conv_1)
c_100_conv_2 = Convolution2D(32, 3)(concatenate([c_10_lrelu_1, c_100_lrelu_1]))
c_100_lrelu_2 = LeakyReLU(0.01)(c_100_conv_2)
c_100_pooling_1 = MaxPooling2D(2)(concatenate([c_10_lrelu_2, c_100_lrelu_2]))
c_100_dropout_1 = Dropout(0.25)(c_100_pooling_1)

c_100_conv_3 = Convolution2D(64, 3)(concatenate([c_10_dropout_1, c_100_dropout_1]))
c_100_lrelu_3 = LeakyReLU(0.01)(c_100_conv_3)
c_100_conv_4 = Convolution2D(64, 3)(concatenate([c_10_lrelu_3, c_100_lrelu_3]))
c_100_lrelu_4 = LeakyReLU(0.01)(c_100_conv_4)
c_100_pooling_2 = MaxPooling2D(2)(concatenate([c_10_lrelu_4, c_100_lrelu_4]))
c_100_dropout_2 = Dropout(0.25)(c_100_pooling_2)

c_100_flatten = Flatten()(concatenate([c_10_dropout_2, c_100_dropout_2]))
c_100_dense_1 = Dense(512)(c_100_flatten)
c_100_lrelu_5 = LeakyReLU(0.01)(c_100_dense_1)
c_100_dropout_3 = Dropout(0.5)(c_100_lrelu_5)
c_100_dense_2 = Dense(100, activation='softmax')(c_100_dropout_3)
c_100_model = Model([input_layer], [c_100_dense_2])

c_100_model.summary()

```

Concatenate layer is used, as generally accept single input in most cases. This layer takes two separate Tensors and produces a Tensor with appropriate shape to hold the two constituents.
Since Keras layers usually accept single Tensor as their argument, I use concatenate in every case,
where I need to connect two of the layers. Not every layer is connected tho - I connect activation
and dropout layers as they're "modifiers" of the previous layers. Not every functionality may be
appended to layer in Keras and some of them are just separate Layer-like objects.

The model is definitely more complex, as summary() indicates:

```
__________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to           
==========================================================================================
input_1 (InputLayer)             (None, 32, 32, 3)     0                                  
__________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 30, 30, 32)    896         input_1[0][0]          
__________________________________________________________________________________________
conv2d_5 (Conv2D)                (None, 30, 30, 32)    896         input_1[0][0]          
__________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)        (None, 30, 30, 32)    0           conv2d_1[0][0]         
__________________________________________________________________________________________
leaky_re_lu_6 (LeakyReLU)        (None, 30, 30, 32)    0           conv2d_5[0][0]         
__________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 30, 30, 64)    0           leaky_re_lu_1[0][0]    
                                                                   leaky_re_lu_6[0][0]    
__________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 28, 28, 32)    9248        leaky_re_lu_1[0][0]    
__________________________________________________________________________________________
conv2d_6 (Conv2D)                (None, 28, 28, 32)    18464       concatenate_1[0][0]    
__________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)        (None, 28, 28, 32)    0           conv2d_2[0][0]         
__________________________________________________________________________________________
leaky_re_lu_7 (LeakyReLU)        (None, 28, 28, 32)    0           conv2d_6[0][0]         
__________________________________________________________________________________________
concatenate_2 (Concatenate)      (None, 28, 28, 64)    0           leaky_re_lu_2[0][0]    
                                                                   leaky_re_lu_7[0][0]    
__________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 14, 14, 32)    0           leaky_re_lu_2[0][0]    
__________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)   (None, 14, 14, 64)    0           concatenate_2[0][0]    
__________________________________________________________________________________________
dropout_1 (Dropout)              (None, 14, 14, 32)    0           max_pooling2d_1[0][0]  
__________________________________________________________________________________________
dropout_4 (Dropout)              (None, 14, 14, 64)    0           max_pooling2d_3[0][0]  
__________________________________________________________________________________________
concatenate_3 (Concatenate)      (None, 14, 14, 96)    0           dropout_1[0][0]        
                                                                   dropout_4[0][0]        
__________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 12, 12, 64)    18496       dropout_1[0][0]        
__________________________________________________________________________________________
conv2d_7 (Conv2D)                (None, 12, 12, 64)    55360       concatenate_3[0][0]    
__________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)        (None, 12, 12, 64)    0           conv2d_3[0][0]         
__________________________________________________________________________________________
leaky_re_lu_8 (LeakyReLU)        (None, 12, 12, 64)    0           conv2d_7[0][0]         
__________________________________________________________________________________________
concatenate_4 (Concatenate)      (None, 12, 12, 128)   0           leaky_re_lu_3[0][0]    
                                                                   leaky_re_lu_8[0][0]    
__________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 10, 10, 64)    36928       leaky_re_lu_3[0][0]    
__________________________________________________________________________________________
conv2d_8 (Conv2D)                (None, 10, 10, 64)    73792       concatenate_4[0][0]    
__________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)        (None, 10, 10, 64)    0           conv2d_4[0][0]         
__________________________________________________________________________________________
leaky_re_lu_9 (LeakyReLU)        (None, 10, 10, 64)    0           conv2d_8[0][0]         
__________________________________________________________________________________________
concatenate_5 (Concatenate)      (None, 10, 10, 128)   0           leaky_re_lu_4[0][0]    
                                                                   leaky_re_lu_9[0][0]    
__________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)   (None, 5, 5, 64)      0           leaky_re_lu_4[0][0]    
__________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)   (None, 5, 5, 128)     0           concatenate_5[0][0]    
__________________________________________________________________________________________
dropout_2 (Dropout)              (None, 5, 5, 64)      0           max_pooling2d_2[0][0]  
__________________________________________________________________________________________
dropout_5 (Dropout)              (None, 5, 5, 128)     0           max_pooling2d_4[0][0]  
__________________________________________________________________________________________
concatenate_6 (Concatenate)      (None, 5, 5, 192)     0           dropout_2[0][0]        
                                                                   dropout_5[0][0]        
__________________________________________________________________________________________
flatten_2 (Flatten)              (None, 4800)          0           concatenate_6[0][0]    
__________________________________________________________________________________________
dense_3 (Dense)                  (None, 512)           2458112     flatten_2[0][0]        
__________________________________________________________________________________________
leaky_re_lu_10 (LeakyReLU)       (None, 512)           0           dense_3[0][0]          
__________________________________________________________________________________________
dropout_6 (Dropout)              (None, 512)           0           leaky_re_lu_10[0][0]   
__________________________________________________________________________________________
dense_4 (Dense)                  (None, 100)           51300       dropout_6[0][0]        
==========================================================================================
Total params: 2,723,492
Trainable params: 2,657,924
Non-trainable params: 65,568
__________________________________________________________________________________________

```

Notice that several layers from the former model were introduced in this one, as "untrainable" weights.
Still, this is smaller than the weights for the previous task, as I ommited the top-most layers and
focused only on low level representation. Secondly, there are far more connections, as most of
the layers are effectively doubled.  Especially Dense layer contributes immensely to the model, (which
probably should be reduced by the way.)

I'll fit it with 50 epochs to see if it is trainable and what kind of output it produces:

```
optimizer = Adam(0.0001, decay = 1e-6)
c_100_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['acc'])

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
hist2 = c_100_model.fit(x_train, to_categorical(y_train), epochs = 50,
              validation_data=(x_test, to_categorical(y_test)))
```

<!--people probably expect some output here-->

For comparison, I've build a model with similar architecture, but without connections to the previous task.
This model has definitely fewer weights

```
c2_100_conv_1 = Convolution2D(32, 3)(input_layer)
c2_100_lrelu_1 = LeakyReLU(0.01)(c2_100_conv_1)
c2_100_conv_2 = Convolution2D(32, 3)(c2_100_lrelu_1)
c2_100_lrelu_2 = LeakyReLU(0.01)(c2_100_conv_2)
c2_100_pooling_1 = MaxPooling2D(2)(c2_100_lrelu_2)
c2_100_dropout_1 = Dropout(0.25)(c2_100_pooling_1)

c2_100_conv_3 = Convolution2D(64, 3)(c2_100_dropout_1)
c2_100_lrelu_3 = LeakyReLU(0.01)(c2_100_conv_3)
c2_100_conv_4 = Convolution2D(64, 3)(c2_100_lrelu_3)
c2_100_lrelu_4 = LeakyReLU(0.01)(c2_100_conv_4)
c2_100_pooling_2 = MaxPooling2D(2)(c2_100_lrelu_4)
c2_100_dropout_2 = Dropout(0.25)(c2_100_pooling_2)

c2_100_flatten = Flatten()(c2_100_dropout_2)
c2_100_dense_1 = Dense(512)(c2_100_flatten)
c2_100_lrelu_5 = LeakyReLU(0.01)(c2_100_dense_1)
c2_100_dropout_3 = Dropout(0.5)(c2_100_lrelu_5)
c2_100_dense_2 = Dense(100, activation='softmax')(c2_100_dropout_3)
c2_100_model = Model([input_layer], [c2_100_dense_2])

c2_100_model.summary()

optimizer = Adam(0.0001, decay = 1e-6)
c2_100_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['acc'])

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
hist3 = c2_100_model.fit(x_train, to_categorical(y_train), epochs = 50,
              validation_data=(x_test, to_categorical(y_test)))
```

Below are the plots of the training and validation losses:

![Training losses of two networks](progressive-2.png)

![Validation losses of two networks](progressive-3.png)

As you can see, the progressive model learn quicker, as it doesn't need to figure out all of thefilters and learns to reuse the one
given by the prelearned network. It also maintains superior accuracy (below). What is interesting is that
the loss from the first model starts to rise quicker.

One more remark arises from accuracy plots:

![Accuracies of both models](progressive-4.png)

As you can see on the progressive model, when the validation starts to rise, we still see a rise in accuracy.
This is against an intuition that says that we should stop fitting when validation loss steadily rises.
If we stopped, we would end with a subpar model. Why this happens - I have no idea, but is an important to note this.

## Conclusion

Progressive networks indeed enable us to learn faster and have better results. This is nowhere near to
state-of-the-art accuracy, but shows the possibility of reusing existing knowledge. The bad thing is
that it comes at cost of many more connections in the network - which translates to enormous complexity,
if more tasks were linked.

Personally, I also notice that some models that seem untrainable may kickstart with other hyperparameters.
I've used networks with 4x the parameter first, without an effort. Fortunately, for easy tasks there
are ready guidelines for both the architecture and fitting setup. For well-curated dataset and familiar tasks,
the models are much more easier to construct.

[1]: https://arxiv.org/pdf/1606.04671.pdf
[2]: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
