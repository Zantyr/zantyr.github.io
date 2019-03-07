---
layout: root
title: Higher order layers
author: PT
tags: [old, ML, experiment]
---

What would happen if we wanted neural network to predict weight of the proper network? We can actually
do this on runtime. Calculation of weights as a Dense of inputs means that we effectively combine
each input variable with each other (and we do this twice). Addition of activation function may change
this to a more elaborate, non-linear relationship. 

Initially I've considered `ReLU`, but the weights would be non-negative, which is not a good thing. Nearly 
equally simple method without this flaw is usage of `LeakyReLU`. I also consider `tanh`, which is what
I'll use in tests here (for convenience).

One problem I've run implementing such a layer is that I had to use two different matrix multiplication
operators - since I don't want to run into problems with splitting the batch into separate multiplications,
I had to use `tf.matmul` directly. This of course limits the portability of the layer.

```
class HigherOrderDense(keras.layers.Dense):
    def __init__(self, units, weight_activation='tanh', 
            weight_bias_regularizer=None, weight_bias_initializer='zeros',
            weight_bias_constraint=None,**kwargs):
        self.weight_activation = keras.activations.get(weight_activation)
        self.weight_bias_initializer = keras.initializers.get(weight_bias_initializer)
        self.weight_bias_regularizer = keras.regularizers.get(weight_bias_regularizer)
        self.weight_bias_constraint = keras.constraints.get(weight_bias_constraint)
        super(HigherOrderDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.weight_bias = self.add_weight(
              shape=(input_dim, self.units),
              initializer=self.weight_bias_initializer,
              name='weight_bias',
              regularizer=self.weight_bias_regularizer,
              constraint=self.weight_bias_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = keras.engine.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        # kernel must have shape (in, in, out)
        # kernel(input) -> weights
        # weights must have shape (example, in, out)
        weights = K.dot(inputs, self.kernel) # dimension
        weights = K.bias_add(weights, self.weight_bias)
        if self.weight_activation is not None:
            weights = self.weight_activation(weights)
        output = tf.matmul(K.expand_dims(inputs, 1), weights)
        output = K.batch_flatten(output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {"weight_activation": self.weight_activation}
        base_config = super(HigherOrderDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

### Efficiency - how to improve

Adding another layer of complexity to the model increases the number of weights to be trained by
the number of input weights. This of course bears very high toll on the model size.

We could try to decouple those operations and try to treat the weights as a product of two
vectors of relative weights for each output. This modification:

```
class ModifiedHigherOrderDense(keras.layers.Dense):
    def __init__(self, units, weight_activation='tanh', 
            weight_bias_regularizer=None, weight_bias_initializer='zeros',
            weight_bias_constraint=None,**kwargs):
        self.weight_activation = keras.activations.get(weight_activation)
        self.weight_bias_initializer = keras.initializers.get(weight_bias_initializer)
        self.weight_bias_regularizer = keras.regularizers.get(weight_bias_regularizer)
        self.weight_bias_constraint = keras.constraints.get(weight_bias_constraint)
        super(ModifiedHigherOrderDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]
        self.kernel_x = self.add_weight(shape=(self.units, 1, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel_x',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_y = self.add_weight(shape=(self.units, input_dim, 1),
                                      initializer=self.kernel_initializer,
                                      name='kernel_y',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.weight_bias = self.add_weight(
              shape=(input_dim, self.units),
              initializer=self.weight_bias_initializer,
              name='weight_bias',
              regularizer=self.weight_bias_regularizer,
              constraint=self.weight_bias_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = keras.engine.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.kernel = K.transpose(tf.matmul(self.kernel_y, self.kernel_x)) # 10, 5, 1 * 10, 1, 5 -> 10, 5, 5
        self.built = True

    def call(self, inputs):
        # kernel must have shape (in, in, out)
        # kernel(input) -> weights
        # weights must have shape (example, in, out)
        weights = K.dot(inputs, self.kernel) # dimension
        weights = K.bias_add(weights, self.weight_bias)
        if self.weight_activation is not None:
            weights = self.weight_activation(weights)
        output = tf.matmul(K.expand_dims(inputs, 1), weights)
        output = K.batch_flatten(output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {"weight_activation": self.weight_activation}
        base_config = super(ModifiedHigherOrderDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

### Related work

As I've built the concept I've decided to jump into Google Research and check whether something
 similar exists. [This rather old paper][1] presents several layers that analyze products of inputs. Sigma-Pi does just that:
 takes as input a polynomial of form (1). (does it also use gates of all inputs?)
Pi-Sigma unit is a Dense layer followed by multiplication of outputs (with weights of 1).

Eq 1: $TEX(W(x_i) = \sum_i{w_i x_i + \sum_j{w_ij x_i x_j}})$

Functional link networks are similarly an artificial extension of a training vector by adding
additional variables that are simple numerical transformations of variables - it is used to provide
network the ability to deal with non-linear transformations without actually learning them.
FLM highly enhance dimensionality of the data without adding new information, but it is a useful
step in processing that is not widely used nowadays. Is it viable?

Product units is something that uses multiplication gates - it is basically Dense on logarithms
 - multiplication and summation are replaced with exponentiation and multiplication respectively.
 Easiest way to implement them is indeed using logarithms, however negative values cannot be logged.
The problem with PUs is also in the fact, that the weights of such unit must in general be of very
specific values - initially close to optimal values and cannot stray away too much to prevent blowing
up. Also - not differentiable at zeros
 
[This publication by Li et al.][2] shows modified sigma-pi-sigma networks. At pi layer, predefined monomials are computed from 
outputs of sigma layer. MSPSNNs have a training procedure that prunes monomials that have little weights
and only remaining architecture is tuned to the problem.

### Implementing SPS-like layer

Sigma-Pi-Sigma layer actually depends only on Pi element. However, the Pi element has multiplicative
weights equal to either 1 or 0, combinations of which should be discovered in training process. I've
decided to optimize the weights directly.

The multiplicative Dense layer in simple terms should look like:

```
class MultiplicativeDense(keras.layers.Dense):
    def call(self, inputs):
        output = K.repeat(inputs, self.units)
        output = output ** (2 * K.sigmoid(K.permute_dimensions(self.kernel, (1,0))))
        output = K.prod(output, 2)
        if self.use_bias:
            output = output * self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

```

Sigmoid is there to prevent from reaching undefined zero power or explosions of powers.

### Efficiency tests

```
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Lambda
from keras.models import Model
import numpy as np

(train_X, train_y), (test_X, test_y) = cifar10.load_data()
train_X = train_X.astype(np.float32)
train_X /= 256
test_X = test_X.astype(np.float32)
test_X /= 256

def common_head():
	input = Input()
	l1 = Conv2D(32, 3, activation='relu')(input)
	l2 = Conv2D(40, 3, activation='relu')(l1)
	l3 = Conv2D(48, 3, activation='relu')(l2)
	l4 = Conv2D(56, 3, activation='relu')(l3)
	l5 = Conv2D(64, 3, activation='relu')(l4)
	l6 = Conv2D(72, 3, activation='relu')(l5)
	l7 = MaxPooling2D(2)(l6)
	l8 = Conv2D(80, 3, activation='relu')(l7)
	l9 = MaxPooling2D(2)(l8)
	l10 = Flatten()(l9)
	return input, l10

input, head = common_head()
d1 = Dense(250, activation='relu')(head)
d2 = Dense(100, activation='relu')(d1)
d3 = Dense(10, activation='softmax')(d2)
model1 = Model(input, d3)

input, head = common_head()
d1 = HigherOrderDense(250, activation='relu')(head)
d2 = HigherOrderDense(100, activation='relu')(d1)
d3 = HigherOrderDense(10, activation='softmax')(d2)
model2 = Model(input, d3)

input, head = common_head()
d1 = ModifiedHigherOrderDense(250, activation='relu')(head)
d2 = ModifiedHigherOrderDense(100, activation='relu')(d1)
d3 = ModifiedHigherOrderDense(10, activation='softmax')(d2)
model3 = Model(input, d3)

input, head = common_head()
d1 = Dense(250, activation='relu')(head)
d2 = Lambda(lambda x: x / K.mean(x))(d1)
d3 = MultiplicativeDense(150, activation='relu')(d2)
d4 = Dense(10, activation='softmax')(d3)
model4 = Model(input, d4)
model4.compile('adam', 'categorical_crossentropy')
model4.summary()

model1.fit(train_X, to_categorical(train_y), epochs=10)
model2.fit(train_X, to_categorical(train_y), epochs=10)
model3.fit(train_X, to_categorical(train_y), epochs=10)
model4.fit(train_X, to_categorical(train_y), epochs=10)
```

I've tested the layers in such a setup. Classic higher order dense had several hundred millions of
parameters, which rendered it untrainable. Modified-layered network ran extremely slow and 
shown no signs of meaningful convergence in first 10 minutes. Due to that, I think these layers
are unusable for any meaningful number of inputs, like in computed vision. SPS behaviour was either
getting near-zero updates in training or getting NaNs in the first iteration, depending on activation
function applied to weights. This may signal gradient explosion/vanishing problem, as the power
operation produces really high outputs leading to very high errors. Even the normalizing layer
added to the model doesn't help (tested with and without the mean).

### Summary

The lessons learnt is that the automatic derivatives ain't magic and we cannot optimize arbitrary
functions. There is so much hustle in getting error values even for polynomials, that despite
being pleasant and appealing, the nonlinear combinations of elements cannot be very easily implemented.

Both linked papers show that product networks are hard to optimize if the optimal values are far from
optimal. Also, in PS or SPS the combinations of Pi network aren't really optimized and must be mined
in some other way... (need to figure out how...)
 
[1] - https://repository.up.ac.za/bitstream/handle/2263/29715/03chapter3.pdf?sequence=4
[2] - https://arxiv.org/abs/1802.00123