# Basic neural network with backpropagation in C#

Hi! Recently I've got C# on my CS course and to familiarize myself more with the topic, I wanted to do something both in line with my interests and that would let me learn more about .NET platform.
In 2018 it seems to be rather friendly programming environment, as it turns out, with a very straightforward way to start. I've decided to implement basics of backpropagation algorithm, something basic, but
nevertheless educational. Here we go:

### First steps

I've started with a basic .NET installation without IDE, so I need to start a project from CLI. Luckily, I don't need to delve into specifics of the build system to get packages installed and running 
(looking at you, Scala). I type:

```
$ mkdir Toy
$ cd Toy
$ dotnet new console
```

A nice, classic 'Hello, world!' code appears in in the folder. Trying to run it (with compilation step running smoothly under the hood) produces an error,
but quick googling tells me to instruct the dotnet which terminal type should it use:

```
$ TERM=xterm dotnet run
```

We need some matrix operation library, I quickly find [MathNet][https://numerics.mathdotnet.com/Packages.html]. Fortunately, dotnet has a pip-like tool for installing packages, the only difference is
that the library used has to be explicitly linked in the project's manifest. There's handy command to do it, which saves me from entering XML by hand:

```
$ dotnet add package MathNet.Numerics
$ dotnet add reference MathNet.Numerics
```

Voila! Everything's ready for coding.

### Dense layer

First, we're gonna build a network using Layers. For absolute simplicity, I implement only Dense layers
with optional activations after them. In simple feed-forward networks, a layer is generally a single
matrix of weights and computation is done through matrix multiplication. Activation is applied after
that operation to transform the inputs. A layer generally has to do three things:

- Compute the outputs given inputs
- Given error on the outputs, update its weights
- Given error on the outputs, backpropagate the errors to the input nodes

For our model to have some reasonable computational capabilities, two extensions must be added. First,
a bias vector which is added to the outputs. Second, the activation function allows the network to model
non-linear relationships. I consider two options: having a sigmoid activation, and none at all.

Considering that, I draft the layer interface:

```
using MathNet.Numerics;
using MathNet.Numerics.Distributions;

public class Dense {

	Matrix<double> weights;
	Vector<double> bias;
	string activation;

	public Dense(int inputSize, int outputSize, string activationChoice){
		var std = Math.Sqrt(2.0 / (inputSize + outputSize));
		var distribution = Normal(0.0, std);
		weights = Matrix<double>.Build.Random(inputSize, outputSize, distribution);
		bias = Vector<double>.Build.Dense(outputSize);
		activation = activationChoice;
	}
	
	public Vector<double> Compute(Vector<double> inputs){
		var outputs = weights.Multiply(inputs) + bias;
		if(activation == "sigmoid"){
			outputs = 1 / (1 + outputs.Exp());
		}
		return outputs;
	}

	private void Update(...){
		...
	}

	public Vector<double> Backpropagate(...){
		...		
	}
}
```

### Backpropagation algorithm

Backpropagation algorithm is described in depth [there][http://neuralnetworksanddeeplearning.com/chap2.html].
Generally, you need to consider the four fundamental equations of backpropagation. I've modified the variables
used there a little bit, so I don't need to store separate outputs and activations of the network and to disconnect
layers from each other. Thus, the Backpropagate function:

```
	public Vector<double> Backpropagate(Vector<double> inputs, Vector<double> outputs, Vector<double> errors){
		Vector<double> scaledErrors;
		if(activation == "sigmoid"){
			scaledErrors = errors.PointwiseMultiply(1 / (outputs - outputs.PointwisePower(2)));			
		} else {
			scaledErrors = errors;
		}
		var weightUpdates = scaledErrors.ToColumnMatrix().Multiply(inputs.ToRowMatrix()); // changed order here
		var biasUpdates = scaledErrors;
		Update(weightUpdates, biasUpdates);
		var backpropagatable = weights.Transpose().Multiply(scaledErrors);
		return backpropagatable;
	}
```

### Weights update

Neural networks are trained by changing their weights by a small value. There is a bunch of algorithms
that map updates to speed up the training. Plain SGD multiplies the updates by a constant learning rate.
In this article, I'll opt for momentum method (link..). For this, I'll need a speed matrix and vector 
to keep track of update velocities. I clip momenta to some small value to prevent it from going wild.

```
	...
	Matrix<double> weightMomentum;
	Vector<double> biasMomentum;
	double learningRate;
	...

	private void Update(Matrix<double> weightUpdates, Vector<double> biasUpdates){
		weightMomentum += learningRate * weightUpdates;
		biasMomentum += learningRate * biasUpdates;
		weightMomentum *= 0.98;
		biasMomentum *= 0.98;
		var sumOfWeightMomenta = Matrix<double>.Abs(weightMomentum).RowSums().Sum();
		var sumOfBiasMomenta = Vector<double>.Abs(biasMomentum).Sum();
		if(sumOfWeightMomenta > 0.001 * inputShape * outputShape){
			weightMomentum /= sumOfWeightMomenta / (0.001 * inputShape * outputShape);
		}
		if(sumOfBiasMomenta > 0.001 * outputShape){
			biasMomentum /= sumOfBiasMomenta / (0.001 * inputShape * outputShape);
		}
		weights += weightMomentum;
		bias += biasMomentum;
	}
```

### Assembling a model

Model is an abstraction, that aggregated multiple layers and coordinates prediction and fitting process.
Due to the API of the Dense, I will skip the shape inference and assume that user manually configures layers
properly:

```
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;

public class Model {

	List<Dense> layers;

	public Model(List<Dense> denseList){
		layers = denseList;
	}

	public Vector<double> Predict(Vector<double> input){
		foreach(var layer in layers){
			input = layer.Compute(input);
		}
		return input;
	}

	public void Fit(List<Vector<double>> inputs, List<Vector<double>> outputs, int epochs){
		...
	}
}
```

Model has to be fit. Recap the Backpropagagte signature - you have to provide inputs and outputs
to the layer, as well as the error. The error is produced by the latter layers during backward pass.
The network is trained fully stochastically. During each epoch, a new permutation of examples is
produced to feed data in a random order. The forward pass computes outputs, and backward propagates
the errors and updates the weights of the layers.

```
	public void Fit(List<Vector<double>> inputs, List<Vector<double>> outputs, int epochs){
		var random = new Random();
		for(int i = 0; i < epochs; i++){
			Console.WriteLine("{0}", i);
			// get a permutation of examples
			var permutation = Enumerable.Range(0, inputs.Count).OrderBy(x => random.Next()).ToArray();
			foreach(int index in permutation){
				var input = inputs[index];
				var layer_inputs = new List<Vector<double>>();
				var layer_outputs = new List<Vector<double>>();
				Vector<double> output;
				foreach(var layer in layers){
					layer_inputs.Add(input);
					output = layer.Compute(input);
					layer_outputs.Add(output);
					input = output;
				}
				var hypothesis = input;
				// error derivative = true - hypotheses
				var error = -(hypothesis - outputs[index]);
				for(int j = layers.Count - 1; j >= 0; j--){
					var layer = layers[j];
					var some_input = layer_inputs[j];
					var some_output = layer_outputs[j];
					error = layer.Backpropagate(some_input, some_output, error);
				}
			}
		}
	}

```

### Evaluation

Code for the model is complete - let's test it. A sample problem is to teach a network to compute
cumulative sum of a random vector. Data may be generated automatically, so there's the code:

```
using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Toy
{
    class Program
    {
        static void Main(string[] args)
        {
        	var example = new List<Vector<double>>();
			for(int i = 0; i < 1000; i++){
				example.Add(CreateVector.Random<double>(10));	
			}
			var outputs = new List<Vector<double>>();
        	for(int i = 0; i < 1000; i++) {
        	    double sum = 0;
				var next = new List<double>();
	            foreach(var item in example[i]) {
	                sum += item;
	                next.Add(sum);
	            }
	            outputs.Add(CreateVector.Dense<double>(next.ToArray()));
        	}
        	// data is made
        	var layers = new List<Dense>();
        	layers.Add(new Dense(10, 10, ""));
        	layers.Add(new Dense(10, 10, ""));
        	var model = new Model(layers);
        	model.Fit(example, outputs, 50000);
        	var data = CreateVector.Random<double>(10);
        	Console.WriteLine(data);
       	    double sum2 = 0;
			var next2 = new List<double>();
            foreach(var item in data) {
                sum2 += item;
                next2.Add(sum2);
            }
            Console.WriteLine(CreateVector.Dense<double>(next2.ToArray()));
        	Console.WriteLine(model.Predict(data));
        }
    }
}
```

Results:

```
DenseVector 10-Double
 0.732838
  -1.1569
  -1.3591
0.0204341
  1.83334
-0.546809
-0.761836
 0.465136
-0.384529
 0.174384

DenseVector 10-Double
 0.732838
-0.424067
 -1.78317
 -1.76274
 0.070604
-0.476205
 -1.23804
-0.772905
 -1.15743
-0.983049

DenseVector 10-Double
 0.732846
-0.424052
 -1.78315
 -1.76271
0.0706408
-0.476167
   -1.238
-0.772859
 -1.15738
-0.982999
```

I won't validate the model numerically - it suffices to see that the results are fairly close to
expectations. 

### Final words

Hope you enjoy the post. Full code available as .tgz archive - [here][/bin/csneural.tgz]
