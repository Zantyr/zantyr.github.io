---
layout: root
---

## Curiosity driven learning

Create a metric for building models part-by-part by selecting transforms that reduce dimensionality
while keeping maximum novelty for the classifier.

One negative result of experimentation [here](bin/Curiosity.ipynb). I genuinely think this is doable, proper approach is needed. The method must be linear in time and maximize a proper metric over a dataset.

## Invariance-based superresolution

Solving equation for "perfect" solutions. Can it be solved analytically? We need to attract inputs and take outputs as the solution for the ResNet module. And can we actually use the censored models for that?

## Domain size measurements

Domain size measurements - given hidden activations in range [0-1], how far do the inputs span?
Generally we want features to be meaningfully generated in only subspace of inputs

## Keeper cell

RNN activation layer that keeps the previous activation and is excited if there is a difference from the previous state
This would saturate to some constant (zero?) over time if the observation is constant, and have a large peak on changes.
Comparable to BatchNorm (running norm) in time

## Car computer with ASR for hands-free using of various functions

Someone for sure has this in their head. This is only for me to remember that this may be a neat sideproject.

## Alternate formulations for training of networks

Rationale: SGD is chaotic. There are lots of other optimization techniques. Why are not they used? Does SGD beat
other trainers? Linear regression is solvable analytically. Can there be an analytic expression for other models?
(probably not, due to non-linearities). Can ReLU models be calculated linearly as censored regression? (values under zero
are treated as unknown and not penalized). 