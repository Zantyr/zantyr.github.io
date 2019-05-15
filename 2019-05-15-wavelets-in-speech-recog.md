---
author: Zantyr
title: Wavelet transforms in speech recognition - draft
tags: [TIL, speech processing, python, draft]
date: 2019-05-15
---

_This post requires working code and I need to figure out how to use the transform. Work in progress._

Wavelets are an alternative to TF transforms. Since the issue in itself is broad and a lot of theory
has been created around it, I will focus only on the simplest aspects of the topic, which may be a
blunt simplification. The goal is to create working prototype in minimal effort and check the
feasibility of the wavelet-based methods of feature extraction.

### What is a wavelet?

Wavelets are functions that are used to obtain components of the signal of specific characteristics.
They have the following properties:

1. $\int_{-\infty}^{\infty}\psi(t)dt = 0$ - they integrate to total zero
2. $\int_{-\infty}^{\infty}|\psi(t)|^2dt < \infty$ - their power/energy is finite
	- some variations normalize the wavelet to unit norm

Wavelet functions (usually designated by psi) are generally used in families, with several different wavelets obtained from
a single mother wavelet through means of translation and dilation:

$\psi_{a,b}(t) = \frac{1}{\sqrt{a}} \psi(\frac{t - b}{a})$

The `b` factor is generally responsible for shifting the transform in time, so for simplicity, only `a`
can be used as a "parameter", which roughly corresponds to scaling the wavelet in frequency. Actually
the continuous wavelet transform is expressed as:

$X_w(a, b) = \frac{1}{|a|^{1/2}} \int_{-\infty}^{\infty} x(t) \psi^{\ast} (\frac{t-b}{a}) dt$

Assuming constant `a` (as we analyze only a single frequency) and assuming that `b` is our position
on a time of the signal, we may describe the equation above as convolution of some parameterised
daughter wavelet with signal:

$ y(t) = \int_{-\infty}^{\infty} \psi(\tau) x(t - \tau) dt $

If causal, wavelets may be therefore described as IIR filters. Application of continuous wavelet
transform produces some kind of "spectrogram", where the bins are calculated using some subset of
IIR filters (those which fulfill conditions imposed on wavelet functions). 

In general, there is no single wavelet transform, rather each mother wavelet generates it's own
transform which slightly differs in properties. This may be analogous to how z-transform creates 
different time-frequency representation basing on the choice of z (or as we speak of continuous
domain, s-transform would be more appropriate).

### Why multiresolution?

Let's assume we want to calculate wavelet transform over signal. Of course, as long as we do not use
some esoteric computing technique, that signal is discretized and so should be the transform. First,
we should assume some cut-off over the wavelet function (coming from quantization), which will then
create finite signals used in convolution. 

Such limited wavelets will have different length using therefore different windows over the signal.
Situation is similar to that of DQT, where lower frequencies vary slowly over time and are smeared
compared to higher frequencies. This however corresponds naturally to what is found in human hearing
- temporal resolution is lower in lower frequencies as more time is needed to determine cyclicality
of the signal.

While in theory the frequency can be scaled in any way, in practice the most common division used
introduces geometric division of frequency space. (TODO: how and why is that?) However, since the 
wavelets may be scaled in anyway, one can obtain arbitrary trade-off between frequency and time resolution. 

Using convolution over the signal means that the resulting transform is defined at each time step.
To reliably use it, it should be decimated, either selecting the points at regular intervals, or using
some aggregation function. However, despite computing it for a large number of points, there are some
advantages over other transforms. Since wavelets tend to decrease to zero over time, there is no need for
windowing function similar to ones used in STFT and in general, the signal does not need to be stationary over
the whole period during which the function is applied. 

### Important wavelets

Most basic wavelet is named after Haar. This wavelet is a simple composition of step functions and has
the following formula:

$$
\psi(t) = \begin{cases}
0 & t \in (-\infty, 0) \cup [2, \infty) \\
1 & t \in [0, 1) \\
-1 &  t \in [1, 2) \\
\end{cases}
$$

Haar wavelets were used heavily as a feature extraction technique in image processing before DNN. They offered
an efficient reduction of the image to a combination of all possible rectangles on the image, therefore
reducing dimensionality of the problem several times.

Morlet wavelet has a shape of exponential function in complex domain that is scaled by Gaussian density function.
Equation of that wavelet is as follows:

$$
\psi(t) = (1 + e^{\sigma^2} - 2e^{-0.75 \sigma^2})^{-0.5} \pi ^ {\frac{-1}{4}} e^{\frac{-t^2}{2}} (e^{i \sigma t} - e^{-0.5 \sigma^2})
$$

In this equation sigma is the temporal resolution of the wavelet, usually set above 5, if the signal is stationar-ish.
In that assumption the frequency of such a wavelet is rather well-defined. 

Morlet wavelets were used as a basis for prosody, syllable prominence and speech tempo description [ilcl.hse.ru]

There are other notable functions used in wavelet analysis: Ricker wavelet or Daubechies (the original one). They 
have other specific properties, which may be noteworthy, however they are omitted for simplicity. Morlet wavelet
seems to be a common choice in speech modelling. Very similar technique is based on the so called Gammatone filter,
which is a topic for another blog post.

### Track of records

Wavelet are used in as a representation for speech prosody, as a method for long-range dependencies [ilcl.hse.ru]

One particular PhD thesis [biswas] was focused on utilization of CWT for speech recognition, showing that
wavelets are better in modelling aperiodic information (consonants and other bursts), with small deficiencies
in capturing voicing information.

Generally - wavelets can help in mitigation of pre-echo effects on transients, which means that harmonics
do not begin before the actual sound begins (which may be common in signal processing as lots of processes shift
the phase). Therefore Wavelet transforms may be useful for modelling stops and other short timed effects. (citation needed)

According to Wikipedia, the temporal resolution of those transforms is the motivation to use them in music analysis.

### How to implement such a transform

Implementation is very simple, as CWT relies on appliction of convolution of specifically designed funtion over signal.
Special function is provided to automatically scale and apply proper functions and is documented [here][scipy.cwt]

```
import numpy as np
import scipy.io.wavfile as sio
import scipy.signal as ss

# just a random recording
PATH = "Denoising/DAE-test"
sr, data = sio.read(PATH + "/24000.wav")

import matplotlib.pyplot as plt
img = ss.cwt(data, ss.morlet, (np.arange(256) + 1) * 32)  # how to take good coefficients?
plt.imshow(img, aspect='auto')
```

Unfortunately, ss.daub is not a compatible function, therefore only morlet is demonstrated.

TODO: How to construct such a function?

### Further reading

Wavelet transforms are an extensive territory to describe and as such, there is a lot that I did not cover.
Since I start using the wavelets, lots of information may be incomplete or inaccurate. There are some sources
that I plan to read and understand in the future:

https://www.biorxiv.org/content/biorxiv/early/2018/08/21/397182.full.pdf
-> argues there is better way to construct wavelet transforms...

TODO: insert that study that learn wavelets

### Sources:

http://ethesis.nitrkl.ac.in/8032/1/2016_PhD_Astik_Biswas_512EE103.pdf
[biswas]: http://ethesis.nitrkl.ac.in/8032/1/2016_PhD_Astik_Biswas_512EE103.pdf
https://en.wikipedia.org/wiki/Haar_wavelet
https://en.wikipedia.org/wiki/Morlet_wavelet
https://ilcl.hse.ru/data/2017/12/08/1161391021/Martti%20Vainio%20-%20Continuous%20wavelet%20transform%20for%20speech%20research.pdf
[ilcl.hse.ru]: https://ilcl.hse.ru/data/2017/12/08/1161391021/Martti%20Vainio%20-%20Continuous%20wavelet%20transform%20for%20speech%20research.pdf
https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.cwt.html
[scipy.cwt]: https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.cwt.html
