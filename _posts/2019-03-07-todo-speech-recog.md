---
Title: What is to be done in speech recognition?
Tags: [ML, speech]
---

What is to be done in speech recognition? The field is perceived as solved as there are practical
implementations of ASRs that can be used in human machine interfaces. Is there anything to be done?

Well, probably yes, as in every other ML domain. While benchmarks are very high, with machines
besting human listeners (I suggest following [Wer are we GitHub repository][https://github.com/syhw/wer_are_we]),
there are still issues that are generally plaguing other fields that rely on deep networks.

## Issues that are inherent to deep learning

1. Over-reliance on data and compute
2. Artifacts
3. Noise
4. Context

Name of the game in deep learning is scale. Scale in data and scale in compute. The techniques we use are dumber
than several years ago and the scale and massive paralellisation simply happens to trump everything we conceived.
Since we need to learn basic facts about the domain we are using the models are very complex. Probably unnecessarily complex.
Due to that, current models are not invariant to compression and meaningless signal parts are interpreted as speech,
as our models are just a pile of linear algebra. The same goes with noise - plague of every ML domain.
While predictable (stationary) noise is OK, rare events are hard to remove by preprocessing.
Context - also common. Requires embedding of the understanding of the world into the machine. At the moment
the models are at the level of the very sloppy and the lazy human. Linking facts and adding understanding
would probably require things that will create an AGI.

## Issues that are speech specific

1. Homophones
2. Accents
3. Unclean speech
4. Relatively high latency
5. Learning new words
6. Diarisation, speaker separation
7. Far-field recognition
8. Integration with language models

Homophones are problematic due to the need for semantic modelling of what we say. This could be easily
solved by better understanding of language and context. Accents and unclean speech is related to 
inherent variability in pronuncation every one of us has. Many languages, especially those present
in many continent have multitude of different vocabularies, accents and pronunciations despite
being classified as a single language. Latency is related to serial nature of signal and the fact
that while the model may be wide and parallelizable, the realtime may only look back in the past
and is usually built using RNNs, which are noticeably slower than CNNs. Especially CTC, in my experience,
seems to be a slow process. 

New terms, onomatopeia, proper names - we cannot possibly enumerate each and every name in existence and
there are quickly new ones. Language models have to take that into account. This could require good
understanding on how speech is produced (as humans do), which could be also used in separation of speakers
from others and other sources - I think speech processing models should learn speech invariants.
Far-field recognition is related to integration of existing speech enhancement models and achieving
good robustness to various conditions (especially to noises not learnt beforehand).

Integration with language models - this is probably something that will be far future of integrated
speech processing technologies. I cannot really tell how will it be done, but this seems to be the solution
for many of the problems related to context and speech.

The sources list several other interesting problems: low-resource, multilinguage, zero annotation trainings...
I think they are not necessary to ordinary speech-based HMI.

## Conclusion

We are not done yet. I simply wondered whether after AI revolution there is something to tinker with and still
be original. Hopefully yes. Even the domains in "traditional" computing are still being moved, even if the 
common practices were established years ago. This brings some joy to the ML fields, which looks like they're
dominated by effects of scale.

## Sources

1. [this][https://www.quora.com/How-does-speech-recognition-work-What-advances-in-software-hardware-need-to-be-made-to-improve-it-or-is-it-just-a-matter-of-building-up-a-larger-database]
2. [this][https://awni.github.io/speech-recognition/]
3. [this][http://lig-membres.imag.fr/blanchon/SitesEns/NLSP/resources/ASR2018.pdf]