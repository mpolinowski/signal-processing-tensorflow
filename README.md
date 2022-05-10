# Signal Processing with Tensorflow

This program fits a model based on audio recordings that contain the audio signal you are looking for
and recording only containing background noise.


The program then runs through your raw recording and searches for instances of the signal you are looking for.


Prepare data/raw (mp3) and a collection of data/positives (wav, 3-5s) and data/negatives (wav, 3-5s)
recordings and run this program.


Further information in [notebook/signal-processing.ipynb](notebook/signal-processing.ipynb)
and on [Tensorflow Audio Classifier](https://mpolinowski.github.io/devnotes/2022-04-01-tensorflow-audio-classifier)


This code is based on the [Tutorial by Nicholas Renotte](https://www.youtube.com/watch?v=ZLIPkmmDJAc) and solves a Deep-Learning Signal Processing Challenge posted on [kaggle.com](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing).