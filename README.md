# krisPunctuator

Punctuation generator for proscript files. 

`punctuator.py` is a modification of the same named file in https://github.com/ottokart/punctuator2

## Requirements
In order to run krisPunctuator you need `Python 2.7` with the following packages installed:

- Theano
- Numpy
- cPickle

## Files to work with

Proscript files are acoustically enriched transcript files. Besides the text, it has word aligned acoustic features. You can create proscript files using https://github.com/alpoktem/Proscripter. 

Models are trained using https://github.com/alpoktem/punkProse

Sample model is trained with pauses between words and mean fundemental frequency. 

Vocabulary is compiled from TED talks.

## Sample model run:
`python punctuator.py -m <model-file> -v <vocabulary-file> -i <input-proscript> -o <predictions-output> -p -f mean.f0.id`

Features that were used for training the model should be listed after `-f`

`-p` indicates that pauses between words were used to train the model



