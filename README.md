# krisPunctuator

Punctuation generator for proscript files. 

Proscript files are acoustically enriched transcript files. Besides the text, it has word aligned acoustic features. You can create proscript files using https://github.com/alpoktem/Proscripter

Models are trained using https://github.com/alpoktem/punkProse

Example model is trained with pauses between words and mean fundemental frequency. 

## Sample model run:
`python punctuator.py -m <model-file> -v <vocabulary-file> -i <input-proscript> -o <predictions-output> -p -f mean.f0.id`

Features that were used for training the model should be listed after `-f`

