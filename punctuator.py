# coding: utf-8
from __future__ import division

import models as models
from optparse import OptionParser

import theano
import sys
import os
import codecs
import cPickle

import theano.tensor as T
import numpy as np

SPACE = ""
PUNCTUATION_VOCABULARY = {0:SPACE, 1:',', 2:'.', 3:'?', 4:'!', 5:'-', 6:';', 7:':'}
EOS_PUNCTUATION_CODES = [2,3,4,5,6,7]

END = "<END>"
UNK = "<UNK>"
EMP = "<EMP>"

MAX_SEQUENCE_LENGTH = 50

def checkArgument(argname, isFile=False, isDir=False):
    if not argname:
        return False
    else:
        if isFile and not os.path.isfile(argname):
            return False
        if isDir and not os.path.isdir(argname):
            return False
    return True

def iterable_to_dict(arr):
    return dict((x.strip(), i) for (i, x) in enumerate(arr))

def read_vocabulary(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f:
        return iterable_to_dict(f.readlines())

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def word_data_to_pickle(talk_data, output_pickle_file):
    with open(output_pickle_file, 'wb') as f:
        cPickle.dump(talk_data, f, cPickle.HIGHEST_PROTOCOL)

def restore_unsequenced_test_data(test_data_path, word_vocabulary, predict_function, with_pause_feature, semitone_feature_names, write_groundtruth, sequence_length, output_text=None, output_pickle=None):
    with open(test_data_path, 'rb') as f:
        test_data = cPickle.load(f)

    word_sequence = test_data['word'] + [END]
    pause_sequence = test_data['pause.id'] + [0]
    otherfeatures_sequences = [test_data[feature_name] + [0] for feature_name in semitone_feature_names]

    i = 0
    with codecs.open(output_text, 'w', 'utf-8') as f_out:
        while True:
            subsequence_words = word_sequence[i: i + sequence_length]
            subsequence_wordIds = [word_vocabulary.get(w, word_vocabulary[UNK]) for w in subsequence_words]
            #subsequence_wordIds = test_data['word.id'][i: i + sequence_length]
            
            subsequence_pauses = pause_sequence[i: i + sequence_length]

            #if write_groundtruth:
            #    subsequence_gold_reduced_puncIds = test_data['punc.red.id'][i: i + sequence_length]
            #    subsequence_gold_puncIds = test_data['punc.id'][i: i + sequence_length]
            other_subsequences = [otherfeatures_sequences[feature_index][i: i + sequence_length] for feature_index in range(len(semitone_feature_names))]

            if len(subsequence_wordIds) == 0:
                break

            if not with_pause_feature:
                y = predict_function(to_array(subsequence_wordIds))
            else:
                if len(other_subsequences) == 0:
                    y = predict_function(to_array(subsequence_wordIds), to_array(subsequence_pauses))
                if len(other_subsequences) == 1:
                    y = predict_function(to_array(subsequence_wordIds), to_array(subsequence_pauses), to_array(other_subsequences[0]))
                if len(other_subsequences) == 2:
                    y = predict_function(to_array(subsequence_wordIds), to_array(subsequence_pauses), to_array(other_subsequences[0]), to_array(other_subsequences[1]))
                if len(other_subsequences) == 3:
                    y = predict_function(to_array(subsequence_wordIds), to_array(subsequence_pauses), to_array(other_subsequences[0]), to_array(other_subsequences[1]), to_array(other_subsequences[2]))
             
            predicted_punctuation_sequence = [0] + [np.argmax(y_t.flatten()) for y_t in y]
            #print(predicted_punctuation_sequence)

            f_out.write(subsequence_words[0])

            last_eos_idx = 0
            punctuations = []
            for y_t in y:

                p_i = np.argmax(y_t.flatten())
                #punctuation = reverse_punctuation_vocabulary[p_i]
                punctuation = p_i

                punctuations.append(punctuation)

                if punctuation in EOS_PUNCTUATION_CODES:
                    last_eos_idx = len(punctuations) # we intentionally want the index of next element

            if subsequence_words[-1] == END:
                step = len(subsequence_words) - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subsequence_words) - 1

            for j in range(step):
                f_out.write(PUNCTUATION_VOCABULARY[punctuations[j]] + " ")
                if j < step - 1:
                    try: 
                        f_out.write(subsequence_words[1+j])
                    except:
                        f_out.write("<unk>")
                        print(subsequence_words[1+j])

            if subsequence_words[-1] == END:
                break

            i += step

def main(options):
    if checkArgument(options.model_file):
        model_file = options.model_file
    else:
        sys.exit("Model file path argument missing")

    if checkArgument(options.vocabulary_file):
        WORD_VOCAB_FILE = options.vocabulary_file
    else:
        sys.exit("Vocabulary file path argument missing")

    if checkArgument(options.input_proscript, isFile=True):
        TEST_FILE = options.input_proscript
    else:
        sys.exit("File to punctuate is missing!")

    if checkArgument(options.output_file):
        output_file = options.output_file
    else:
        sys.exit("Output file path argument missing")

    word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)

    print("Model: %s"%model_file)
    print("Test file: %s"%TEST_FILE)

    x = T.imatrix('x')
    p = None
    a = None
    b = None
    c = None

    if options.trained_with_pause:
        print("Punctuating with pause")
        p = T.imatrix('p')
    else:
        num_semitone_features = -1

    semitone_feature_names = options.semitone_features
    num_semitone_features = len(options.semitone_features)

    print("Semitone features (%i):"%num_semitone_features)

    if num_semitone_features == 1: 
        print("Punctuating with %s"%(options.semitone_features[0]))
        a = T.imatrix('a')
    elif num_semitone_features == 2:
        print("Punctuating with %s"%(options.semitone_features[0]))
        a = T.imatrix('a')
        print("Punctuating with %s"%(options.semitone_features[1]))
        b = T.imatrix('b')
    elif num_semitone_features == 3: 
        print("Punctuating with %s"%(options.semitone_features[0]))
        a = T.imatrix('a')
        print("Punctuating with %s"%(options.semitone_features[1]))
        b = T.imatrix('b')
        print("Punctuating with %s"%(options.semitone_features[2]))
        c = T.imatrix('c')
    elif num_semitone_features > 3:
        sys.exit("Too many features (for now)")

    print("Loading model parameters...")
    net, _ = models.load(model_file, 1, x, p, a, b, c, num_semitone_features=num_semitone_features)
    inputs = [x] + [i for i in [p,a,b,c] if not i == None]

    print("Building model...")
    predict = theano.function(inputs=inputs, outputs=net.y)

    print("Generating punctuation...")
    #restored_data = restore_sequenced_test_data(TEST_FILE, 
    #                                            predict_function=predict, 
    #                                            with_pause_feature=options.trained_with_pause, 
    #                                            semitone_feature_names=semitone_feature_names, 
    #                                            reduced_punctuation=options.reduced_punctuation)

    restore_unsequenced_test_data(TEST_FILE,
                                  word_vocabulary=word_vocabulary,
                                  predict_function=predict, 
                                  with_pause_feature=options.trained_with_pause, 
                                  semitone_feature_names=semitone_feature_names, 
                                  write_groundtruth=True,
                                  sequence_length=MAX_SEQUENCE_LENGTH,
                                  output_text=options.output_file)

    #word_data_to_pickle(restored_data, output_file)
    print("Predictions written to %s."%output_file)

if __name__ == "__main__":
    usage = "usage: %prog [-s infile] [option]"
    parser = OptionParser(usage=usage)
    parser.add_option("-m", "--model_file", dest="model_file", default=None, help="model filename", type="string")
    parser.add_option("-v", "--vocabulary_file", dest="vocabulary_file", default=None, help="vocabulary file (pickle)", type="string")
    parser.add_option("-i", "--input_proscript", dest="input_proscript", default=None, help="input proscript file (pickle)", type="string")
    parser.add_option("-o", "--output_file", dest="output_file", default=100, help="output file to write predictions", type="string")
    parser.add_option("-p", "--trained_with_pause", dest="trained_with_pause", default=False, help="flag if trained with pause", action="store_true")
    parser.add_option("-f", "--semitone_features", dest="semitone_features", default=[], help="semitone feature names", type="string", action='append')

    (options, args) = parser.parse_args()

    print(options)
    main(options)