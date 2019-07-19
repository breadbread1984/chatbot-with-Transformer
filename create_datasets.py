#!/usr/bin/python3

import os;
import re;
import tensorflow as tf;
import tensorflow_datasets as tfds;

# Maximum number of samples to preprocess
MAX_SAMPLES = 50000;
# Maximum sentence length
MAX_LENGTH = 40;

def create_datasets():
    
    # 1) download the datasets and unzip it.
    path_to_zip = tf.keras.utils.get_file('cornell_movie_dialogs.zip', origin = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip', extract = True);
    # get the unzipped files' path.
    path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus");
    path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt');
    path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt');

    # 2) load questions and answers from the datasets in list of string.
    questions, answers = load_conversations(path_to_movie_lines, path_to_movie_conversations);
    
    # 3) Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size = 2**13);
    # Define start and end token to indicate the start and end of a sentence
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1];
    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2;
    # tokenize the datasets.
    questions, answers = tokenize_and_filter(questions, answers, start_token, end_token, tokenizer);
    
    print('Vocab size: {}'.format(VOCAB_SIZE));
    print('Number of samples: {}'.format(len(questions)));
    
    # 4) convert the datasets to tf.data.Dataset format
    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ));
    return dataset, tokenizer;

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence

def load_conversations(path_to_movie_lines, path_to_movie_conversations):
    # dictionary of line id to text
    id2line = {}
    with open(path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs

# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs, start_token, end_token, tokenizer):
    tokenized_inputs, tokenized_outputs = [], []
    
    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = start_token + tokenizer.encode(sentence1) + end_token
        sentence2 = start_token + tokenizer.encode(sentence2) + end_token
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
    
    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
    
    return tokenized_inputs, tokenized_outputs

if __name__ == "__main__":

    tf.executing_eagerly();
    create_datasets();
