#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;
from create_datasets import preprocess_sentence;

class Predictor(object):

    MAX_LENGTH = 40;

    def __init__(self, model_path = 'checkpoints/transformer_13780.h5', tokenizer_prefix = 'tokenizer'):

        self.transformer = tf.keras.models.load_model(model_path, custom_objects = {'tf': tf, 'ReLU': tf.keras.layers.ReLU});
        self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_prefix);
        self.start_token, self.end_token = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1];

    def predict(self, sentence):

        # tokenize input
        sentence = self.tokenizer.encode(preprocess_sentence(sentence));
        # add batch dim to inputs
        # sentence.shape = (batch = 1, seq_length)
        sentence = tf.expand_dims(self.start_token + sentence + self.end_token, axis = 0);
        # add batch dim to dec_inputs
        # output.shape = (batch = 1, 1);
        output = tf.expand_dims(self.start_token, 0);

        for i in range(self.MAX_LENGTH):

            # predictions.shape = (batch = 1, seq_length(inputs), vocab_size)
            predictions = self.transformer([sentence, output]);
            # use the output sentence's last token
            # predictions.shape = (batch = 1, vocab_size)
            predictions = predictions[:,-1:,:];
            token_id = tf.cast(tf.argmax(predictions, axis = -1), dtype = tf.int32);
            # stop when end token is predicted
            if tf.equal(token_id, self.end_token[0]): break;
            output = tf.concat([output, token_id], axis = -1);

        output = tf.squeeze(output, axis = 0);
        # detokenize the output
        output = self.tokenizer.decode([i for i in output if i < self.tokenizer.vocab_size]);
        return output;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    predictor = Predictor();
    while True:
        ask = input('say something: ');
        reply = predictor.predict(ask);
        print(reply);

