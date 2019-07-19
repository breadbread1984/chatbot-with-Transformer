#!/usr/bin/python3

import tensorflow as tf;
from create_datasets import create_datasets;
from Transformer import Transformer;

MAX_LENGTH = 40;
BATCH_SIZE = 64;
BUFFER_SIZE = 20000;
D_MODEL = 256;
EPOCHS = 20;

def main():

    print('loading dataset...');
    dataset, tokenizer = create_datasets();
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE);    
    tokenizer.save_to_file('tokenizer');
    print('creating model...');
    model = Transformer(tokenizer.vocab_size + 2, d_model = D_MODEL);
    print('training model...');
    learning_rate = CustomSchedule(D_MODEL);
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9);
    
    log = tf.summary.create_file_writer('checkpoints');
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    for epoch in range(EPOCHS):
        for (inputs,outputs) in dataset:
            with tf.GradientTape() as tape:
                pred = model(inputs[0], inputs[1]);
                loss = loss_function(outputs, pred);
                avg_loss.update_state(loss);
            # write log
            if tf.equal(optimizer.iterations % 100, 0):
                with log.as_default():
                    tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
                    tf.summary.scalar('accuracy', accuracy(outputs, pred), step = optimizer.iterations);
                print('Step #%d Loss: %.6f' % (optimizer.iteration, avg_loss.results()));
                avg_loss.reset_states();
            grads = tape.gradient(loss, model.trainable_variables);
            optimizer.apply_gradients(zip(grads, model.trainable_variables));
        # save model once every epoch
        model.save('checkpoints/transformer_%d.h5' % optimizer.iterations);

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
    return accuracy

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
