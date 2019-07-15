#!/usr/bin/python3

import tensorflow as tf;

def Attention(query_shape, key_shape, value_shape):

    # the inputs tensor.shape = (batch, num_heads, depth, 1)
    tf.debugging.Assert(tf.math.logical_and(tf.equal(tf.shape(query_shape)[-1], 3), tf.equal(query_shape[-1], 1)), [query_shape]);
    tf.debugging.Assert(tf.math.logical_and(tf.equal(tf.shape(key_shape)[-1], 3), tf.equal(key_shape[-1], 1)), [key_shape]);
    tf.debugging.Assert(tf.math.logical_and(tf.equal(tf.shape(value_shape)[-1], 3), tf.equal(key_shape[-1], 1)), [value_shape]);
    # length of key vector equals to that of value vector.
    tf.debugging.Assert(tf.equal(key_shape[-2],value_shape[-2]), [key_shape, value_shape]);
    # inputs
    query = tf.keras.Input(query_shape);
    key = tf.keras.Input(key_shape);
    value = tf.keras.Input(value_shape);
    # normalized outer product of query and key.
    # logits.shape = (batch, num_heads, query_size = depth, key_size = depth)
    logits = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True) / tf.math.sqrt(tf.cast(tf.shape(x[1])[-1], dtype = tf.float32)))([query, key]);
    # attention.shape = (batch, num_heads, query_size = depth, key_size = depth)
    attention = tf.keras.layers.Softmax()(logits);
    # attended value
    # results.shape = (batch, num_heads, query_size, 1)
    results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, value]);
    return tf.keras.Model(inputs = (query, key, value), outputs = results);

def MultiHeadAttention(query_shape, key_shape, value_shape, d_model, num_heads):
    
    tf.debugging.Assert(tf.equal(tf.shape(query_shape)[-1], 1), [query_shape]);
    tf.debugging.Assert(tf.equal(tf.shape(key_shape)[-1], 1), [key_shape]);
    tf.debugging.Assert(tf.equal(tf.shape(value_shape)[-1], 1), [value_shape]);
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    query = tf.keras.Input(query_shape);
    key = tf.keras.Input(key_shape);
    value = tf.keras.Input(value_shape);
    # dense
    query_dense = tf.keras.layers.Dense(units = d_model)(query);
    key_dense = tf.keras.layers.Dense(units = d_model)(key);
    value_dense = tf.keras.layers.Dense(units = d_model)(value);
    # split heads
    # query.shape = (batch, num_heads, depth, 1)
    query_splitted = tf.keras.layers.Reshape((num_heads, d_model // num_heads, 1))(query_dense);
    # key.shape = (batch, num_heads, depth, 1)
    key_splitted = tf.keras.layers.Reshape((num_heads, d_model // num_heads, 1))(key_dense);
    # value.shape = (batch, num_heads, depth, 1)
    value_splitted = tf.keras.layers.Reshape((num_heads, d_model // num_heads, 1))(value_dense);
    # attention.shape = (batch, num_heads, depth, 1)
    attended = Attention(query_splitted.shape[1:], key_splitted.shape[1:], value_splitted.shape[1:])([query_splitted, key_splitted, value_splitted]);
    # results.shape = (batch, d_model)
    results = tf.keras.layers.Reshape((d_model,))(attended);
    return tf.keras.Model(inputs = (query, key, value), outputs = results);

def Transformer():
    
    pass;

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    attention = MultiHeadAttention((50,),(30,),(30,),100,10);
