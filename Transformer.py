#!/usr/bin/python3

import tensorflow as tf;

def Attention(query_shape, key_shape, value_shape, mask_shape):

    # the inputs tensor.shape = (batch, num_heads, seq_length, depth)
    tf.debugging.Assert(tf.equal(tf.shape(query_shape)[-1], 3), [query_shape]);
    tf.debugging.Assert(tf.equal(tf.shape(key_shape)[-1], 3), [key_shape]);
    tf.debugging.Assert(tf.equal(tf.shape(value_shape)[-1], 3), [value_shape]);
    tf.debugging.Assert(tf.equal(tf.shape(mask_shape)[-1], 3), [mask_shape]);
    # inputs
    query = tf.keras.Input(query_shape);
    key = tf.keras.Input(key_shape);
    value = tf.keras.Input(value_shape);
    mask = tf.keras.Input(mask_shape);
    # normalized outer product of query and key.
    # logits.shape = (batch, num_heads, seq_length, seq_length)
    qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([query, key]);
    depth = tf.keras.layers.Lambda(lambda x: tf.cast(tf.shape(x)[-1], dtype = tf.float32))(key);
    logits = tf.keras.layers.Lambda(lambda x: x[0] / tf.math.sqrt(x[1]))([qk, depth]);
    weighted_mask = tf.keras.layers.Lambda(lambda x: x * -1e9)(mask);
    logits = tf.keras.layers.Add()([logits, weighted_mask]);
    # attention.shape = (batch, num_heads, seq_length, seq_length)
    attention = tf.keras.layers.Softmax()(logits);
    # attended value
    # results.shape = (batch, num_heads, seq_length, depth)
    results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, value]);
    return tf.keras.Model(inputs = (query, key, value, mask), outputs = results);

def MultiHeadAttention(seq_length, query_dim, key_dim, value_dim, d_model, num_heads):
    
    # query.shape = (batch, seq_length, query_dim)
    # key.shape = (batch, seq_length, key_dim)
    # value.shape = (batch, seq_length, value_dim)
    assert type(seq_length) is int;
    assert type(query_dim) is int;
    assert type(key_dim) is int;
    assert type(value_dim) is int;
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    query = tf.keras.Input((seq_length,query_dim));
    key = tf.keras.Input((seq_length,key_dim));
    value = tf.keras.Input((seq_length,value_dim));
    mask = tf.keras.Input((1,1,seq_length));
    # dense.shape = (batch, seq_length, d_model)
    query_dense = tf.keras.layers.Dense(units = d_model)(query);
    key_dense = tf.keras.layers.Dense(units = d_model)(key);
    value_dense = tf.keras.layers.Dense(units = d_model)(value);
    # splitted.shape = (batch, num_heads, seq_length, depth)
    query_splitted = tf.keras.layers.Reshape((seq_length, num_heads, d_model // num_heads))(query_dense);
    query_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(query_splitted);
    key_splitted = tf.keras.layers.Reshape((seq_length, num_heads, d_model // num_heads))(key_dense);
    ker_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(key_splitted);
    value_splitted = tf.keras.layers.Reshape((seq_length, num_heads, d_model // num_heads))(value_dense);
    value_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(value_splitted);
    # attention.shape = (batch, seq_length, num_heads, depth)
    attended = Attention(query_splitted.shape[1:], key_splitted.shape[1:], value_splitted.shape[1:], mask.shape[1:])([query_splitted, key_splitted, value_splitted, mask]);
    attended = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(attended);
    # concated.shape = (batch, seq_length, d_model)
    concated = tf.keras.layers.Reshape((seq_length, d_model))(attended);
    # results.shape = (batch, seq_length, d_model)
    results = tf.keras.layers.Dense(units = d_model)(concated);
    return tf.keras.Model(inputs = (query, key, value, mask), outputs = results);

def PositionalEncoding(input_shape):
    
    tf.debugging.Assert(tf.equal(tf.shape(input_shape), 2), [input_shape]);
    seq_length_max = tf.cast(input_shape[0], dtype = tf.float32);
    d_model = tf.cast(input_shape[1], dtype = tf.float32);
    inputs = tf.keras.Input(input_shape);
    # positions.shape = (seq_length_max, 1)
    positions = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.range(x, dtype = tf.float32),1))(seq_length_max);
    # j.shape = (1, d_model)
    j = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.range(x, dtype = tf.float32),0))(d_model);
    # j (which is 2 * i if j is even or 2 * i + 1 if j is odd) is the index in d_model
    # i.shape = (1, d_model)
    i = tf.keras.layers.Lambda(lambda x: tf.math.round(tf.math.divide(x,2)))(j);
    # power = 2 * i / d_model
    # power.shape = (1, d_model)
    power = tf.keras.layers.Lambda(lambda x: 2 * x[0] / tf.cast(x[1], dtype = tf.float32))((i, d_model));
    # angle = position / (10000^power)
    # angle.shape = (seq_length_max, d_model)
    angles = tf.keras.layers.Lambda(lambda x: x[0] / tf.math.pow(10000,x[1]))((positions,power));
    # sines.shape = (seq_length_max, d_model / 2)
    sines = tf.keras.layers.Lambda(lambda x: tf.math.sin(x[:,0::2]))(angles);
    # cosines.shape = (seq_length_max, d_model / 2)
    cosines = tf.keras.layers.Lambda(lambda x: tf.math.cos(x[:,1::2]))(angles);
    # pos_encoding.shape = (seq_length_max, d_model)
    pos_encoding = tf.keras.layers.Concatenate()([sines,cosines]);
    # add batch dim
    pos_encoding = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,0))(pos_encoding);
    # add pos encoding to embedding
    results = tf.keras.layers.Add()([inputs, pos_encoding]);
    return tf.keras.Model(inputs = inputs, outputs = results);

def Encoder(seq_length_max, d_model, num_heads, internal_dim, dropout_rate):
    
    inputs = tf.keras.Input((seq_length_max, d_model));

def Transformer():
    
    pass;

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    attention = MultiHeadAttention(10, 50, 30, 30, 100, 10);
    attention.save('attention.h5');
    encoding = PositionalEncoding((10,5));
