#!/usr/bin/python3

import tensorflow as tf;

def Attention(seq_length, d_model, num_heads):

    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # inputs
    query = tf.keras.Input((num_heads, seq_length, d_model // num_heads));
    key = tf.keras.Input((num_heads, seq_length, d_model // num_heads));
    value = tf.keras.Input((num_heads, seq_length, d_model // num_heads));
    mask = tf.keras.Input((1, None, seq_length));
    # normalized outer product of query and key.
    # logits.shape = (batch, num_heads, seq_length, seq_length)
    qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([query, key]);
    depth = tf.keras.layers.Lambda(lambda x: tf.cast(tf.shape(x)[-1], dtype = tf.float32))(key);
    logits = tf.keras.layers.Lambda(lambda x: x[0] / tf.math.sqrt(x[1]))([qk, depth]);
    masked_logits = tf.keras.layers.Lambda(lambda x: x[0] + x[1] * -1e9)((logits, mask));
    # attention.shape = (batch, num_heads, seq_length, seq_length)
    attention = tf.keras.layers.Softmax()(masked_logits);
    # attended value
    # results.shape = (batch, num_heads, seq_length, depth)
    results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, value]);
    return tf.keras.Model(inputs = (query, key, value, mask), outputs = results);

def MultiHeadAttention(seq_length, d_model, num_heads):
    
    # query.shape = (batch, seq_length, d_model)
    # key.shape = (batch, seq_length, d_model)
    # value.shape = (batch, seq_length, d_model)
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    query = tf.keras.Input((seq_length,d_model));
    key = tf.keras.Input((seq_length,d_model));
    value = tf.keras.Input((seq_length,d_model));
    mask = tf.keras.Input((1, None, seq_length));
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
    attended = Attention(seq_length, d_model, num_heads)([query_splitted, key_splitted, value_splitted, mask]);
    attended = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(attended);
    # concated.shape = (batch, seq_length, d_model)
    concated = tf.keras.layers.Reshape((seq_length, d_model))(attended);
    # results.shape = (batch, seq_length, d_model)
    results = tf.keras.layers.Dense(units = d_model)(concated);
    return tf.keras.Model(inputs = (query, key, value, mask), outputs = results);

def PositionalEncoding(seq_length, d_model):
    
    inputs = tf.keras.Input((seq_length, d_model));
    # positions.shape = (seq_length, 1)
    positions = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.range(tf.cast(tf.shape(x)[1], dtype = tf.float32), dtype = tf.float32),1))(inputs);
    # j.shape = (1, d_model)
    j = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.range(tf.cast(tf.shape(x)[2], dtype = tf.float32), dtype = tf.float32),0))(inputs);
    # j (which is 2 * i if j is even or 2 * i + 1 if j is odd) is the index in d_model
    # i.shape = (1, d_model)
    i = tf.keras.layers.Lambda(lambda x: x // 2)(j);
    # power = 2 * i / d_model
    # power.shape = (1, d_model)
    power = tf.keras.layers.Lambda(lambda x: 2 * x[0] / tf.cast(tf.shape(x[1])[2], dtype = tf.float32))((i, inputs));
    # angle = position / (10000^power)
    # angle.shape = (seq_length, d_model)
    angles = tf.keras.layers.Lambda(lambda x: x[0] / tf.math.pow(10000.,x[1]))((positions,power));
    # sines.shape = (seq_length, d_model / 2)
    sines = tf.keras.layers.Lambda(lambda x: tf.math.sin(x[:,0::2]))(angles);
    # cosines.shape = (seq_length, d_model / 2)
    cosines = tf.keras.layers.Lambda(lambda x: tf.math.cos(x[:,1::2]))(angles);
    # pos_encoding.shape = (seq_length, d_model)
    pos_encoding = tf.keras.layers.Concatenate()([sines,cosines]);
    # add batch dim
    pos_encoding = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x[0],0), (tf.shape(x[1])[0], 1, 1)))((pos_encoding, inputs));
    # add pos encoding to embedding
    results = tf.keras.layers.Add()([inputs, pos_encoding]);
    return tf.keras.Model(inputs = inputs, outputs = results);

def EncoderLayer(seq_length, d_model, num_heads, code_dim, dropout_rate):
    
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # inputs
    inputs = tf.keras.Input((seq_length, d_model));
    mask = tf.keras.Input((1, 1, seq_length));
    # attended.shape = (batch, seq_length, d_model)
    attended = MultiHeadAttention(seq_length, d_model, num_heads)([inputs, inputs, inputs, mask]);
    attended = tf.keras.layers.Dropout(rate = dropout_rate)(attended);
    inputs_attended = tf.keras.layers.Add()([inputs, attended]);
    attended = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(inputs_attended);
    outputs = tf.keras.layers.Dense(units = code_dim, activation = 'relu')(attended);
    outputs = tf.keras.layers.Dense(units = d_model)(outputs);
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(outputs);
    attended_outputs = tf.keras.layers.Add()([attended, outputs]);
    outputs = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(attended_outputs);
    return tf.keras.Model(inputs = (inputs, mask), outputs = outputs);

def Encoder(vocab_size, num_layers, seq_length, d_model, num_heads, code_dim, dropout_rate):

    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # inputs
    inputs = tf.keras.Input((seq_length,));
    mask = tf.keras.Input((1, 1, seq_length));
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs);
    embeddings = tf.keras.layers.Lambda(lambda x, d_model: tf.math.sqrt(tf.cast(d_model, dtype = tf.float32)) * x, arguments = {'d_model': d_model})(embeddings);
    embeddings = PositionalEncoding(seq_length, d_model)(embeddings);
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(embeddings);
    for i in range(num_layers):
        outputs = EncoderLayer(seq_length, d_model, num_heads, code_dim, dropout_rate)([outputs, mask]);
    return tf.keras.Model(inputs = (inputs, mask), outputs = outputs);

def DecoderLayer(seq_length, d_model, num_heads, code_dim, dropout_rate):
    
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # inputs
    inputs = tf.keras.Input((seq_length, d_model));
    code = tf.keras.Input((seq_length, d_model));
    look_ahead_mask = tf.keras.Input((1, seq_length, seq_length));
    padding_mask = tf.keras.Input((1, 1, seq_length));
    
    attention1 = MultiHeadAttention(seq_length, d_model, num_heads)([inputs, inputs, inputs, look_ahead_mask]);
    attention1_inputs = tf.keras.layers.Add()([attention1, inputs]);
    attention1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(attention1_inputs);
    
    attention2 = MultiHeadAttention(seq_length, d_model, num_heads)([attention1, code, code, padding_mask]);
    attention2 = tf.keras.layers.Dropout(rate = dropout_rate)(attention2);
    attention2_attention1 = tf.keras.layers.Add()([attention2, attention1]);
    attention2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(attention2_attention1);
    
    outputs = tf.keras.layers.Dense(units = code_dim, activation = 'relu')(attention2);
    outputs = tf.keras.layers.Dense(units = d_model)(outputs);
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(outputs);
    outputs_attention2 = tf.keras.layers.Add()([outputs, attention2]);
    outputs = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(outputs_attention2);
    
    return tf.keras.Model(inputs = (inputs, code, look_ahead_mask, padding_mask), outputs = outputs);

def Decoder(vocab_size, num_layers, seq_length, d_model, num_heads, code_dim, dropout_rate):
    
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    inputs = tf.keras.Input((seq_length,));
    code = tf.keras.Input((seq_length, d_model));
    look_ahead_mask = tf.keras.Input((1, seq_length, seq_length));
    padding_mask = tf.keras.Input((1, 1, seq_length));
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs);
    embeddings = tf.keras.layers.Lambda(lambda x, d_model: tf.math.sqrt(tf.cast(d_model, dtype = tf.float32)) * x, arguments = {'d_model': d_model})(embeddings);
    embeddings = PositionalEncoding(seq_length, d_model)(embeddings);
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(embeddings);
    for i in range(num_layers):
        outputs = DecoderLayer(seq_length, d_model, num_heads, code_dim, dropout_rate)([outputs, code, look_ahead_mask, padding_mask]);
    return tf.keras.Model(inputs = (inputs, code, look_ahead_mask, padding_mask), outputs = outputs);

def Transformer(vocab_size, num_layers, seq_length, d_model, num_heads, code_dim, dropout_rate):
    
    pass;

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    encoder = Encoder(100, 5, 10, 100, 10, 100, 0.5);
    encoder.save('encoder.h5');
    decoder = Decoder(100, 5, 10, 100, 10, 100, 0.5);
    decoder.save('decoder.h5');
