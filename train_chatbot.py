#!/usr/bin/python3

import tensorflow as tf;
from create_datasets import create_datasets;
from Transformer import Transformer;

def main():

    trainset, tokenizer = create_datasets();

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
