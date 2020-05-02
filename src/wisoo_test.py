import json
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import numpy as np
from gpt2_keras.gpt2 import GPT2
from gpt2_keras.builder import original_gpt2
from gpt2_keras.builder.builder import build
# from .builder.builder import build


with open("./models/124M/hparams.json") as f:
    config = json.load(f)
#
gpt2 = GPT2(config, name='gpt2')

# x= tf.placeholder(dtype=tf.int32, shape=[None, None])
# y = gpt2(x)

# print(type(config))

# gpt2= build(config, "./models/124M/checkpoint", name='gpt2')
print(type(gpt2))
print(gpt2.layers[1].layers)

# print(gpt2.summary())
# print(gpt2.summary())

# with tf.Session() as sess:
#     print(sess.run(y, feed_dict={x:[None,None]}))


