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

# gpt2= build(config, "./models/124M/model.ckpt.data-00000-of-00001", name='gpt2')
gpt2= build(config, "./models/124M/model.ckpt", name='gpt2')

print(type(gpt2))
# print(gpt2.layers[1].layers) # The Transformer

embedding_layer = gpt2.layers[0]


print(embedding_layer)  # The Embedding Layer


print("printing vocab size:",  embedding_layer.vocab_size) #50257
print("printing word embedding:",  embedding_layer.word_embedding) #(50257 , 768)=


# gpt2.compile(
#     optimizer=tf2.optimizers.RMSprop(lr=0.01),
#     loss = tf2.keras.losses.MeanSquaredError(),
#     metrics = ['accuracy']
# )


print(gpt2.summary())
print("printing Transformer summary")
print(gpt2.layers[1].summary())

# print(gpt2.summary())
batch_size =1
max_seq_length = 1024
word_embedding = 768
tf.keras.backend.set_floatx('float64')
# input1 = np.random.randint(embedding_layer.vocab_size, size=(batch_size, 5, embedding_layer.word_embedding[-1]))
input1 = np.random.randint(embedding_layer.vocab_size, size=(batch_size,max_seq_length))
# print(input1)
output = gpt2(input1)

print(output)

# with tf.Session() as sess:
#     print(sess.run(y, feed_dict={x:[None,None]}))



