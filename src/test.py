import json
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import numpy as np
from gpt2_keras.gpt2 import GPT2
from gpt2_keras.builder import original_gpt2
from gpt2_keras.builder.builder import build
# from .builder.builder import build
from gpt2_keras.encoder import get_encoder



def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.compat.v1.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
        pred=tf.equal(k, 0),
        true_fn=lambda: logits,
        false_fn=lambda: _top_k(),
    )




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



# input1 = np.random.randint(embedding_layer.vocab_size, size=(batch_size,max_seq_length))
# output = gpt2(input1)
# print(output)

model_dir = "./models/"
model_name = "124M"

enc =get_encoder(model_name, model_dir)
raw_text = "Is your family well?"

raw_text += '<|endoftext|>'
bpe_tokens = enc.encode(raw_text)


print(bpe_tokens)

output2 = gpt2([bpe_tokens])

print("printing argmax of logits")
print(np.argmax(output2, axis=2))

# start_token = enc.encoder['<|startoftext|>'] #
end_token = enc.encoder['<|endoftext|>']

# print("end_token is: ", start_token)
print("end_token is: ", end_token)

decoded = enc.decode(bpe_tokens)

print(decoded)


print("printing output2 :", output2)

print("")



# with tf.Session() as sess:
#     print(sess.run(y, feed_dict={x:[None,None]}))

#labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result)
#  and dtype int32 or int64. Each entry in labels must be an index in [0, num_classes).
#  Other values will raise an exception when this op is run on CPU,
#  and return NaN for corresponding loss and gradient rows on GPU.

#logits: Unscaled log probabilities of shape
# [d_0, d_1, ..., d_{r-1}, num_classes] and dtype float16, float32, or float64



# context = tf.placeholder(tf.int32, [batch_size, None])
# loss = tf.reduce_mean(
#         input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
#             labels=context[:, 1:], logits=output2[:, :-1]))


