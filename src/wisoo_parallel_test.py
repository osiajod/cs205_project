import json
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
from mpi4py import MPI
from gpt2_keras.builder.gpt2parallelize import GPT2, MultiLayerPerceptron
from gpt2_keras.builder import original_gpt2
from gpt2_keras.builder.builderParallel import build
# from .builder.builder import build
from gpt2_keras.encoder import get_encoder
from preprocess_test import preprocess
import pickle

print("import success")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
batch_size = 1
word_embedding = 768
max_seq_length = 500
# !!!!!!!!! UNCOMMENT BELOW WHEN RUNNING ON AWS
# assert size >= 12  # Let us program for the case where each core can get one MLP layer each

# (There are 12 MLPs in GPT-2)

if rank == 0: # If this is the master, start building the GPT2 model
    with open("./models/124M/hparams.json") as f:
        config = json.load(f)

    model_dir = "./models/"
    model_name = "124M"
    enc = get_encoder(model_name, model_dir)
    gpt2= build(config, "./models/124M/model.ckpt", name='gpt2')

    corpus = preprocess(
        filepath="../data/target_texts/the_circular_ruins_lines.txt",
        model_dir="./models/",
        model_name="124M")
    # print(corpus)

    # Once the build is complete, take all the MLPs in the Decoder blocks
    embedding_layer = gpt2.layers[0]
    transformer_model = gpt2.layers[1]
    MLPs = [] # list of MLP layers( to be precise, their config dicts) in all decoder blocks
    ATTNs = []
    for decoderblock in transformer_model.layers[:-1]:  # The last layer is the Layer Normalization
        #point-to-point communication
        ser = tf.keras.layers.serialize(decoderblock.mlp)
        # print(ser)
        MLPs.append(ser)
        ATTNs.append(decoderblock.attention)

    # Broadcast all the MLPs to everyone (much more efficient than simple copy of the entire Transformer)
    comm.bcast(MLPs, root=0)  # The MLPs are broadcast from rank=0

    # for worker_num, MLP in enumerate(MLPs):  # The last layer is the Layer Normalization
    #     #point-to-point communication
    #     #Send
    #     continue_forward = True
    #     comm.isend(MLPs[worker_num], dest=worker_num+1, tag=worker_num+1)

    #Start doing forward runs


    # For simplicity, let's train for the case where 1 line = 1 instance of a batch

    epochs = 10

    for ith_epoch in range(epochs):
        embedded = embedding_layer(corpus)
        A1 = ATTNs[0](embedded)
        # print(A1.shape) # (104, 500, 768)  == (batch, max_seq, embed_size)
        #Evenly Send out partintioned Z matrix to ALL THE WORKERS. That is, each worker gets max_seq(=500) / size
        Z1 = embedded + A1
        # print(Z1.shape) # (104, 500, 768)  == (batch, max_seq, embed_size)

        for worker_num in range(size):
            #point-to-point communication
            #Send
            if ith_epoch == epochs-1:
                continue_forward = False
            else:
                continue_forward = True
            worker_share = max_seq_length // size  # 500 / 12 = 41.666 ==> 41
            offset = worker_num*worker_share # 0 ==> 41 ==> 82 ....
            print(offset)
            if worker_num != size-1:
                req_Z1 = comm.isend(Z1[:, offset: offset+worker_share,:], dest=worker_num+1, tag=worker_num+301) # 301 : tag code for Z1 slice
            else:
                req_Z1 = comm.isend(Z1[:, offset:, :], dest=worker_num+1, tag=worker_num+301)

        # Continue on with embedding



        # After finishing embedding each batch, test() for processed Z1

        # Reassemble outputs mlp(z1) mlp





    print("printing vocab size:",  embedding_layer.vocab_size) #50257
    print("printing word embedding:",  embedding_layer.word_embedding) #(50257 , 768)=


    tf1.keras.backend.set_floatx('float64')



    raw_text = "What is interesting is the fact that the first"
    raw_text1 = "My family is doing fine."
    raw_text2 = "But, I think"
    # raw_text += '<|endoftext|>'
    bpe_tokens = enc.encode(raw_text)
    bpe_tokens1 = enc.encode(raw_text1)
    # bpe_tokens2 = enc.encode(raw_text2)



    print("bpe_tokens: ", bpe_tokens)
    print("bpe_tokens1: ", bpe_tokens1)


    # Some mechanism to set continue_forward = False when feed forward completes

    # Don't forget to do the layer norm at the end

else: # worker cores  / nodes
    #Receive the MLPs
    data = None
    # Wait for the mlp config dicts to arrive
    # For the receivers, calling bcast is actually the act of receiving
    MLPs = comm.bcast(data, root=0)
    # req_mlp = comm.irecv(source=0, tag=rank)
    continue_forward = True
    recon_MLPs = []
    #reconstruct MLP using from_config
    for mlp in MLPs:
        temp = MultiLayerPerceptron(embedding_size=768,
                             perceptron_size=3072,
                             trainable=True,
                             initializer_range=0.02,
                             name=None
                             )
        temp.layer_norm.weights = mlp["layer_norm"]
        temp.perceptron.weights = mlp["perceptron"]
        temp.projection.weights = mlp["projection"]
        recon_MLPs.append(temp)



    while continue_forward:
        # For each MLPs check if isend was called
        inputs = comm.irecv(source=0, tag=rank+200) # rank + 200 : inputs from this decoder block's
        # mlp_outputs = mlp.call(inputs)
        continue_forward = comm.irecv(source=rank - 1,
                                      tag=rank + 100)  # rank + 100 indicates the Boolean for stopping the forward run

        # remember to tell the next guy about continue_forward
        comm
        comm.isend(continue_forward)
