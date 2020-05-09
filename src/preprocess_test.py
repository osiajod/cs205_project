import tensorflow as tf
from gpt2_keras.encoder import get_encoder

def preprocess(filepath = "../data/target_texts/the_circular_ruins_lines.txt", model_dir="./models/", model_name = "124M", max_seq_len=500):
    # read using with open
    with open(filepath, "r") as fp:
        lines = fp.readlines()

    # print(lines)

    # fix the length to 1024, pad space
    # model_dir = "./models/"
    # model_name = "124M"
    enc = get_encoder(model_name, model_dir)

    corpus = []

    for line in lines:
        corpus.append(enc.encode(line))

    for idx, line in enumerate(corpus):
        while len(line) < max_seq_len:
            line.append(enc.encode(" ")[0])
        if len(line) > max_seq_len: raise Exception(f"> {max_seq_len}!")

        if idx == len(corpus)-1:
            line[-1] = 50256 # # code for <|endoftext|>
        corpus[idx] = line



    return corpus

