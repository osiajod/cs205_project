import logging
import gpt_2_simple as gpt2
import re
import itertools

import nltk



class GPT2Generator:
    """GPT2Generator
    Loads given GPT2 checkpoint, then generates text from prompt."""

    def __init__(self, checkpoint_dir, run_name):
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.sess = gpt2.start_tf_sess()

        gpt2.load_gpt2(self.sess,
                run_name=self.run_name,
                checkpoint_dir=self.checkpoint_dir,
                model_name=None,
                model_dir='models')

    def generate(self, **kwargs):
        generations = gpt2.generate(self.sess,
                run_name=self.run_name,
                checkpoint_dir=self.checkpoint_dir,
                top_k=0,
                top_p=0.95,
                temperature=1.0,
                nsamples=1,
                batch_size=1,
                return_as_list=True,
                include_prefix=True,
                truncate=None,
                **kwargs)

        generations = [self.filter_prefix(gen, kwargs.get('prefix')) for gen in generations]


        return generations[0]

    def filter_prefix(self, generation, prefix):
        return generation[len(prefix):]


class SentenceTruncation:

    def __init__(self, generator, maxsentences=100, minsentences=1):
        self.generator = generator
        self.MAX_RETRIES = 100
        self.maxsentences = maxsentences
        self.minsentences = minsentences

    def generate(self, **kwargs):
        generation = ""
        i = 0
        while i < self.MAX_RETRIES:
            i += 1
            generation = self.generator.generate(**kwargs)
            sentences = nltk.sent_tokenize(generation)
            if len(sentences) >= self.minsentences:
                return ' '.join(sentences[:self.maxsentences])
        raise Exception("Can't generate enough sentences after {self.MAX_RETRIES} retries")

class TokenFilter:

    def __init__(self, generator, token_list=['<|startoftext|>', '<|endoftext|>']):
        assert token_list
        self.token_list = token_list
        self.generator = generator
        self.MAX_RETRIES = 100

    def generate(self, **kwargs):
        generation = self.token_list[0]
        i = 0
        while self.contains_bad_tokens(generation):
            i += 1
            if i == self.MAX_RETRIES:
                raise Exception(f"Can't generate correct content after {self.MAX_RETRIES} retries")
            generation = self.generator.generate(**kwargs)
        return generation

    def contains_bad_tokens(self, content):
        mask = [token in content for token in self.token_list]
        answer = any(mask)
        return answer
