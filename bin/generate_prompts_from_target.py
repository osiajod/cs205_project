import logging 
import re

import fire
import nltk


def build_prompt(lines, start, window_size):
    res = []
    idx = 0
    num_sentences = 0
    num_lines = len(lines)

    while num_sentences < window_size:
        if start + idx == num_lines:
            return None

        line = lines[start + idx]
        
        if line.strip() != '':
            num_sentences += 1

            """
            last sentence or end of paragraph -- no following space
            """
            if (num_sentences == window_size) or \
               (start + idx + 1 == num_lines) or \
               (lines[start + idx + 1] == '\n'):
                res.append(line.strip())
            else:
                res.append(line.strip() + ' ')

        else:
            res.append("\n\n")
        
        idx += 1

    if res[0] == "\n\n":
        res = res[1:]


    prompt = ''.join(res)

    return prompt 


def prompts_generator(infile, window_size=3, step_size=1):
    """prompts_generator
    Prompt generator which returns consecutive chunks of the input text, broken
    up by sentences. Sentences can stretch across multiple paragraphs
    :param window_size: number of sentences per prompt 
    :param step_size: number of lines to shift by at each iteration 

    padding text to include the proper number of sentences
    stopping when the window reaches the end of the text
    """
    with open(infile, 'r') as f:
        lines = f.readlines()
        logging.info(f'loaded {len(lines)} lines')

    num_lines = len(lines)

    for start in range(0, len(lines), step_size):
        if lines[start].strip() == '': continue

        prompt = build_prompt(lines, start, window_size)
        
        if prompt is None:
            break

        yield prompt

        #yield prompt


def sep_sentences_to_lines(infile, outfile, sep_paragraphs=True):
    """
    tokenizes sentences of infile and writes them as separate lines in 
    outfile. if sep_paragraphs is true, inclues a newline between paragraphs in 
    outfile.
    """

    with open(infile, 'r') as i:
        text = i.read()

    paragraphs = text.split('\n\n')

    with open(outfile, 'w') as o:
        for par in paragraphs:
            sentences = nltk.sent_tokenize(par)
            for sentence in sentences:
                o.write(sentence + '\n')

            if sep_paragraphs:
                o.write('\n')




if __name__ == '__main__':
    fire.Fire({
        'sep_sentences_to_lines': sep_sentences_to_lines,
        'prompts_generator': prompts_generator
    })  

