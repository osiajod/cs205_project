import fire
import generators
from generate_prompts import prompts_generator
import tqdm

def generate_lines(generator, prompts, gen_length):
    for prompt in prompts:
        print("prompt:")
        print(prompt)
        generation = generator.generate(prefix=prompt, length=gen_length)
        print("generation:")
        print(generation)
        yield generation

def write(output_file,
          checkpoint_dir,
          prompts_file,
          window_size=3,
          step_size=1,
          maxsentences=1,
          minsentences=1,
          gen_length=1000,
          run_name=''):

    # Decorated generators
    generator = generators.GPT2Generator(checkpoint_dir=checkpoint_dir, 
run_name=run_name)

    generator = generators.SentenceTruncation(generator, 
                                  maxsentences=maxsentences, 
                                  minsentences=minsentences)

    generator = generators.TokenFilter(generator)


    prompts = prompts_generator(prompts_file, window_size=window_size, step_size=step_size)

    lines = generate_lines(generator, prompts, gen_length)

    print("Starting text generation")

    with open(output_file, 'a') as o:
        for line in tqdm.tqdm(lines):
            o.write(line + '\n')


if __name__ == '__main__':
    fire.Fire(write)
