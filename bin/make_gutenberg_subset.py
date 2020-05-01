from pathlib import Path
import zipfile
import zlib

import fire
from gutenberg_cleaner import simple_cleaner, super_cleaner
import pandas as pd
import tqdm


FILE_NAME_FORMAT = "raw/{id}_raw.txt"
OUT_FORMAT = "{genre}/{id}.txt"
COMPRESSION = zipfile.ZIP_DEFLATED

PG_HEADER = [
      "start of this project gutenberg ebook"
]


PG_FOOTER = [
              "end of project gutenberg's",
              "end of the project gutenberg ebook",
              "end of this project gutenberg ebook"
]


def remove_header_footer_old(pgid, data):
    new_data = []
    past_header = False
    past_footer = False
    try:
        split_data = data.decode('utf-8').split('\n')
    except UnicodeDecodeError:
        print(f"{pgid} did not have utf 8 encoding")
        split_data = data.decode('cp1252').split('\n')

    for line in split_data:
        if past_header and not past_footer:
            for pg_footer in PG_FOOTER:
                 if pg_footer in line.lower():
                    past_footer = True 
            if not past_footer: 
                new_data.append(line)
        else:
            for pg_header in PG_HEADER:
                if pg_header in line.lower():
                    past_header = True
                    
                 
    return ('\n'.join(new_data)).strip().encode('utf-8')


skip_lines = [
     'distributed proofreader',
     'distributed proofreading',
     'publisher',
     'editor',
     'html',
     'proofreaders',
     "transcriber's",
     'copyright',
     "the internet archive",
     "illustration"
]

def remove_header_footer(pgid, data):
    try:
        string = data.decode('utf-8')
    except UnicodeDecodeError:
        string = data.decode('cp1252')

    #cleaned_string = simple_cleaner(string) 
    cleaned_string_lines = super_cleaner(string).strip().split('\n')
    cleaned_string_lines = [l for l in cleaned_string_lines if l.strip() != "[deleted]"]
    cleaned_string = ('\n'.join(cleaned_string_lines)).strip()
    cleaned_string_lines = cleaned_string.split('\n') 


    remove_first_line = False
    remove_last_line = False
    for l in skip_lines:
        if l in cleaned_string_lines[0].lower():
            remove_first_line = True
   
    for l in skip_lines:
        if l in cleaned_string_lines[-1].lower():
            remove_last_line = True
   
    if remove_first_line:
        cleaned_string_lines = cleaned_string_lines[1:]

    if remove_last_line:
        cleaned_string_lines = cleaned_string_lines[:-1]

    cleaned_string = '\n'.join(cleaned_string_lines)
    

    return cleaned_string.strip().encode('utf-8')

def make_genre_subset(genre_file, pg_raw="./data/project_gutenberg_raw.zip", out_zip=None):
    print(f"Reading from {genre_file}")
    genre_df = pd.read_csv(genre_file) 

    genre_file_base = Path(genre_file).stem

    out_zip = f"./data/{genre_file_base}.zip"
    zf_out = zipfile.ZipFile(out_zip, mode='w', compression=COMPRESSION)

    print(f"Writing files to {out_zip}")

    ids = list(genre_df.id)

    zf_in = zipfile.ZipFile(pg_raw, 'r')

    for _, row in  tqdm.tqdm(genre_df.iterrows(), total=len(genre_df)):
        pgid = row['id']
        name = row['title']
        filename = FILE_NAME_FORMAT.format(id=pgid)
        outname = OUT_FORMAT.format(genre=genre_file_base, id=pgid) 
        try:
            data = zf_in.read(filename)
        except KeyError:
            print(f"{pgid}, {name} not found")
            continue
            
        data = remove_header_footer(pgid, data)
             
        zf_out.writestr(outname, data)

    zf_in.close()
    zf_out.close()


   
    

if __name__ == '__main__':
    fire.Fire(make_genre_subset)
    


