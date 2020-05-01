from pathlib import Path
import zipfile
import zlib

import fire
import pandas as pd
import tqdm


FILE_NAME_FORMAT = "raw/{id}_raw.txt"
OUT_FORMAT = "{genre}/{id}.txt"
COMPRESSION = zipfile.ZIP_DEFLATED

PG_HEADER = [
      "start of this project gutenberg ebook"
]


PG_FOOTER = [
              "end of the project gutenberg ebook",
              "end of this project gutenberg ebook"
]


def remove_header_footer(pgid, data):
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
    


