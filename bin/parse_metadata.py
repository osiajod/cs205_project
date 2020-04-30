import fire 
import pandas as pd

def filter_subjects(s):
    res = []
    
    for e in eval(s):

        res += e.lower().split('--')
        
    res = [r.strip() for r in res]
    
    return res


def generate_subjects(infile="./data/metadata.csv.gz", outfile="./data/subjects.csv"):
    df = pd.read_csv(infile).query('type == "Text"')

    all_subjects = df.subjects.apply(filter_subjects)
    subject_series = pd.Series([item for sublist in all_subjects for item in sublist])
    subject_series.value_counts().to_csv(outfile,header=None)

def get_subject_records(search_phrase, infile="./data/metadata.csv.gz", outfile=None):
    df = pd.read_csv(infile).query('type == "Text"')
    search_phrase = search_phrase.lower()

    if outfile is None:
        outfile = search_phrase.replace(' ', '_')
        outfile += '.csv'
        outfile = './data/' + outfile

    filtered_df = (df
     .query('subjects.str.lower().str.contains(@search_phrase)')
     .query('language == "[\'en\']"')
    )


    columns = [
        'id',
        'title',
        'author',
        'authoryearofbirth',
        'authoryearofdeath',
        'subjects'
    ]

    filtered_df.loc[:, columns].to_csv(outfile, index=False)
    print(f"Wrote {len(filtered_df)} lines to {outfile}")



if __name__ == '__main__':
    fire.Fire({
      'generate_subjects': generate_subjects,
      'get_subject_records': get_subject_records
  })
