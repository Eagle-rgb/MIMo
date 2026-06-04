""" This file is used to summarize many .csv files into a single .csv file.
'success_after_training.py' produces .csv files ..._act{x}_body{y}_...
This file takes all actuator and body ages and puts them into a single csv
file. The original csv file had columns 'Run' and 'Success_Rate'. The new
csv file has extra columns 'Physio_Age' (Act_Age), 'Morph_Age' (Body_Age).

The file is execute in the current working directory and does not go recursively.

We purposely make this an extra file, because we want to keep the ability to
perform 'success_after_training.py' in parallel. And that works best if they
all produce separate test_success_rate.csv files. Afterwards, just call this function
and all your results get summarized.
"""
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,os.path.join(dir_path, '..'))
sys.path.insert(0,os.path.join(dir_path, '..', '..'))

import argparse
from utils import DATE_FORMAT, valid_date
import re
import pandas as pd

pattern = re.compile(r'(\d{2}-\d{2}-\d{2})_((?:supine)|(?:prone))_([a-z0-9_-]+)_act(\d)_body(\d)_test_success_rate.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=valid_date, required=True)
    parser.add_argument('--suffix', type=str, required=True),
    parser.add_argument('--haltung', type=str, choices=['supine', 'prone'], required=True)
    args = parser.parse_args()
    args_date = args.date.strftime(DATE_FORMAT)

    df_list = []

    for f in os.listdir('.'):
        if not os.path.isfile(f):
            continue

        match = pattern.search(f)
        if not match: continue

        date, haltung, suffix, physio_age, morph_age = match.groups()
        if args_date != date or args.suffix != suffix or args.haltung != haltung: continue

        print(f"Found file {f}!")

        df = pd.read_csv(f)
        df['Physio_Age'] = physio_age
        df['Morph_Age'] = morph_age
        df_list.append(df)

    if len(df_list) == 0:
        print("Error. No files found...")
    else:
        df: pd.DataFrame = pd.concat(df_list)
        df.to_csv(f'{args_date}_{args.haltung}_{args.suffix}_cee_test_success_rate.csv')
        print("Finished!")









