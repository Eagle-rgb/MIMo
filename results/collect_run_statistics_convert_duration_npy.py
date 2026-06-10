""" This file takes all .csv files in the current directory and converts (copies) them
to .npy files for roll to lateral durations. """

import pandas as pd
import os
import re
import numpy as np

pattern = re.compile(r'(\d{2}-\d{2}-\d{2})_' # date
                     r'(prone|supine)_' # haltung
                     r'age(\d)_cee_' # age
                     r'act(\d)_' # actuator (physio) age
                     r'body(\d)_' # body (morph) age
                     r'statistics.csv')

files = [f for f in os.listdir('.') if os.path.isfile(f)]

for file in files:
    match = re.search(pattern, file)
    if not match: continue
    date = match.group(1)
    haltung = match.group(2)
    source_age = match.group(3)
    physio_age = match.group(4)
    morph_age = match.group(5)

    # Load existing .csv data.
    df = pd.read_csv(file, index_col=['Run', 'Episode'])
    # Reduce to only successful episodes.
    successful_df = df[df['Success'] == True]
    # Get durations until reaching lateral
    durations = successful_df['Time_SideLying']

    np.save(f'{date}_{haltung}_age{source_age}_'
            f'cee_physio{physio_age}_morph{morph_age}_'
            'durations.npy', durations)
