""" This file takes all .csv files in the current directory and converts (copies) them
to .npy files for roll to lateral durations. """

import pandas as pd
import os
import re
import numpy as np

pattern = re.compile(r'(\d{2}-\d{2}-\d{2})_' # date
                     r'(prone|supine)_' # haltung
                     r'age(\d)_' # age
                     r'(?:transferlearning_age(\d)_)?'
                     r'statistics.csv')

files = [f for f in os.listdir('.') if os.path.isfile(f)]

for file in files:
    match = re.search(pattern, file)
    if not match: continue
    date = match.group(1)
    haltung = match.group(2)
    source_age = match.group(3)
    tf_age = match.group(4)  # None or the transferlearning age

    transferlearning = tf_age is not None

    # Load existing .csv data.
    df = pd.read_csv(file, index_col=['Run', 'Episode'])
    # Reduce to only successful episodes.
    successful_df = df[df['Success'] == True]
    # Get durations until reaching lateral
    durations = successful_df['Time_SideLying']

    if transferlearning:
        np.save(f'duration_{'lateral'}_{haltung}_age{source_age}_transferlearning_age{tf_age}.npy', durations)
    else:
        np.save(f'duration_{'lateral'}_{haltung}_age{source_age}.npy', durations)

