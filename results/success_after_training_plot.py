import pandas as pd
import argparse
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import icdlplot
DATE_FORMAT = r'%y-%m-%d'

def valid_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, DATE_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid date: {s!r}")

if __name__ == '__main__':
    max_models = 8
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlabel', required=True, type=str, help="X Axis Label")
    parser.add_argument('--savefile', required=True, type=str, help="Save filename")
    parser.add_argument('--boldlabel', required=False, default=-1, type=int, help="Number of the label to display as bold")
    parser.add_argument('--haltung', required=True, type=str, choices=['supine', 'prone'])

    for i in range(1,max_models+1):
        parser.add_argument(f'--date{i}', required=i==1, type=valid_date, help=f"Date of the runs {i}")
        parser.add_argument(f'--suffix{i}', required=i==1, type=str, help=f"Model name suffix {i}")
        parser.add_argument(f'--label{i}', required=i==1, type=str, help=f"Label for model {i}")
        #parser.add_argument(f'--tag{i}', required=i==1, choices=['success_rate'], help=f"Tag to load for model {i}")
    args = parser.parse_args()
    dates = []
    suffixes = []
    labels = []
    haltungen = []
    tags = []

    for i in range(1,max_models+1):
        date = getattr(args, f'date{i}')
        if date is None:
            break

        suffix = getattr(args, f'suffix{i}')
        if suffix is None:
            raise ValueError(f"No suffix provided for model {i}")
        
        label = getattr(args, f'label{i}')
        if label is None:
            raise ValueError(f"No label provided for model {i}")
        
        if args.boldlabel != -1 and i == args.boldlabel:
            label = r'$\mathbf{' + label + r'}$'
        
        #tag = getattr(args, f'tag{i}')
        #if suffix is None:
        #    raise ValueError(f"No tag provided for model {i}")
        
        dates.append(date.strftime(DATE_FORMAT))
        suffixes.append(suffix)
        labels.append(label)
        #tags.append("rollout/" + tag)
        tags.append("rollout/success_rate")

    if len(dates) == 0:
        raise ValueError("No models specified...")
    
    data = {
        'Successful': [],
        'Ambigious': [],
        'Not Successful': [],
    }
    for i in range(len(dates)):
        df = pd.read_csv(f'{dates[i]}_{args.haltung}_{suffixes[i]}_test_success_rate.csv')
        # Count number of entries with success rate > 90%; < 10% and in-between.
        sr = df['Success_Rate']
        cnt = sr.count()
        cnt_90 = (sr > 0.9).sum()
        cnt_10 = (sr < 0.1).sum()
        cnt_other = cnt - cnt_90 - cnt_10

        # Normalize in percent.
        cnt_90 /= cnt / 100.0
        cnt_10 /= cnt / 100.0
        cnt_other /= cnt / 100.0

        data['Successful'].append(cnt_90)
        data['Not Successful'].append(cnt_10)
        data['Ambigious'].append(cnt_other)

    fig, ax = plt.subplots(figsize=(2.5,2.5))
    bottom = np.zeros(len(data['Ambigious']))
    width = 0.5

    colors = (
        icdlplot.PLT_COLORS[0]
        icdlplot.PLT_COLORS[2]
        icdlplot.PLT_COLORS[1]
    )

    indx = 0
    for region, weight in data.items():
        p = ax.bar(labels, weight, width, color=colors[indx], label=region, bottom=bottom)
        bottom += weight
        indx += 1

    ax.legend(loc='lower left')
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel('Number of Models [%]')
    plt.tight_layout()
    plt.savefig(f'{args.savefile}.pdf',
        dpi=300,
        bbox_inches='tight',
        format='pdf')

