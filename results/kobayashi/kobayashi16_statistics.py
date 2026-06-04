""" This file is used to perform Jensen-Shannon Divergence Statistical Test
on Kobayashi data. """
import argparse
import numpy as np
from scipy.spatial import distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_1', required=True, type=str,
                        help="Distribution 1 (as .npy)")
    parser.add_argument('--npy_2', required=True, type=str,
                        help="Distribution 2 (as .npy)")
    args = parser.parse_args()

    p = np.load(args.npy_1)
    q = np.load(args.npy_2)

    print(f"JS(P || Q): {(distance.jensenshannon(p, q) ** 2.0 * 100.0):.2f}%")


