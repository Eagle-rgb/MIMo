""" Deterministic Initial State Sampling

This file generates (or reads, if provided) a file containing many initial seeds and
performs Kobayashi Pattern Observation collection (the Kobayashi Classification must
be done in an additional call to kobayashi.py).
"""
import numpy as np
from results.utils import make_env, valid_date, DATE_FORMAT
import argparse
from mimoEnv.envs.roll_over import MIMoRollOverEnv
from results.collect_observation_util import collect_kobayashi_displacements_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=valid_date, required=True)
    parser.add_argument('--suffix', type=str, required=True)
    # Haltung is always 'supine'
    haltung = 'supine'
    parser.add_argument('--age', choices=[1,3,6,9], type=int, required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--run', type=int, required=True,
                        help="The run number to collect statistics for.")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory to save the collected data to.")
    parser.add_argument('--model_dir', type=str, required=True,
                        help="Any parent directory of the model.")
    args = parser.parse_args()
    date = args.date.strftime(DATE_FORMAT)
    env = make_env(age_physio=args.age, age_morph=args.age,
                   starting_position=haltung)

    qpos_shape = env.data.qpos.shape[0]
    qpos_shape -= 7  # we only want to alter the joint qpos
    num_test_cases = args.num_samples
    np.random.seed(42)  # Important for reproducibility
    initial_offsets = np.random.uniform(low=-0.1, high=0.1,
                size=(num_test_cases, qpos_shape))
    np.save(f'{args.save_dir}/initial_offsets.npy', initial_offsets)
    initial_qpos = np.zeros((num_test_cases, qpos_shape))

    print("Generating initial qpos...")
    for i in range(len(initial_offsets)):
        env.deterministic_initial_state_sampling = initial_offsets[i]
        env.reset()
        initial_qpos[i] = env.data.qpos[7:]

    np.save(f'{args.save_dir}/initial_qpos.npy', initial_qpos)
    
    print("Collecting episode data...")
    df = collect_kobayashi_displacements_all(env=env, date=date, pos=haltung, suffix=args.suffix,
                                             n_tries=num_test_cases, diss=initial_offsets,
                                             run_num=args.run, model_dir=args.model_dir)
    
    df.to_csv(f'{args.save_dir}/kobayashidata_{date}_{args.suffix}_run{args.run}_diss.csv')
