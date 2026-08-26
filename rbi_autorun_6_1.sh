#!/bin/bash

if [ $# -ne 2 ] 
	then
		echo "Error. Provide rbi username, model name"
		return
fi

USERNAME=$1
HOSTPREFIXES=("adrastos" "alkmene" "ajax" "anaxo" "achilles" "axylos")
MODEL_NAME=$2

# Date. Used to run gemini_plot script after all runs are finished. Do this before
# starting the runs in case we start them before midnight so we do not read a wrong
# date.
today=$(date +%y-%m-%d)

for i in $(seq 0 5); do
	HOSTNAME="${HOSTPREFIXES[i]}"".rbi.cs.uni-frankfurt.de"
	echo "Playing MIMo on host $HOSTNAME"
	# Putting '--save_every' as same value as '--train_for' saves only the very last model.
	ssh -o "StrictHostKeyChecking accept-new" -l ${USERNAME} ${HOSTNAME} \
		"conda activate mimo && "\
		"cd MIMo && "\
		"python mimoEnv/illustrations.py" \
		"--train_for=1000000" \
		"--save_every=1000000" \
		"--roll_over_starting_position=supine" \
		"--algorithm=SAC" \
		"--her" \
		"--sparse_reward" \
		"--eval_every=25000" \
		"--eval_episodes=20" \
		"--train_freq=4" \
		"--pen_factor=0.02" \
		"--no_done_active" \
		"--goal_low=0.0" \
		"--goal_high=0.95" \
		"--episode_steps=200" \
		"--roll_over_model_path_auto" \
		"--goal_achievement_function=cos" \
		"--save_model=${MODEL_NAME}_run_$((i))" \
		"--morph_age=9" \
		"--physio_age=9" \
		"--lr=0.0003" &
done

# Wait until all ssh commands finished, i.e. all MIMo simulations are finished. Then, plot the results.
wait


