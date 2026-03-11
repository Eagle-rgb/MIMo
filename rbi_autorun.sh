#!/bin/bash

if [ $# -ne 3 ] 
	then
		echo "Error. Provide rbi username, model name and number of runs."
		return
fi

USERNAME=$1
HOSTPREFIXES=("adrastos" "alkmene" "ajax" "anaxo" "achilles" "axylos" "aktor"
	"admeta" "amata" "agylla" "adamas" "arabia" "adonis" "aither" "apate"
	"atropos" "aletheia" "acheloos" "anemoi")
NUMBER_OF_RUNS=$3
MODEL_NAME=$2

# Date. Used to run gemini_plot script after all runs are finished. Do this before
# starting the runs in case we start them before midnight so we do not read a wrong
# date.
today=$(date +%y-%m-%d)

if [ $NUMBER_OF_RUNS -ge ${#HOSTPREFIXES[@]} ]
	then
		echo "Error. Too many runs. Maximum ${#HOSTPREFIXES[@]} runs allowed."
		return
fi

for i in $(seq 0 $((NUMBER_OF_RUNS-1))); do
	HOSTNAME="${HOSTPREFIXES[i]}"".rbi.cs.uni-frankfurt.de"
	echo "Playing MIMo on host $HOSTNAME"
	# Putting '--save_every' as same value as '--train_for' saves only the very last model.
	ssh -o "StrictHostKeyChecking accept-new" -l ${USERNAME} ${HOSTNAME} \
		"conda activate mimo && "\
		"cd MIMo && "\
		"python mimoEnv/illustrations.py" \
		"--train_for=1000000" \
		"--save_every=1000000" \
		"--roll_over_starting_position=prone" \
		"--algorithm=PPO" \
		"--pen_factor=0.02" \
		"--roll_over_model_path_auto" \
		"--goal_achievement_function=cos" \
		"--save_model=${MODEL_NAME}_run_$((i))" \
		"--pbrs" \
		"--age=9" \
		"--pbrs_w=100" \
		"--lr=0.0003" &
done

# Wait until all ssh commands finished, i.e. all MIMo simulations are finished. Then, plot the results.
wait

# Create gemini plots.
ssh -l ${USERNAME} "adrastos.rbi.cs.uni-frankfurt.de" "conda activate mimo" "&&"\
	"cd MIMo" "&&" \
	"python" "results/tb_plot_2.py" "--date=${today}" "--suffix=${MODEL_NAME}"

# Copy gemini plots to our local machine.
# Gemini plot files always have the same prefix: yy-mm-dd_<model sufix>_****.png
scp "${USERNAME}@adrastos.rbi.cs.uni-frankfurt.de:~/MIMo/png/${today}_${MODEL_NAME}_*" "."


