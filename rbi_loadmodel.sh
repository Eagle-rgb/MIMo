#!/bin/bash

if [ $# -ne 2 ] 
	then
		echo "Error. Provide rbi username and number of runs."
		return
fi

USERNAME=$1
HOSTPREFIXES=("adrastos" "alkmene" "ajax" "anaxo" "achilles" "axylos" "aktor"
	"admeta" "amata" "agylla" "adamas" "arabia" "adonis" "aither" "apate"
	"atropos" "aletheia" "acheloos" "anemoi")
NUMBER_OF_RUNS=$2

# Date. Used to run gemini_plot script after all runs are finished. Do this before
# starting the runs in case we start them before midnight so we do not read a wrong
# date.
today=$(date +%y-%m-%d)

model_date="26-03-02"
model_starting_position="supine"
model_name="age_6"

if [ $NUMBER_OF_RUNS -ge ${#HOSTPREFIXES[@]} ]
	then
		echo "Error. Too many runs. Maximum ${#HOSTPREFIXES[@]} runs allowed."
		return
fi

for i in $(seq 1 $NUMBER_OF_RUNS); do
	HOSTNAME="${HOSTPREFIXES[i]}"".rbi.cs.uni-frankfurt.de"
	echo "Playing MIMo on host $HOSTNAME"
	# Putting '--save_every' as same value as '--train_for' saves only the very last model.
	ssh -o "StrictHostKeyChecking accept-new" -l ${USERNAME} ${HOSTNAME} \
		"conda activate mimo && "\
		"cd MIMo && "\
		"python mimoEnv/illustrations.py" \
		"--train_for=1000000" \
		"--save_every=1000000" \
		"--roll_over_starting_position=${model_starting_position}" \
		"--load_model=models/roll_over/${model_date}/${model_starting_position}/${model_date}_${model_starting_position}_${model_name}_run_${i}/model_1.zip" &
done

# Wait until all ssh commands finished, i.e. all MIMo simulations are finished. Then, plot the results.
wait

# Create gemini plots.
ssh -l ${USERNAME} "adrastos.rbi.cs.uni-frankfurt.de" "conda activate mimo" "&&"\
	"cd MIMo" "&&" \
	"python" "results/tb_plot_2.py" "--date=${model_date}" "--suffix=${model_name}"

# Copy gemini plots to our local machine.
# Gemini plot files always have the same prefix: yy-mm-dd_<model sufix>_****.png
scp "${USERNAME}@adrastos.rbi.cs.uni-frankfurt.de:~/MIMo/png/${model_date}_${model_name}_*" "."


