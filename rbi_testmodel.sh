#!/bin/bash

if [ $# -ne 4 ] 
	then
		echo "Error. Provide rbi username, date, model name to load and number of runs."
		return
fi

USERNAME=$1
HOSTPREFIXES=("adrastos" "alkmene" "ajax" "anaxo" "achilles" "axylos" "aktor"
	"admeta" "amata" "agylla" "adamas" "arabia" "adonis" "aither" "apate"
	"atropos" "aletheia" "acheloos" "anemoi")
NUMBER_OF_RUNS=$4
MODEL_DATE=$2
MODEL_NAME=$3

if [ $NUMBER_OF_RUNS -ge ${#HOSTPREFIXES[@]} ]
	then
		echo "Error. Too many runs. Maximum ${#HOSTPREFIXES[@]} runs allowed."
		return
fi

today=$(date +%y-%m-%d)

RUN_OFFSET=${18}

for i in $(seq 0 $((NUMBER_OF_RUNS-1))); do
	HOSTNAME="${HOSTPREFIXES[i]}"".rbi.cs.uni-frankfurt.de"
	echo "Playing MIMo on host $HOSTNAME"
	# Putting '--save_every' as same value as '--train_for' saves only the very last model.
	ssh -o "StrictHostKeyChecking accept-new" -l ${USERNAME} ${HOSTNAME} \
		"conda activate mimo && "\
		"cd MIMo && " \
		"python mimoEnv/illustrations.py" \
		"--test" \
        "--render_frames" \
		"--render_video" \
		"--roll_over_starting_position=supine" \
        "--load_model=models/roll_over/${MODEL_DATE}/supine/${MODEL_DATE}_supine_${MODEL_NAME}_run_$((i+RUN_OFFSET))/model_1.zip" &
done

# Wait until all ssh commands finished, i.e. all MIMo simulations are finished. Then, plot the results.
wait

# Copy images.
for i in $(seq 0 $((NUMBER_OF_RUNS-1))); do
    echo "Copying run ${i}"
    scp "${USERNAME}@adrastos.rbi.cs.uni-frankfurt.de:~/MIMo/models/roll_over/${MODEL_DATE}/supine/${MODEL_DATE}_supine_${MODEL_NAME}_run_$((i+RUN_OFFSET))/*.png" "models/arch/laterality/${MODEL_DATE}_supine_${MODEL_NAME}_run_$((i+RUN_OFFSET))" &
done
wait

# Copy videos.
for i in $(seq 0 $((NUMBER_OF_RUNS-1))); do
    echo "Copying run ${i}"
    scp "${USERNAME}@adrastos.rbi.cs.uni-frankfurt.de:~/MIMo/models/roll_over/${MODEL_DATE}/supine/${MODEL_DATE}_supine_${MODEL_NAME}_run_$((i+RUN_OFFSET))/*.avi" "models/arch/laterality/${MODEL_DATE}_supine_${MODEL_NAME}_run_$((i+RUN_OFFSET))" &
done
wait

