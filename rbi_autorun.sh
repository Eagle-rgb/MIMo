#!/bin/bash

if [ $# -ne 3 ] 
	then
		echo "Error. Provide rbi username, model name and number of runs."
		return
fi

USERNAME=$1
HOSTPREFIXES=("adrastos" "alkmene" "ajax" "anaxo" "achilles" "axylos" "aktor"
	"admeta" "amata" "agylla" "ares" "adamas" "arabia" "adonis" "aither" "apate"
	"arges" "atropos" "aletheia" "acheloos" "anemoi")
NUMBER_OF_RUNS=$3
MODEL_NAME=$2

if [ $NUMBER_OF_RUNS -ge ${#HOSTPREFIXES[@]} ]
	then
		echo "Error. Too many runs. Maximum ${#HOSTPREFIXES[@]} runs allowed."
		return
fi

for i in $(seq 1 $NUMBER_OF_RUNS); do
	HOSTNAME="${HOSTPREFIXES[i]}"".rbi.cs.uni-frankfurt.de"
	echo "Playing MIMo on host $HOSTNAME"
	ssh -l ${USERNAME} ${HOSTNAME} "conda activate mimo && cd MIMo && python mimoEnv/illustrations.py" "--train_for=1000000" "--test_for=500" "--roll_over_starting_position=prone" "--roll_over_model_path_auto" "--save_model=${MODEL_NAME}_run_${i}" "--render_video" &
done

