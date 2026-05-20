#!/bin/bash
# This file is used to perform crossembodiment statistics. As input you enter:
# * rbi username
# * the age of the model at which it was trained at
# * the suffix of the model
# * the date of the model
# * the posture ('haltung'): prone / supine

if [ $# -ne 5 ] 
	then
		echo "Error. Provide rbi username, age, suffix, date and posture"
		return
fi

USERNAME=$1
MODEL_AGE=$2
MODEL_SUFFIX=$3
MODEL_DATE=$4
HALTUNG=$5

HOSTPREFIXES=("adrastos" "alkmene" "ajax" "anaxo" "achilles" "axylos" "aktor"
	"admeta" "amata" "agylla" "adamas" "arabia" "adonis" "aither" "apate"
	"atropos" "aletheia" "acheloos" "anemoi")

today=$(date +%y-%m-%d)

AGES=( 1 3 6 9 )

for ((i=0; i<${#AGES[@]}; i++)); do
	HOSTNAME="${HOSTPREFIXES[i]}"".rbi.cs.uni-frankfurt.de"
	echo "Playing MIMo on host $HOSTNAME"
	# Putting '--save_every' as same value as '--train_for' saves only the very last model.
	ssh -o "StrictHostKeyChecking accept-new" -l ${USERNAME} ${HOSTNAME} \
		"conda activate mimo && "\
		"cd MIMo && " \
		"python results/collect_run_statistics.py" \
		"--date=${MODEL_DATE}" \
		"--age=${MODEL_AGE}" \
		"--suffix=${MODEL_SUFFIX}" \
		"--haltung=${HALTUNG}" \
		"--transfer_learning" \
		"--transferlearning_age=${AGES[i]}" &
done

# Wait until all ssh commands finished, i.e. all MIMo simulations are finished. Then, plot the results.
wait

