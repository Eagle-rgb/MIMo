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
num_ages=${#AGES[@]}

for ((i=0; i<num_ages; i++)); do
	for ((j=0; j<num_ages; j++)); do
		host_idx=$((i * num_ages + j))
		HOSTNAME="${HOSTPREFIXES[host_idx]}"".rbi.cs.uni-frankfurt.de"
		echo "Playing MIMo on host $HOSTNAME"
		# Putting '--save_every' as same value as '--train_for' saves only the very last model.
		ssh -o "StrictHostKeyChecking accept-new" -l ${USERNAME} ${HOSTNAME} \
			"conda activate mimo && "\
			"cd MIMo && " \
			"python results/success_after_training.py" \
			"--date=${MODEL_DATE}" \
			"--age=${MODEL_AGE}" \
			"--suffix=${MODEL_SUFFIX}" \
			"--haltung=${HALTUNG}" \
			"--age_act=${AGES[i]}" \
			"--age_body=${AGES[j]}" &
done
done

# Wait until all ssh commands finished, i.e. all MIMo simulations are finished. Then, plot the results.
wait

