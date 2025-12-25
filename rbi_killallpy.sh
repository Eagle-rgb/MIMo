#!/bin/bash

if [ $# -ne 2 ] 
	then
		echo "Error. Provide rbi username and number of runs."
		return
fi

USERNAME=$1
HOSTPREFIXES=("adrastos" "alkmene" "ajax" "anaxo" "achilles" "axylos" "aktor"
	"admeta" "amata" "agylla" "ares" "adamas" "arabia" "adonis" "aither" "apate"
	"arges" "atropos" "aletheia" "acheloos" "anemoi")
NUMBER_OF_RUNS=$2

if [ $NUMBER_OF_RUNS -ge ${#HOSTPREFIXES[@]} ]
	then
		echo "Error. Too many runs. Maximum ${#HOSTPREFIXES[@]} runs allowed."
		return
fi

for i in $(seq 1 $NUMBER_OF_RUNS); do
	HOSTNAME="${HOSTPREFIXES[i]}"".rbi.cs.uni-frankfurt.de"
	echo "Killing MIMo on host $HOSTNAME"
	ssh -l ${USERNAME} ${HOSTNAME} "killall python"
done

