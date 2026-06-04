#!/bin/bash

for file in *_cee* ; do
	[[ -f "$file" ]] || continue

	new_name="${file//_cee/}"
	echo "Rename file: '$file' -> '$new_name'"
	mv "$file" "$new_name"
done

echo "Fertig"
