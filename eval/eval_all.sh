#!/usr/bin/env bash

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

out_file=all_results.txt
rm $out_file

echo "Input files: $@"
for f in $@
do
    echo " ---> Evaluating $f <--- "
    bash $SCRIPT_DIR/eval.sh $f >> $out_file
    echo "" >> $out_file
    echo "*********************************************" >> $out_file
    echo "" >> $out_file
done