#!/usr/bin/env bash
# Evaluate with ROUGE and METEOR
# INPUTS: a summary text file; and a directory to store the JSON summary files

# The file to evaluate
test_file=$1
# Directory to store the summary files
# Label for the current system we are evaluating

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# The folder that contains all the references
ref_dir=$SCRIPT_DIR/test_cnndm_references
label=$test_file

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

rm -rf system-$1
mkdir -p system-$1
json_store_dir=system-$1

echo " * ---Evaluate model output $test_file--- * "
echo " * ---References folder: $ref_dir--- * "
wc -l $test_file
ls -lh $ref_dir | wc -l

# Split the summaries into multiple files, to match the format of ROUGE.
python $SCRIPT_DIR/../preprocessing/split_summaries.py -summary_file $test_file -out_dir $json_store_dir

# Get the root directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
BASE="$(dirname "$SCRIPTPATH")"

export ROUGE=$BASE/RELEASE-1.5.5

echo "ROUGE path: $ROUGE"
echo "RUNNING ROUGE..."
python $SCRIPT_DIR/../eval/eval_acl.py --rouge --decode_dir=$json_store_dir \
    --ref_dir=$ref_dir > rouge_$label.txt
cat rouge_$label.txt

# Statistics 
python $SCRIPT_DIR/../eval/dataset-stat.py -target_file $test_file \
    -sentence_separator " <s> "

# Other metrics
echo ""
# Repeat rate, decreasing is better --> less redundancy in the summary
python $SCRIPT_DIR/../eval/repeat_rate.py $test_file

echo " * Finished eval of $test_file! * "
rm -rf system-$1
rm rouge_$label.txt