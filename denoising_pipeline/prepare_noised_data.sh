#!/usr/bin/env bash
### Preprocess the JSON files, preparing data to train/evaluate summary denoising models.

# The root folder where the CNN/DM JSON files are stored.
json_files_root=$1
# The type of noise to apply
noise_type=$2
# Apply noise this many times.
noise_count=$3
# Noise probability distribution
if [ -z "$4" ]; then
    noise_prob=""
else
    noise_prob="-noise_prob_distr $4"
fi

echo " * Extract and prepare data from $json_files_root * "

train_folder=train
valid_folder=val
test_folder=test
paraphrased_train=paraphrased_train
paraphrased_val=paraphrased_val
paraphrased_test=paraphrased_test

curr_dir=$PWD
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
echo $SCRIPT_DIR

echo ""
echo " * Processing train * "
cd $json_files_root/$train_folder
python $SCRIPT_DIR/../preprocessing/extract_json.py -noise_type $noise_type -noise_count $noise_count \
    -out_clean $curr_dir/train.clean.clean -out_noisy $curr_dir/train.noisy.clean $noise_prob \
    -paraphrased_folder $paraphrased_train

echo ""
echo " * Processing valid * "
cd $json_files_root/$valid_folder
python $SCRIPT_DIR/../preprocessing/extract_json.py -noise_type $noise_type -noise_count $noise_count \
    -out_clean $curr_dir/valid.clean.clean -out_noisy $curr_dir/valid.noisy.clean $noise_prob \
    -paraphrased_folder $paraphrased_val

echo ""
echo " * Processing test * "
cd $json_files_root/$test_folder
# Only noise once to keep comparable
python $SCRIPT_DIR/../preprocessing/extract_json.py -noise_type $noise_type -noise_count 1 \
    -out_clean $curr_dir/test.clean.clean -out_noisy $curr_dir/test.noisy.clean $noise_prob \
    -paraphrased_folder $paraphrased_test

cd $curr_dir
