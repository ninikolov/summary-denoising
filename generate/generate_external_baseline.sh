#!/usr/bin/env bash
## Denoise an external baseline output.
## Applies BPE to the input file, constructus a fairseq dataset and applies the denoising model.

# The txt file containing the summaries
baseline=$1
# The model to use to denoise
model_path=$2

if [ -z "$3" ]; then
    min_denoise_sent_length=""
else
    min_denoise_sent_length="--min_sent_length $3"
fi

baseline_name=$(basename $baseline)
out_file_name=denoise_$baseline_name$3.txt
out_file_name_ord=denoise_$baseline_name$3.txt.ord

line_count="$(wc -l < $out_file_name_ord)"
line_count=${line_count%?};
expected_length=11490

if [ -f $out_file_name_ord ] && [ "$line_count" -eq "$expected_length" ]; then
    echo " !!! $baseline_name already decoded, skipping. !!!"
    exit
fi

echo "Min sent length: $min_denoise_sent_length"

mkdir -p $baseline_name
cp $baseline $baseline_name/$baseline_name
cd $baseline_name
cp ../test.clean .

# BPE
python $BPE_CODE/apply_bpe.py --input $baseline_name --output test.noisy --codes ../train.noisy.clean.bpe

# Prepare fairseq dataset
python $FAIRSEQ_PATH/preprocess.py -s noisy -t clean --testpref test --destdir $baseline_name-data \
    --nwordssrc 50000 --nwordstgt 50000 --srcdict ../fairseq-data/dict.noisy.txt \
    --tgtdict ../fairseq-data/dict.clean.txt

# Copt dictionaries from the original dataset folder
cp ../fairseq-data/dict.*.txt $baseline_name-data


# Generate
CUDA_VISIBLE_DEVICES=$CUDA_DEV python $FAIRSEQ_PATH/generate_file.py $baseline_name-data \
    --path ../$model_path --batch-size 100 --beam 5 --output_file $out_file_name --remove-bpe --quiet \
    --max-len-a 3 $min_denoise_sent_length

cp $out_file_name.ord ../
cd ..