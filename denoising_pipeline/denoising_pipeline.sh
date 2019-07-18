#!/usr/bin/env bash
# Simple noise pipeline: prepare data, train, generate, eval
# Example Usage: bash denoising_pipeline.sh 0 replace 1 true true "0.15 0.85"

export CUDA_DEV=$1
## Noise settings
# Noise type to apply. Currently implemented: shuffle, replace.
noise_type=$2
# Number of times to noise each data point
noise_count=$3
# if set to true, will regenerate the data. False reuses old preprocessed data.
prepare_data=$4
# if set to true, will train/continue training a model. False assumes we already have a trained model.
train_model=$5
# Noise probability distribution
noise_prob=$6

curr_dir=${PWD##*/}
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export BASE_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export FAIRSEQ_PATH=$BASE_PROJECT_DIR/fairseq-apr19
export BPE_CODE=$BASE_PROJECT_DIR/subword-nmt

if [ -z $CNNDM_PATH ] ; then
    echo "You didn't set the CNNDM_PATH variable. Check the documentation."
    exit 
else 
    echo " * Starting pipeline with $noise_type noise. * "
    echo ""
    echo "Environment variables:"
    echo "FAIRSEQ_PATH=$FAIRSEQ_PATH"
    echo "CNNDM_PATH=$CNNDM_PATH"
    echo "BPE_CODE=$BPE_CODE"
    echo ""
fi

#model_architecture=lstm
export MODEL=lstm_tiny
export MODEL_PATH=$MODEL

### Prepare data
if [ "$prepare_data" = true ] ; then
    bash $SCRIPT_DIR/prepare_noised_data.sh $CNNDM_PATH $noise_type $noise_count "$noise_prob"
else 
    echo " * Skipping data preparation. * "
fi

### Train
if [ "$train_model" = true ] ; then
    bash $SCRIPT_DIR/train-seq2seq.sh noisy clean $CUDA_DEV $prepare_data $MODEL
else 
    echo " * Skipping training. * "
fi

if [ ! -f $MODEL_PATH/checkpoint_best.pt ]; then
    echo "ERROR: Can't find the model checkpoint $MODEL_PATH/checkpoint_best.pt. Training may have failed."
    exit 1 
fi

### Generate outputs for each summarization system and evaluate using ROUGE
bash $SCRIPT_DIR/../generate/generate_all_baselines.sh
bash $SCRIPT_DIR/../eval/eval_all.sh denoise_*ord