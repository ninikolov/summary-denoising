#!/usr/bin/env bash
## Generate all the baseline outputs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Path to baselines
# Extractive
#lead3=$SCRIPT_DIR/../results/summarization-systems-outputs/lead-3.txt
#lexrank1=$SCRIPT_DIR/../results/summarization-systems-outputs/lexrank-1.txt
lexrank2=$SCRIPT_DIR/../results/summarization-systems-outputs/lexrank-2.txt
lexrank3=$SCRIPT_DIR/../results/summarization-systems-outputs/lexrank-3.txt
lexrank4=$SCRIPT_DIR/../results/summarization-systems-outputs/lexrank-4.txt
lexrank5=$SCRIPT_DIR/../results/summarization-systems-outputs/lexrank-5.txt
lexrank6=$SCRIPT_DIR/../results/summarization-systems-outputs/lexrank-6.txt
#rnnext1=$SCRIPT_DIR/../results/summarization-systems-outputs/rnn-ext-1.txt
rnnext2=$SCRIPT_DIR/../results/summarization-systems-outputs/rnn-ext-2.txt
rnnext3=$SCRIPT_DIR/../results/summarization-systems-outputs/rnn-ext-3.txt
rnnext4=$SCRIPT_DIR/../results/summarization-systems-outputs/rnn-ext-4.txt
rnnext5=$SCRIPT_DIR/../results/summarization-systems-outputs/rnn-ext-5.txt
rnnext6=$SCRIPT_DIR/../results/summarization-systems-outputs/rnn-ext-6.txt

wholearticle=$SCRIPT_DIR/../results/summarization-systems-outputs/test.article.clean

# Abstractive
birnn=$SCRIPT_DIR/../results/summarization-systems-outputs/birnn.txt
pointer_gen=$SCRIPT_DIR/../results/summarization-systems-outputs/pointer-gen.txt
pointer_gen_cov=$SCRIPT_DIR/../results/summarization-systems-outputs/pointer-gen-cov.txt
rnnrl=$SCRIPT_DIR/../results/summarization-systems-outputs/rnn-rl-rerank.txt
bottomup=$SCRIPT_DIR/../results/summarization-systems-outputs/bottom-up.txt

EXTRACTIVE_BASELINES=("$lexrank2" "$lexrank3" "$lexrank4" "$lexrank5" "$lexrank6" "$rnnext2" "$rnnext3" "$rnnext4" "$rnnext5" "$rnnext6" "$wholearticle")
ABSTRACTIVE_BASELINES=("$birnn" "$rnnrl" )
echo ""
echo "Extractive baselines: "${EXTRACTIVE_BASELINES[*]}""
echo "Abstractive baselines: "${ABSTRACTIVE_BASELINES[*]}""
echo ""

# Generate the summaries from the noisy test set
CUDA_VISIBLE_DEVICES=$CUDA_DEV python $FAIRSEQ_PATH/generate_file.py fairseq-data \
    --path $MODEL_PATH/checkpoint_best.pt --batch-size 100 --beam 5 --output_file denoise_noisy_ground_truth.txt \
    --remove-bpe --quiet --max-len-a 3


# Generate denoiosed version for each baseline
for baseline in ${EXTRACTIVE_BASELINES[*]}
do
    echo " ---> Denoising $baseline <--- "
    echo ""
    bash $SCRIPT_DIR/generate_external_baseline.sh $baseline $MODEL_PATH/checkpoint_best.pt
    echo "*********************************"
    echo ""
done

for baseline in ${ABSTRACTIVE_BASELINES[*]}
do
    echo " ---> Denoising $baseline <--- "
    echo ""
    bash $SCRIPT_DIR/generate_external_baseline.sh $baseline $MODEL_PATH/checkpoint_best.pt
    echo "*********************************"
    echo ""
done
