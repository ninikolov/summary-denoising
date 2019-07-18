"""Extract the summaries from the JSON files.
Optionally apply noise."""

import sys
import glob
import json
from random import shuffle
import argparse
import logging
from tqdm import *
import numpy as np
from random import randint, uniform 
from nltk import ngrams 
import copy
from summary_rewriting.summarization_systems.oracle import get_sentence_ranking

import os.path


NOISE_TYPES_IMPLEMENTED = [None, "shuffle", "ngram_shuffle", "ngram_delete",
    "replace", "replace_paraphrased", "placeholder", "extra", "extra_paraphrased", "repeat", "backtranslate", "mixture"]
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DEFAULT_NOISE_PROB = [0.15, 0.85]

parser = argparse.ArgumentParser()
parser.add_argument('-noise_type', help='The noise to apply to the data. Implemented noise types: {}'.format(
    NOISE_TYPES_IMPLEMENTED), default="shuffle")
parser.add_argument('-noise_count', help='Number of times to apply noise on an article.',
                    default=1, type=int)
parser.add_argument('-noise_prob_distr', help='Probability distribution for the number of sentences to apply noise to. '
                                              'This is a list of probabilities, where the 1st entry represents '
                                              'the probability of noising 0 sentences, the second 1 sentences, etc.'
                                              'For example, {} will generate examples with no noise with a probability '
                                              'of {}, and examples with 1 noise '
                                              'of probability {}.'.format(
                                                    DEFAULT_NOISE_PROB, DEFAULT_NOISE_PROB[0], DEFAULT_NOISE_PROB[1]),
                    default=DEFAULT_NOISE_PROB, nargs='+', type=float)
parser.add_argument('-out_clean', help='The clean output file.', required=True)
parser.add_argument('-out_noisy', help='The noisy output file.', required=True)
parser.add_argument('-source_key', help='The source key to extract, that contains the articles.', default="article")
parser.add_argument('-target_key', help='The target key to extract, that contains the summaries.', default="abstract")
parser.add_argument('-extracted_key', help='The key that contains the extracted sentences.', default="extracted")
parser.add_argument('-paraphrased_folder', help='The folder containing paraphrases.', default="paraphrased")
args = parser.parse_args()

if args.noise_type == "backtranslate":
    args.noise_count = 1 

if args.noise_type in ["None", "backtranslate"]:
    args.noise_type = None

if args.noise_type not in NOISE_TYPES_IMPLEMENTED:
    logging.error("Argument {} noise_type not in the implemented types: {}".format(args.noise_type, NOISE_TYPES_IMPLEMENTED))
    raise Exception

if sum(args.noise_prob_distr) != 1.:
    logging.error("Your noise probabilities {}, sum={} need to sum to 1.".format(
        args.noise_prob_distr, sum(args.noise_prob_distr)))
    raise Exception

probability_boundaries = []
curr_prob = 0.
for p in args.noise_prob_distr:
    probability_boundaries.append((curr_prob, curr_prob + p))
    curr_prob += p
logging.info("Sentence noise probability boundaries: {}".format(
    {"Noise {} sentences prob".format(i): b for i, b in enumerate(probability_boundaries)}))
noise_stat = []

mixture_boundaries = {
    "replace": (0., 0.33),
    "repeat": (0.33, 0.66),
    "extra": (0.66, 1.)
}


def draw_sentences_to_noise():
    """
    Determine how many sentences to noise for the current sample, using the probability boundaries.

    :return: number of sentences to be noised
    """
    draw = uniform(0, 1)
    sentences_to_noise = 0
    for i, (low, up) in enumerate(probability_boundaries):
        if low < draw <= up:
            sentences_to_noise = i
    noise_stat.append(sentences_to_noise)
    return sentences_to_noise


def shuffle_noise(summary_sents, sentences_to_noise=None):
    """
    Apply shuffle noise to the sentences. 
    
    :param summary_sents: 
    :param sentences_to_noise: 
    :return: 
    """
    max_select = len(summary_sents) - 1

    if sentences_to_noise is None:
        # Use probability boundaries to determine how many sentences to noise
        sentences_to_noise = draw_sentences_to_noise()
        if sentences_to_noise == 0:
            return copy.deepcopy(summary_sentences)

    for _ in range(sentences_to_noise):
        choice = randint(0, max_select)
        position = randint(0, max_select)
        sent1 = copy.deepcopy(summary_sents[choice])
        sent2 = copy.deepcopy(summary_sents[position])
        summary_sents[position] = sent1
        summary_sents[choice] = sent2
    return summary_sents


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def shuffle_ngram_noise(summary_string, n, noise_p): 
    n_grams = chunks(summary_string.split(" "), n)
    output = []
    for n_g in n_grams:
        draw = uniform(0, 1)
        if draw < noise_p: 
            shuffle(n_g)
        output += n_g 
    return " ".join(output)


def delete_ngram_noise(summary_string, n, noise_p): 
    output = []
    for n_g in chunks(summary_string.split(" "), n):
        draw = uniform(0, 1)
        if draw < noise_p: 
            continue 
        else: 
            output += n_g 
    return " ".join(output)


def replace_placeholder_noise(summary_string, n, noise_p):
    output = []
    for n_g in chunks(summary_string.split(" "), n):
        draw = uniform(0, 1)
        if draw < noise_p:
            output += ["<p>"]
        else:
            output += n_g
    return " ".join(output)


def ngram_shuffle(summary_sents): 
    summary_out = []
    for s in summary_sents: 
        for j, p in enumerate(args.noise_prob_distr):
            if j == 0: 
                continue  
            s = shuffle_ngram_noise(s, j + 1, p)
        summary_out.append(s)
    return summary_out


def delete_noise(summary_sents): 
    summary_out = []
    for s in summary_sents: 
        for j, p in enumerate(args.noise_prob_distr):
            if j == 0: 
                continue  
            s = delete_ngram_noise(s, j, p)
        summary_out.append(s)
    return summary_out


def placeholder_noise(summary_sents):
    summary_out = []
    for s in summary_sents:
        for j, p in enumerate(args.noise_prob_distr):
            if j == 0:
                continue
            s = replace_placeholder_noise(s, j, p)
        summary_out.append(s)
    return summary_out


def replace_sentence(extracted_input_sentences, summary_sents, input_sents, sentences_to_noise=None):
    """
    Apply replace noise to sentences. 
    
    :param extracted_input_sentences: 
    :param summary_sents: 
    :param input_sents: 
    :param sentences_to_noise: 
    :return: 
    """
    if len(extracted_input_sentences) < 1:
        return summary_sents

    if sentences_to_noise is None:
        sentences_to_noise = draw_sentences_to_noise()
        if sentences_to_noise == 0:
            return copy.deepcopy(summary_sents)

    selected = []
    for _ in range(sentences_to_noise):
        replacement_index = randint(0, len(extracted_input_sentences) - 1)

        iters = 0
        while replacement_index in selected: # Pick another one
            replacement_index = randint(0, len(extracted_input_sentences) - 1)
            iters += 1
            if iters > 5:
                break

        closer_sentence_index = extracted_input_sentences[replacement_index]
        if len(input_sents) <= closer_sentence_index:
            continue
        summary_sents[replacement_index] = input_sents[closer_sentence_index]
    return summary_sents


def extra_sentence(extracted_input_sentences, summary_sents, input_sents,
                   order_sentences=False, sentences_to_noise=None):
    """
    Add an extra sentence from the input, putting it in its natural location.

    :param extracted_input_sentences:
    :param summary_sents:
    :param input_sents:
    :param sentences_to_noise:
    :return:
    """
    if len(extracted_input_sentences) < 1:
        logging.warning("Empty article")
        return summary_sents

    if sentences_to_noise is None:
        sentences_to_noise = draw_sentences_to_noise()
        if sentences_to_noise == 0:
            return copy.deepcopy(summary_sents)

    if not order_sentences:
        selected_history = []
        for _ in range(sentences_to_noise):
            input_selection = randint(0, len(input_sents) - 1)
            summary_sents.append(input_sents[input_selection])
            selected_history.append(input_selection)
    else:
        for _ in range(sentences_to_noise):
            input_selection = randint(0, len(input_sents) - 1)
            iters = 0
            while input_selection in extracted_input_sentences:
                input_selection = randint(0, len(input_sents) - 1)
                if iters > 5:
                    break
                iters += 1
            insert_location = None
            for j, (idx, sent) in enumerate(zip(extracted_input_sentences, summary_sents)):
                if input_selection < idx:
                    insert_location = j

            if insert_location is None:
                summary_sents.append(input_sents[input_selection])
            else:
                summary_sents.insert(insert_location, input_sents[input_selection])
    return summary_sents


def repeat_sentence(summary_sents, sentences_to_noise=None):
    """
    Pick random sentences from the summary and repeat them at the end.

    :param summary_sents:
    :param sentences_to_noise:
    :return:
    """

    if sentences_to_noise is None:
        sentences_to_noise = draw_sentences_to_noise()
        if sentences_to_noise == 0:
            return copy.deepcopy(summary_sents)

    for _ in range(sentences_to_noise):
        selection = randint(0, len(summary_sents) - 1)
        summary_sents.insert(selection, summary_sents[selection])
    return summary_sents


mixture_noise_stat = {
    k: 0 for k in mixture_boundaries.keys()
}


def mixture_noise(extracted_input_sentences, summary_sents, 
        input_sents, sentences_to_noise=None):
    noise_type_draw = uniform(0, 1)
    noise_type_selected = None 
    for noise_type in mixture_boundaries.keys(): 
        low, up = mixture_boundaries[noise_type]
        if low <= noise_type_draw < up: 
            noise_type_selected = noise_type
            break 

    mixture_noise_stat[noise_type_selected] += 1 
    
    if noise_type_selected == "shuffle":
        return shuffle_noise(copy.deepcopy(summary_sents))
    elif noise_type_selected == "replace":
        return replace_sentence(
            extracted_input_sentences, copy.deepcopy(summary_sents), input_sents)
    elif noise_type_selected == "extra":
        return extra_sentence(
            extracted_input_sentences, copy.deepcopy(summary_sents), input_sents)
    elif noise_type_selected == "repeat":
        return repeat_sentence(copy.deepcopy(summary_sents))
    else:
        return copy.deepcopy(summary_sents)


if __name__ == '__main__':
    # Sort the filenames
    json_files = sorted(glob.glob("*json"))
    file_ints = [int(fname.split(".")[0]) for fname in json_files]
    sorted_idx = np.argsort(file_ints)
    sorted_json_files = [json_files[i] for i in sorted_idx]

    logging.info("Extracting data with {} noise, {} times, prob dist {}.".format(
        "no" if args.noise_type is None else args.noise_type, 
        args.noise_count, args.noise_prob_distr))

    out_clean = open(args.out_clean, "w")
    out_noisy = open(args.out_noisy, "w")

    with tqdm(desc="Process JSON files", total=len(sorted_json_files)) as pbar:
        for f in sorted_json_files:
            json_document = json.load(open(f))

            summary_sentences = [line.strip() for line in json_document[args.target_key]]

            paraphrased_file_path = "../{}/{}.dec".format(args.paraphrased_folder, f.split(".")[0])
            if args.noise_type in ["extra_paraphrased", "replace_paraphrased"] and os.path.exists(paraphrased_file_path):
                input_sentences = [line.strip() for line in open(paraphrased_file_path).readlines()]
            else:
                input_sentences = [line.strip() for line in json_document[args.source_key]]

            if len(input_sentences) < 1 or len(summary_sentences) < 1:
                continue

            if args.extracted_key in json_document.keys() and len(json_document[args.extracted_key]) > 0:
                # Use the existing extracted sentences, provided with the dataset.
                extracted_sentences = list(json_document[args.extracted_key])
            else:
                # Sometimes we don't have those (e.g. for the test file).
                # In this case, generate the sentences by matching the strings.
                extracted_sentences = []
                for sent in summary_sentences:
                    selection_order = get_sentence_ranking(input_sentences, summary_sentences)
                    for input_index in selection_order:
                        if input_index not in extracted_sentences:
                            extracted_sentences.append(input_index)
                            break

            clean_summary = " <s> ".join(summary_sentences)
            write_summary = True

            for _ in range(args.noise_count):
                if args.noise_type == "shuffle":
                    summary_sentences_out = shuffle_noise(copy.deepcopy(summary_sentences))
                elif args.noise_type == "ngram_shuffle":
                    summary_sentences_out = ngram_shuffle(copy.deepcopy(summary_sentences))
                elif args.noise_type == "ngram_delete":
                    summary_sentences_out = delete_noise(copy.deepcopy(summary_sentences))
                elif args.noise_type in ["replace", "replace_paraphrased"]:
                    summary_sentences_out = replace_sentence(
                        extracted_sentences, copy.deepcopy(summary_sentences), input_sentences)
                elif args.noise_type == "placeholder":
                    summary_sentences_out = placeholder_noise(copy.deepcopy(summary_sentences))
                elif args.noise_type in ["extra", "extra_paraphrased"]:
                    summary_sentences_out = extra_sentence(
                        extracted_sentences, copy.deepcopy(summary_sentences),
                        input_sentences, order_sentences=True)
                elif args.noise_type == "repeat":
                    summary_sentences_out = repeat_sentence(copy.deepcopy(summary_sentences))
                elif args.noise_type == "mixture": 
                    summary_sentences_out = mixture_noise(
                        extracted_sentences, copy.deepcopy(summary_sentences), input_sentences)
                else:
                    summary_sentences_out = copy.deepcopy(summary_sentences)

                if write_summary:
                    noisy_summary = " <s> ".join(summary_sentences_out)
                    out_clean.write("{}\n".format(clean_summary.strip()))
                    out_noisy.write("{}\n".format(noisy_summary.strip()))
                    write_summary = True 
            pbar.update()

    out_clean.close()
    out_noisy.close()
    logging.info("Finished processing {} files.".format(len(sorted_json_files)))
    logging.info("Noise statistics by sentence: \n{}".format(
        {
            "Summaries with {} noisy sentence".format(s): c
            for s, c in zip(*np.unique(noise_stat, return_counts=True))
        }
    ))
    if args.noise_type == "mixture": 
        logging.info("Mixture statistics by noise type: \n{}".format(
            mixture_noise_stat
        ))
