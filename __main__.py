#!python3
"""
__main__.py

This is where things happen.
We need to:
    1. Load data
    2. Call run_model
    3. Do any final evaluations
    4. Generate the concept diagram (see old/run_model.py for how this happened)



By the way, if you name a file __main__.py, then you can give python3 the directory instead of the script.
Like, you can do
    python3 .
or, if the scripts are in a folder called "OneEmbedding",
    python3 ~/path/to/OneEmbedding
instead of
    python3 __main__.py
    python3 ~/path/to/OneEmbedding/__main__.py


"""

import sys
import random
import time
import torch
import click

from util.maybe_cuda import maybe_cuda
from util.run_model_f1_with_loss import run_model
from util.evaluate_embedding import evaluate_embedding, looks_like_is_a
from util.get_negative_samples import fast_accurate_negative_samples
from util.sgd import rng
from util.get_similar_pairs import get_similar_pairs

random.seed(522)

import matplotlib.pyplot as plt

# parsing
from util.parse_file import parse_file, unpack, unpack_subset

def find_root(words, pairs):
    # a root object should be in the hypernym position (index 1) of every other word
    counts = {word: 0 for word in words}
    for pair in pairs:
        counts[pair[1]] += 1

    # everything except itself
    number_needed = len(words) - 1
    for word in words:
        if counts[word] >= number_needed:
            return word

    # there might not be a root
    return None


@click.command(no_args_is_help=True)
@click.argument("datafile", required=True, type=click.Path(exists=True))
@click.argument("dimension", required=True, type=int)
@click.argument("alpha", required=True, type=int)
@click.argument("beta", required=False, default=1, type=int)
@click.argument("gamma", required=False, default=1, type=int)
@click.argument("learning_rate", required=True, type=float)
@click.argument("learning_bias", required=True, type=float)
@click.argument("neg_samp_multiplier", required=False, default=1, type=int)
@click.option("--reconstruction", is_flag=True, help="reconstruct a lattice only (no validation or test)")
@click.option("--split", default=0, type=int, help="size of train split")
@click.option("--closed", default=True, is_flag=True, help="if true, assume dataset is already transitive closure")
@click.option("--random-init", default=False, is_flag=True, help="if true, initialize randomly instead of to all 0's")
@click.option("--stop-width", default=50, type=int)
@click.option("--iterations", default=500, type=int)
@click.option("--batchsize", default=0, type=int, help="batchsize must be greater than or equal to 500")
@click.option("--verbose", default=1, type=float)
@click.option("--train", default=None, type=click.Path(exists=True), required=False)
@click.option("--val", default=None, type=click.Path(exists=True), required=False)
@click.option("--val-neg", default=None, type=click.Path(exists=True), required=False)
@click.option("--test", default=None, type=click.Path(exists=True), required=False)
@click.option("--test-neg", default=None, type=click.Path(exists=True), required=False)
@click.option("--run-full-adj", default=False, type=bool, required=False, help="if true, evaluate on full adjacency matrix afterward")
def main( datafile, dimension, alpha, beta, gamma, learning_rate, learning_bias, neg_samp_multiplier, reconstruction, split, closed, random_init, stop_width, iterations, batchsize, verbose, train, val, val_neg, test, test_neg, run_full_adj ):
    # print current time in green
    print("\u001b[32;1m")
    print("STARTING at time:", time.asctime())
    print("\u001b[m")
    start_time = time.time()

    if (not reconstruction) and (split == 0) and (test is None):
        print("Error: Must pass --reconstruction for reconstruction, or --split <number> or --train, --val, --val-neg, --test, --test-neg for prediction")
        exit(1)

    if reconstruction and (split > 0):
        print("Error: Cannot use both reconstruction and split options")
        exit(1)

    if (train is None) and any(d is not None for d in (val, val_neg, test, test_neg)):
        print("Error: Passing --train requires --val, --val-neg, --test, --test-neg as well")
        print("(remove --train to generate train/val/test automatically)")
        exit(1)
    if (train is not None) and any(d is None for d in (val, val_neg, test, test_neg)):
        print("Error: Passing --train, --val, --val-neg, --test, or --test-neg requires all 5")
        print("(remove all five to generate train/val/test automatically)")
        exit(1)

    # Automatically generate train/val/test if train is not provided
    auto_tvt = (train is None)
    if not auto_tvt:
        reconstruction = False


    """
    Step 1: Load Data
    """
    pairs = parse_file(datafile)
    # use sets to clean out duplicates, but convert to list at the end
    (words, num_words, word_to_index, pair_numbers) = unpack(pairs)
    print("Entities:", num_words)
    # Find the root of the hierarchy, if there is one
    root_object = find_root(words, pairs)
    if verbose >= 1:
        if root_object is not None:
            print("The root is:", root_object)
        else:
            print("This lattice has no root.")

    splittable_pairs = list((A,B) for (A,B) in pair_numbers if A != B) # copy it
    random.shuffle(splittable_pairs)
    num_dupes = 0

    pair_tensor = torch.tensor(splittable_pairs, dtype=torch.int64); # size: (number of pairs, 2)
    print(pair_tensor)

    print("split is", split)

    """
    Step 2: Create Train/Test Split
    """

    if split * 2 >= len(splittable_pairs):
        raise ValueError("Error: attempting to use {} pairs for testing out of {} total".format(split, len(splittable_pairs)))

    if not auto_tvt:
        full_tensor = pair_tensor
        train_pairs = torch.tensor(unpack_subset(parse_file(train), word_to_index), dtype=torch.int64, device=maybe_cuda)
        val_pairs = torch.tensor(unpack_subset(parse_file(val), word_to_index), dtype=torch.int64, device=maybe_cuda)
        val_negatives = torch.tensor(unpack_subset(parse_file(val_neg), word_to_index), dtype=torch.int64, device=maybe_cuda)
        test_pairs = torch.tensor(unpack_subset(parse_file(test), word_to_index), dtype=torch.int64, device=maybe_cuda)
        test_negatives = torch.tensor(unpack_subset(parse_file(test_neg), word_to_index), dtype=torch.int64, device=maybe_cuda)
        print("Parsing from files:")
        print("Train+ = {}, Val+ = {}, Val- = {}, Test+ = {}, Test- = {}".format(
            len(train_pairs),
            len(val_pairs), len(val_negatives),
            len(test_pairs), len(test_negatives)))
    else:
        full_tensor = pair_tensor
        train_pairs = pair_tensor if split == 0 else pair_tensor[ : -2 * split, ... ]

        val_pairs = pair_tensor[-2 * split : -split, ... ]
        val_negatives = None # run_model will create them
        test_pairs = pair_tensor[-split : , ... ]
        test_negatives = fast_accurate_negative_samples(len(words), test_pairs, set(pair_numbers), 1)
    print("training on", len(train_pairs), "pairs")
    # default to size of train pairs
    if batchsize <= 0:
        batchsize = len(train_pairs)

    # get siblings of the network, based only on train data
    train_pair_numbers = list(list(x.item() for x in r) for r in train_pairs)
    train_siblings = get_similar_pairs(train_pair_numbers)
    train_siblings = torch.tensor(train_siblings, dtype=torch.int64, device=maybe_cuda)

    """
    # For memorization
    #if split * 2 >= len(splittable_pairs):
        #raise ValueError("Error: attempting to use {} pairs for testing out of {} total".format(split, len(splittable_pairs)))

    if not auto_tvt:
        full_tensor = pair_tensor
        train_pairs = torch.tensor(unpack_subset(parse_file(train), word_to_index), dtype=torch.int64, device=maybe_cuda)
        val_pairs = torch.tensor(unpack_subset(parse_file(val), word_to_index), dtype=torch.int64, device=maybe_cuda)
        val_negatives = torch.tensor(unpack_subset(parse_file(val_neg), word_to_index), dtype=torch.int64, device=maybe_cuda)
        test_pairs = torch.tensor(unpack_subset(parse_file(test), word_to_index), dtype=torch.int64, device=maybe_cuda)
        test_negatives = torch.tensor(unpack_subset(parse_file(test_neg), word_to_index), dtype=torch.int64, device=maybe_cuda)
        print("Parsing from files:")
        print("Train+ = {}, Val+ = {}, Val- = {}, Test+ = {}, Test- = {}".format(len(train_pairs), len(val_pairs), len(val_negatives), len(test_pairs), len(test_negatives)))
    else:
        full_tensor = pair_tensor
        pairs_count = len(pair_tensor)	
        train_pairs_before_val = pair_tensor[ : int(split_percentage * pairs_count) ]
        train_pairs_before_val_len = len(train_pairs_before_val)
        train_pairs = pair_tensor if split == 0 else train_pairs_before_val[ : int(0.8*train_pairs_before_val_len)] # We split train pairs into 80-20% for train and validation data
        val_pairs = train_pairs_before_val[int(0.8*train_pairs_before_val_len): ]
        val_negatives = None # run_model will create them
        test_pairs = pair_tensor[ int(split_percentage * pairs_count)  : ]
        test_negatives = fast_accurate_negative_samples(len(words), test_pairs, set(pair_numbers), 1)

    print("training on", len(train_pairs), "pairs")
    # default to size of train pairs
    if batchsize <= 0:
        batchsize = len(train_pairs)
    """

    """
    Step 3: Call run_model
    """
    final_embeddings = run_model(words, dimension, root_object, train_pairs.to(device=maybe_cuda), train_siblings.to(device=maybe_cuda), val_pairs.to(device=maybe_cuda),
            alpha=alpha, beta=beta, gamma=gamma,
            learning_rate=learning_rate, learning_bias=learning_bias,
            neg_samp_multiplier=neg_samp_multiplier,
            start_from_zero = not random_init,
            stop_width = stop_width, max_iterations = iterations,batchsize = batchsize,
            val_negatives = val_negatives,
            verbose = verbose)

    """
    Step 4: Evaluate
    """
    # This is an O(n^2) operation that prints data to the console.
    # We skip it if there are too many words.
    if len(words) < 4096:
        true_pairs_set = set((word_to_index[w1], word_to_index[w2]) for (w1, w2) in pairs if w1 != w2)
        # O(n^2) operation incoming!

        # What does the model think the hypernyms are?
        final_embeddings_cpu = final_embeddings.to(device='cpu')
        model_pairs_set = set(
                (index1, index2)
                for index1 in range(len(words)) for index2 in range(len(words))
                # ...only look for what the model predicts is a hypernym
                if index1 != index2 and looks_like_is_a(final_embeddings_cpu[index1], final_embeddings_cpu[index2])
                )
        model_equal_set = set(
                (index1, index2)
                for index1 in range(len(words)) for index2 in range(len(words))
                if index1 != index2 and torch.all(final_embeddings_cpu[index1] == final_embeddings_cpu[index2])
                )

        # Now create the set of true and false
        true_positives = true_pairs_set.intersection(model_pairs_set)
        false_positives = (model_pairs_set | model_equal_set) - true_pairs_set
        false_negatives = true_pairs_set - model_pairs_set

        # Don't print specifics if there are too many.
        if len(words) < 200:
            print("===== FINAL RESULTS =====")
            for word_index in range(len(words)):
                print("{:15s}".format(words[word_index]), " embedded as ", final_embeddings_cpu[word_index])
            print("\u001b[32;1mTrue Positives ({}):".format(len(true_positives)))
            for pair in sorted(list(true_positives)):
                print(pair[0], words[pair[0]], "is a", pair[1], words[pair[1]])

            print("\n\u001b[31;1mFalse Positives ({}):".format(len(false_positives)))
            for pair in sorted(list(false_positives)):
                is_equal = all(final_embeddings_cpu[pair[0]] == final_embeddings_cpu[pair[1]])
                print(pair[0], words[pair[0]], "is wrongly called a", pair[1], words[pair[1]], "(same embedding)" if is_equal else "(strict hyponym)")

            print("\n\u001b[33;1mFalse Negatives ({}):".format(len(false_negatives)))
            for pair in sorted(list(false_negatives)):
                print(pair[0], words[pair[0]], "is supposed to be a", pair[1], words[pair[1]])

        # print a summary
        num_tp = len(true_positives)
        num_fn = len(false_negatives)
        num_fp = len(false_positives)
        print("\u001b[36m")
        total_possible_pairs = len(words) * (len(words) - 1)
        pos = num_tp + num_fn
        neg = total_possible_pairs - pos
        print("Overall scores: TP = {0}/{3}  FN = {1}/{3}  FP = {2}/{4}".format(num_tp, num_fn, num_fp, pos, neg))
        print("F1 score = {}, Balanced Accuracy = {}".format(
            2*num_tp / (2*num_tp + num_fp + num_fn),
            (num_tp/pos + 1 - num_fp/neg)/2))


    print("\u001b[35m")
    print("Parameters:", " ".join(sys.argv[1:]))
    print("Test Dataset:")
    (test_TP, test_FN, test_FP, _) = evaluate_embedding(
            final_embeddings.to(device='cpu'),
            test_pairs.to(device='cpu'),
            test_negatives.to(device='cpu')
            )
    print("Test TP = {}  Test FN = {}  Test FP = {}/{}".format(test_TP, test_FN, test_FP, len(test_negatives)))
    print("Test F1-score: {:.4f}".format(2*test_TP/(2*test_TP + test_FP + test_FN)))
    print("Test Accuracy: {:.4f}".format( 1 - (test_FN + test_FP) / (len(test_negatives) + len(test_pairs)) ))
    total_time = time.time() - start_time
    print("Overall Time: {:02d}h {:02d}m {:02d}s".format(int(total_time / 3600), int(total_time / 60) % 60, int(total_time) % 60))

    if run_full_adj:
        print("Now running FULL ADJACENCY MATRIX...")
        start_time = time.time()
        # Still O(n^2) but smaller constant factor due to GPU
        num_nodes = len(words)
        all_nodes = torch.arange(num_nodes, dtype=torch.int64, device=final_embeddings.device)
        all_nodes.resize_((num_nodes, 1))

        empty_tensor = torch.empty((0, 2), device=final_embeddings.device, dtype=torch.int64)

        # get the full positive scores
        (tp, fn, _, _) = evaluate_embedding(final_embeddings, pair_tensor.to(device=final_embeddings.device), empty_tensor)
        fp = 0
        tn = 0
        for node in range(num_nodes):
            # left and right
            ids = torch.cat( (
                # left: just this node
                torch.ones((num_nodes - 1, 1), dtype=torch.int64, device=final_embeddings.device) * node,
                # right: everything EXCEPT this node
                torch.cat( (all_nodes[:node], all_nodes[node+1:]), dim=0 )
            ), dim=1 )
            
            # Assume everything is negative
            # We can correct for this later
            (_, _, cur_fp, cur_tn) = evaluate_embedding(final_embeddings, empty_tensor, ids)
            fp += cur_fp
            tn += cur_tn
            if node in (1, 10, 100, 1000, 10000, 100000, 1000000):
                print("finished node {}, t={:.3g}".format(node, time.time() - start_time))

        # now we correct for it
        # false negatives are FALSE negatives, NOT true negatives
        tn -= fn
        fp -= tp

        print("\u001b[33;1m") # yellow
        print("Evaluated Full Adjacency Matrix!!")
        print()
        print("TP = {}  FN = {}  FP = {}  TN = {}".format(tp, fn, fp, tn))
        tpr = tp / (tp + fn)
        fpr = fp/(tn+fp)
        print("Balanced Accuracy: {:.4f}".format((tpr + 1-fpr) / 2))
        print("Balanced F1: {:.4f}".format(2*tpr / (1 + tpr + fpr)))
        print("Raw Accuracy: {:.4f}".format((tp+tn)/(tp+tn+fp+fn)))
        print("Raw F1: {:.4f}".format(2*tp / (2*tp + fn + fp)))
        print()
        # this is a new and interesting one
        print("Balanced P4: {:.4f}".format(
            4 * tpr * (1-fpr) / (4*tpr*(1-fpr) + (tpr+1-fpr) * (fpr+1-tpr) )
        ))
        print("Raw P4: {:.4f}".format(4*tp*tn / (4*tp*tn + (tp + tn) * (fp + fn) ) ))

        full_adj_time = time.time() - start_time
        if full_adj_time > 3600:
            print("Time taken: {}h {}m".format(int(full_adj_time / 3600), int(full_adj_time / 60) % 60 ))
        elif full_adj_time > 60:
            print("Time taken: {}m {}s".format(int(full_adj_time / 60), int(full_adj_time % 60) ))
        else:
            print("Time taken: {:.3g}s".format(full_adj_time))


    print("\u001b[m") # reset colors

    #for word_index in range(500):
        #print("{:15s}".format(words[word_index]), " embedded as (first 30 bits)", final_embeddings[word_index, :min(30, dimension)])

    #plt.show()


main()
