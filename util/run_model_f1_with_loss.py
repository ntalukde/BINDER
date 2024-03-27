"""
run_model.py

Puts it all together!
"""

import time
import math

import torch
import numpy as np
from util.sgd_with_loss import rng, initialize, get_gradient_and_loss, multithread_gradient
from util.flip_probability import flip_prob
from util.evaluate_embedding_with_loss import evaluate_embedding_with_loss
from util.get_negative_samples import get_negative_samples, fast_negative_samples, fast_accurate_negative_samples
from util.maybe_cuda import maybe_cuda

import matplotlib.pyplot as plt


def seconds():
    return time.time()


def run_model(words, dimension, root_object, train_pairs, train_siblings, val_pairs, *,
        alpha, beta, gamma, learning_rate, learning_bias, neg_samp_multiplier,
        start_from_zero = True,
        stop_width = 50, max_iterations = 100, batchsize,
        val_negatives = None,
        verbose = 1):
    """
    Step 1: Initialize the model

    "A is-a B" means "B is a hypernym of A" so B is in hypernyms_of[A]
    """

    # if we're given no validation pairs, it's just a reconstruction task
    reconstruction = (len(val_pairs) == 0)

    # Create the embeddings.
    embeddings = initialize(len(words), dimension, start_from_zero)

    train_pairs_set = frozenset((int(p[0]), int(p[1])) for p in train_pairs.to(device="cpu")) # for negative sampling, to check if a pair is positive in O(1) time
    val_pairs_set = frozenset((int(p[0]), int(p[1])) for p in val_pairs.to(device="cpu"))
    train_val_pairs_set = val_pairs_set.union(train_pairs_set)

    unknown_set = set(range(len(words))) - (
            set(p[0] for p in train_pairs_set).union(set(p[1] for p in train_pairs_set))
    )
    print("There are {} unknown entities".format(len(unknown_set)))
    for un in unknown_set:
        embeddings[un,:] = -1 # an "undefined" value

    # Negative samples for validation set
    # we can do the slow way
    # I also realized I can fool get_negative_samples into corrupting val_pairs but checking all pairs
    print("generating negative validation of size", len(train_pairs if reconstruction else val_pairs))
    if val_negatives is None:
        val_negatives = fast_accurate_negative_samples(len(words), train_pairs if reconstruction else val_pairs, train_val_pairs_set, 1)
    print("done")


    # batch size
    if batchsize <= 0 or batchsize > len(train_pairs):
        batchsize = len(train_pairs)

    f1_scores = []
    accuracies = []
    val_losses = []
    train_losses = []

    best_embedding = None
    best_loss = 1e300
    best_f1_score = 0
    #best_accuracy = 0
    best_iteration = -1
    best_tuple = ()

    """
    Step 2: Run the algorithm
    """
    finished = False
    iteration = 1
    last_print_time = None
    try:
        while not finished:
            # so if verbose is 0.1 it prints only every 10 times
            # 1e-12 avoids double printing due to float errors
            verbose_level1 = ((iteration * verbose + 1e-12) % 1) < verbose
            # Print the header information.
            if verbose_level1:
                print("----- Iteration {} -----".format(iteration))
                if last_print_time is not None:
                    print("time since last update: {:.3g}s".format(seconds() - last_print_time))
                last_print_time = seconds()

            sub_iter_len = math.ceil(len(train_pairs)/batchsize)
            train_loss = 0
            for i in range(sub_iter_len):
                if i == (sub_iter_len-1):
                    train_minibatch = train_pairs[i*batchsize : len(train_pairs)]
                else:
                    train_minibatch = train_pairs[i*batchsize : (i+1)*batchsize]
                t = seconds()
                #negative_samples = get_negative_samples(words, pairs, pairs_set, neg_samp_multiplier)
                negative_samples = fast_negative_samples(len(words), train_minibatch, neg_samp_multiplier)
                train_pairs_flip = torch.empty(train_minibatch.size(), dtype=torch.int64, device=negative_samples.device)
                train_pairs_flip[:, 0] = train_minibatch[:,1]
                train_pairs_flip[:, 1] = train_minibatch[:,0]
                negative_samples = torch.cat((negative_samples , train_pairs_flip))
                neg_sample_time = seconds() - t
                # print time
                if verbose_level1:
                    print("Negative samples: {:.3g}s".format(seconds() - t))
                # Run the SGD algorithm
                t = seconds()
                (gradient, partial_loss) = get_gradient_and_loss(embeddings, dimension, train_minibatch, negative_samples, train_siblings,
                        alpha = alpha, beta = beta, gamma = gamma,
                        use_tensors=True)
                train_loss += partial_loss
                gradient_time = seconds() - t
                if verbose >= 2:
                    for word_index in range(len(words)):
                        print("{:15s}".format(words[word_index]), " embedded as ", embeddings[word_index], " gradient is ", gradient[word_index])

                # Flip the words based on this gradient
                # I don't expect this to be very slow
                t = seconds()
                probs = flip_prob(gradient.float() * learning_rate + learning_bias)
                # sample random floats
                flip = torch.rand(probs.size(), device=maybe_cuda) < probs
                if verbose >= 2:
                    # The " ".join() prints strings separated by spaces.
                    # Here, it makes an F where we flip and _ where we don't.
                    '''
                    for word_index in range(len(words)):
                        print("{:15s}".format(words[word_index]),
                            "flips", " ".join("F" if flip[i] == 1 else "_" for i in range(len(flip))),
                            "Flip Probabilities: ", np.around(probs, 2) )
                    '''
                    pass
                # now use the exclusive or function to actually flip them
                # not sure why the trailing underscore means "in place"
                flip[embeddings < 0] = 0 # if embeddings are missing (unseen entity) don't even try to flip them
                embeddings.bitwise_xor_(flip)
                flip_time = seconds() - t
                if verbose_level1:
                    print("Times: NegativeSamples = {:.3g}s, Gradient = {:.3g}s, Flip = {:.3g}s".format(neg_sample_time, gradient_time, flip_time))

            # Evaluate the model
            # if we're doing reconstruction, use train pairs
            # if we're doing prediction, use validation pairs
            t = seconds()
            (TP, FN, FP, TN, loss_pos) = evaluate_embedding_with_loss(embeddings,
                                              train_pairs if reconstruction else val_pairs,
                                              val_negatives)
            val_loss = loss_pos * alpha + FP * beta
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0 # ?!?
            recall = TP / (TP + FN)
            neg_recall = TN / (TN + FP)
            balanced_precision = recall / (recall + 1.0 - neg_recall) if (recall + 1.0 - neg_recall) > 1e-8 else 0
            #f1_score = 2.0 / (1.0/balanced_precision + 1.0/recall) if recall > 0 else 0
            f1_score = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
            accuracy = (recall + neg_recall) / 2.0
            if verbose_level1:
            	train_template = "TRAINING:   TP = {}   FN = {}  VALIDATION:  FP = {}  (FP out of {}). Loss = {}"
            	val_template = "VALIDATION: TP = {}   FN = {}   FP = {}  (FP out of {}). Loss = {}"
            	print((train_template if reconstruction else val_template).format(TP, FN, FP, len(val_negatives), val_loss))
            	print("F1 score = {:.3f} • Accuracy = {:.3f}    (took {:.3g}s)".format(f1_score, accuracy, seconds() - t))
            accuracies.append(accuracy)
            f1_scores.append(f1_score)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            '''
            if accuracy > best_accuracy:
               best_embedding = embeddings.clone().detach()
               best_iteration = iteration
               best_accuracy = accuracy
               best_tuple = (TP, FN, FP, TN)
	    '''
            
            if val_loss < best_loss and iteration >= min(10, max_iterations - 1): # don't determine "best loss" super early
                best_embedding = embeddings.clone().detach()
                best_iteration = iteration
                best_loss = val_loss
                best_f1_score = f1_score
                best_tuple = (TP, FN, FP, TN)
            
            # determine if we are done...
            iteration += 1
            if iteration >= max_iterations:
             	finished = True
            
            if len(val_losses) >= 2*stop_width and sum(val_losses[-2*stop_width : -stop_width]) <= sum(val_losses[-stop_width:]):
                finished = True
            '''
            if len(accuracies) >= 2*stop_width and sum(accuracies[-2*stop_width : -stop_width]) >= sum(accuracies[-stop_width:]):
              	finished = True
            '''
    except KeyboardInterrupt:
        print("Interrupted!")

    print("\u001b[36;1m")
    
    '''
    if verbose >= 0:
        #print("Beginning: ", accuracies[:500])
        #print("After 500: ", accuracies[500:])
        #print(accuracies)
        print("Last: ", accuracies[-10:])
        print()
    
    '''    
    if verbose > 0:
        print("Beginning: ", f1_scores[:500])
        print("History: ", f1_scores[::25])
        print("Last: ", f1_scores[-500:])
        print()
    
    
    print("Best Validation Loss appeared on iteration {} with Loss {} and F1 score {:.4f} {}".format(best_iteration, best_loss, best_f1_score, str(best_tuple)))
    #print("Best Validation accuracy appeared on iteration {} with accuracy {:.4f} {}".format(best_iteration, best_accuracy, str(best_tuple)))
    print("\u001b[m")

    '''
    if reconstruction:
        plt.plot(f1_train_scores, label="Train")
    else:
        plt.plot(f1_val_scores, label="Validation")
        plt.legend()
        pass
    '''

    print("===== TRAINING LOSSES below =====")
    print(train_losses)
    print("===== VALIDATION LOSSES below =====")
    print(val_losses)
    # loss_log = open("losses_d_{}.log".format(time.strftime("%Y-%m-%d_%H-%M-%S")), "w")
    #for loss in losses:
        #print(loss, file=loss_log)
        #print(loss)

    # Return the embeddings so we can use them.
    return best_embedding




