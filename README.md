# BINDER: BINary orDER Embedding
by a nonempty list of authors

## Packages to install

PyTorch, random, time, click, numpy, matplotlib, math, threading

## Running

The main syntax for our script is the following, which must be run from the directory containing `__main__.py`.
```bash
python3 __main__.py data/whatever_file.txt [OPTIONS] <dimension> <alpha> <beta> <gamma> <learn_rate> <learn_bias> <neg_samp_mult>
```
(Note that `__main__.py` can be replaced with `.`, denoting the current directory.)

Dimension, alpha, beta, gamma, and neg\_samp\_mult must be integers. Learn rate and bias can be floating point values.

Inside `[OPTIONS]` you **must** include exactly one of the following:
- `--reconstruction` causes the model to run a reconstruction task. There is no validation data; the model simply tries to embed the lattice. (Effectively, the task _is_ to completely fit the training data.)
- Manual dataset: pass five filenames containing the training, positive validation, negative validation, positive test, and negative test data, using `--train`, `--val`, `--val-neg`, `--test`, `--test-neg` respectively.
- `--run-full-adj yes` for running representation experiment on full adj matrix as reported in the paper.

### Examples

To run representation on Medical with 128-bit embeddings:
```bash
python3 . data/Medical_data_direct_closure.tsv.full_transitive \
	128 30 10 0 0.008 0.01 128 \
	--reconstruction --verbose 0 \
	--iterations 10000 --stop-width 9999 --run-full-adj yes
```

To run transitive closure link prediction on 50% transitive closure of WordNet nouns, with 120-bit embeddings:
```bash
python3 . data/noun_closure.tsv.full_transitive \
	--train data/noun_closure.tsv.train_50percent \
	--val data/noun_closure.tsv.valid \
	--val-neg data/noun_closure.tsv.valid_neg \
	--test data/noun_closure.tsv.test \
	--test-neg data/noun_closure.tsv.test_neg \
	--iterations 2000 --stop-width 9999 --verbose 0 \
	120 50000 10 0 0.008 0.01 12
```

### Hyper-parameter explanations
`alpha` is the positive sample weight. `beta` is the negative sample weight. Note that the formula `(alpha * pos_grad + beta * neg_grad) * LR` is used, so doubling alpha and beta has exactly the same effect as doubling the LR.

We experimented with a third loss term for "similarity", with associated weight `gamma`, which aims to move sibling nodes (two nodes with the same direct parent) closer together. We abandoned this idea after it failed to improve results, so we strongly recommend passing 0.

`neg_samp_mult` is the ratio of negative samples to positive samples during training. This does not affect validation and testing.

Other options:
- `--iterations <number>` stops after this number of iterations.
- `--stop-width <number>` controls the early-stop criterion. If, of the last `2*stop_width` iterations, the first `stop_width` have a higher average validation accuracy than the last `stop_width` iterations, the code stops.
- `--verbose <number>`: Number can be 2 or a real number between 0 and 1 inclusive. If 0, prints only at start and end of training. If 1, prints validation accuracy and training time for each iteration. If `x` strictly between 0 and 1, prints every `1.0/x` iterations. If 2, prints embedding and flipped bits each iteration.

#### Deprecated options:
- `--split <number>`: Takes a _full lattice_ dataset and automatically splits it. The number is the size of the validation (and test) split. _This version does NOT respect the transitive closure_; it instead pulls out edges at random from the training set.
- `--batchsize <number>` controls the batch size. **Defaults to the entire dataset if not present**.
- `--threads <number>`: Does nothing.

