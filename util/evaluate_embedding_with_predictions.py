"""
evaluate_embedding.py

Given an embedding, evaluates it by determining how many of the positive and negative pairs are correctly classified.

Returns (TP, FN, FP, TN).
"""

import torch

def looks_like_is_a(emb1, emb2):
    return (
        torch.all(
            (torch.bitwise_and(emb1, emb2) == emb2) &
            (emb1 >= 0) & (emb2 >= 0),
        dim=-1, keepdim=True) &
        (torch.any(emb1 != emb2, dim=-1, keepdim=True))
    )

def evaluate_embedding(embeddings, positive_is_a, negative_is_a):
    falseneg = torch.tensor([0], dtype=torch.int32, device=embeddings.device)
    falsepos = torch.tensor([0], dtype=torch.int32, device=embeddings.device)

    # This is very similar to sgd.py
    d = embeddings.size(1)

    # Process this in pseudo-minibatches to avoid excessive GPU memory usage
    # (this is NOT the minibatch parameter)
    group_size = 14000

    pos_predictions = torch.tensor(size=(positive_is_a.size(0), 1));
    neg_predictions = torch.tensor(size=(negative_is_a.size(0), 1));

    # reason:
    # 10000 pairs * 500 (maximum reasonable dimension) * 4 bytes per pos
    # = 20'000'000 bytes per array.
    # actually there are multiple copies but it's still well under 1GB
    for i in range(0, positive_is_a.size(0), group_size):
        allAs = positive_is_a[i:(i+group_size), :1].repeat(1, d)
        allBs = positive_is_a[i:(i+group_size), 1:].repeat(1, d)

        embA = torch.gather(embeddings, 0, allAs)
        embB = torch.gather(embeddings, 0, allBs)

        # one false negative for every pair where
        # A=0 and B=1 in any position
        pos_falseneg_batch = (
                torch.any((1-embA) * embB, dim=1).to(dtype=torch.int32) |
                torch.all(embA == embB, dim=1).to(dtype=torch.int32) |
                torch.any(embA == -1, dim=1).to(dtype=torch.int32) |
                torch.any(embB == -1, dim=1).to(dtype=torch.int32))  # if either pair is unseen (-1) 

        # predictions should be 1 when it is positive
        pos_predictions[i:(i+group_size)] = 1 - pos_falseneg_batch
        falseneg += torch.sum(pos_prediction_batch)

    for i in range(0, negative_is_a.size(0), group_size):
        # negative samples
        # same algorithm, just with true negatives
        allAs = negative_is_a[i:(i+group_size), :1].repeat(1, d)
        allBs = negative_is_a[i:(i+group_size), 1:].repeat(1, d)
        embA = torch.gather(embeddings, 0, allAs)
        embB = torch.gather(embeddings, 0, allBs)

        neg_falsepos_batch = (
                looks_like_is_a(embA, embB).to(dtype=torch.int32) |
                torch.any(embA == -1, dim=1, keepdim=True).to(dtype=torch.int32) |
                torch.any(embB == -1, dim=1, keepdim=True).to(dtype=torch.int32))  # if either pair is unseen (-1) 
        neg_predictions[i:(i+group_size)] = neg_falsepos_batch
        falsepos += torch.sum(neg_falsepos_batch)

    """
    for (A, B) in positive_is_a:
        # A is-a B
        if looks_like_is_a(embeddings[A], embeddings[B]):
            truepos += 1
        else:
            falseneg += 1

    for (A, B) in negative_is_a:
        # A is-not-a B
        if looks_like_is_a(embeddings[A], embeddings[B]):
            falsepos += 1
        else:
            trueneg += 1
    """

    # now parse
    truepos = len(positive_is_a) - falseneg
    trueneg = len(negative_is_a) - falsepos

    return (
            pos_predictions,
            neg_predictions,
            tuple(tensor.item() for tensor in (truepos, falseneg, falsepos, trueneg))
    )


