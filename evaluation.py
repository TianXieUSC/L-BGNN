import numpy as np

"""
Evaluation metrics for recommendation (information retrieval).
"""

# for each user
def precision_and_recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1

    # precision is among the ranked list, how many are ranked correctly in the top-K in ground list
    pre = hits / (1.0 * len(ranked_list))

    # recall is how many correct test cases are discovered
    rec = hits / (1.0 * len(ground_list))
    return pre, rec


# average precision;
# for each user
# mAP = (AP_1, AP_2, ... AP_N) / N; i - user
def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0


# reciprocal rank
# for each user user
def RR(ranked_list, ground_list):
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0


# NDCG score
# WRONG!
def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / np.log(rank + 1, 2)
    return dcg / idcg


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / np.log(i + 2, 2)
    return idcg
