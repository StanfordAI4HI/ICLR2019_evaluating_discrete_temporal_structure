from munkres import Munkres, make_cost_matrix
from sklearn.metrics.cluster import contingency_matrix
from utils import *


def munkres_score(gt, pred):
    """
    :param gt: a list of lists, each containing ints
    :param pred: a list of lists, each containing ints
    :return: accuracy
    """

    # Combine all the sequences into one long sequence for both gt and pred
    gt_combined = np.concatenate(gt)
    pred_combined = np.concatenate(pred)

    # Make sure we're comparing the right shapes
    assert(gt_combined.shape == pred_combined.shape)

    # Build out the contingency matrix
    # This follows the methodology suggested by Zhou, De la Torre & Hodgkins, PAMI 2013.
    mat = contingency_matrix(gt_combined, pred_combined)

    # Make the cost matrix
    # Use the fact that no entry can exceed the total length of the sequence
    cost_mat = make_cost_matrix(mat, lambda x: gt_combined.shape[0] - x)

    # Apply the Munkres method (also called the Hungarian method) to find the optimal cluster correspondence
    m = Munkres()
    indexes = m.compute(cost_mat)

    # Pull out the associated 'costs' i.e. the cluster overlaps for the correspondences found
    cluster_overlaps = mat[list(zip(*indexes))]

    # Now compute the accuracy
    accuracy = np.sum(cluster_overlaps)/float(np.sum(mat))

    return accuracy


def repeated_structure_score(gt, pred, aligned=False, by_cluster=False, with_purity=True, substring=False):
    gt_clusters = np.unique(gt)
    pred_clusters = np.unique(pred)
    purity_dict = {x: {y: 0 for y in gt_clusters} for x in pred_clusters}

    for cg, cp in zip(gt, pred):
        purity_dict[cp][cg] += 1

    assignment_dict = {}
    for cp in purity_dict:
        max_cg = -1
        max_val = -100
        for cg in purity_dict[cp]:
            if purity_dict[cp][cg] > max_val:
                max_cg = cg
                max_val = purity_dict[cp][cg]
            assignment_dict[cp] = max_cg

    segment_dict = {y: {} for y in gt_clusters}
    prev_cg = gt[0]
    prev_cp = -1
    prev_boundary = 0
    token_list = []
    weights = []
    for i, (cg, cp) in enumerate(zip(gt, pred)):
        if cg != prev_cg:
            segment_dict[prev_cg][(prev_boundary, i - 1)] = (token_list, weights)
            prev_cg = cg
            prev_boundary = i
            token_list = []
            weights = []
            prev_cp = -1
        if cp != prev_cp:
            token_list.append(cp)
            weights.append(0)
            prev_cp = cp
        weights[-1] += 0 if assignment_dict[cp] != cg and with_purity else 1
    segment_dict[prev_cg][(prev_boundary, i)] = (token_list, weights)

    # Compute the metric
    metric = 0
    normalizer = 0
    per_gt_label_metrics, per_gt_label_normalizers = [], []
    for gt_cluster in gt_clusters:
        normalizer += len(segment_dict[gt_cluster]) * np.sum([b - a + 1 for a, b in segment_dict[gt_cluster]])
        per_gt_label_normalizers.append(len(segment_dict[gt_cluster]) * np.sum([b - a + 1 for a, b in segment_dict[gt_cluster]]))
        this_metric = 0.
        for s1, w1 in segment_dict[gt_cluster].values():
            for s2, w2 in segment_dict[gt_cluster].values():
                if aligned and not substring:
                    score = heaviest_common_subsequence_with_alignment(s1, s2, w1, w2)
                elif not aligned and not substring:
                    score = heaviest_common_subsequence(s1, s2, w1, w2)
                elif substring:
                    score = heaviest_common_substring(s1, s2, w1, w2)
                metric += score
                this_metric += score
        per_gt_label_metrics.append(this_metric)

    normalizer *= 2.
    metric = metric / normalizer

    per_gt_label_metrics, per_gt_label_normalizers = np.array(per_gt_label_metrics), 2*np.array(per_gt_label_normalizers)
    per_gt_label_metrics = per_gt_label_metrics / per_gt_label_normalizers

    if not by_cluster:
        return metric
    else:
        return {gt_cluster: val for gt_cluster, val in zip(gt_clusters, per_gt_label_metrics)}


def compute_HSC_given_SG(gt, pred):
    segment_dict = get_segment_dict(gt, pred)

    unnormalized_score = 0.
    for cg in segment_dict:
        for a, b in segment_dict[cg]:
            segment_length = (b - a + 1)
            segment_prob = segment_length / float(len(gt))
            _, _, segment = segment_dict[cg][(a, b)]
            H_SC_given_SG = entropy(relabel_clustering(segment))

            unnormalized_score += segment_prob * H_SC_given_SG

    return unnormalized_score


def compute_HC_given_SG(gt, pred):
    segment_dict = get_segment_dict(gt, pred)

    unnormalized_score = 0.
    for cg in segment_dict:
        for a, b in segment_dict[cg]:
            segment_length = (b - a + 1)
            segment_prob = segment_length / float(len(gt))
            _, _, segment = segment_dict[cg][(a, b)]
            H_C_given_SG = entropy(segment)

            unnormalized_score += segment_prob * H_C_given_SG

    return unnormalized_score


def transition_structure_score(gt, pred):
    H_SC = entropy(relabel_clustering(pred))

    HSC_given_SG = compute_HSC_given_SG(gt, pred)

    metric = HSC_given_SG / (H_SC) if H_SC != 0 else 0.

    return max(1 - metric, 0.)


def label_agnostic_segmentation_score(gt, pred):
    H_SC = entropy(relabel_clustering(pred))
    H_SG = entropy(relabel_clustering(gt))
    H_SC_given_SG = compute_HSC_given_SG(gt, pred)
    H_SG_given_SC = compute_HSC_given_SG(pred, gt)

    metric = (H_SC_given_SG + H_SG_given_SC) / (H_SG + H_SC) if (H_SC + H_SG) != 0 else 0.

    return max(1 - metric, 0.)


def segment_completeness_score(gt, pred):
    H_C = entropy(pred)
    H_C_given_SG = compute_HC_given_SG(gt, pred)

    metric = H_C_given_SG / (H_C) if H_C != 0 else 0.

    return max(1 - metric, 0.)


def segment_homogeneity_score(gt, pred):
    return segment_completeness_score(pred, gt)


def segment_structure_score_new(gt, pred):
    H_SC = entropy(relabel_clustering(pred))
    H_SG = entropy(relabel_clustering(gt))
    H_SC_given_SG = compute_HSC_given_SG(gt, pred)
    H_SG_given_SC = compute_HSC_given_SG(pred, gt)

    H_C = entropy(pred)
    H_C_given_SG = compute_HC_given_SG(gt, pred)
    H_G = entropy(gt)
    H_G_given_SC = compute_HC_given_SG(pred, gt)

    metric = (H_SC_given_SG + H_SG_given_SC + H_C_given_SG + H_G_given_SC) / (H_SG + H_SC + H_C + H_G) if (H_SC + H_SG + H_C + H_G) != 0 else 0.

    return max(1 - metric, 0.)


def temporal_structure_score_new(gt, pred, beta=1.0, aligned=False):
    c = segment_structure_score_new(gt, pred)
    p = repeated_structure_score(gt, pred, aligned=aligned, substring=True, with_purity=True)
    return ((1 + beta) * c * p)/(beta * p + c)


#https://gist.github.com/mblondel/7337391
def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


#https://gist.github.com/mblondel/7337391
def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best
