import numpy as np
from collections import OrderedDict
from munkres import Munkres, make_cost_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder


def entropy(labels):
    """Calculates the entropy for a labeling."""
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))


def get_segment_dict(gt, pred):
    gt_clusters = np.unique(gt)
    segment_dict = OrderedDict({y: OrderedDict() for y in gt_clusters})
    prev_cg = gt[0]
    prev_cp = -1
    prev_boundary = 0
    token_list = []
    weights = []
    segment = []
    for i, (cg, cp) in enumerate(zip(gt, pred)):
        if cg != prev_cg:
            segment_dict[prev_cg][(prev_boundary, i - 1)] = (token_list, weights, segment)
            prev_cg = cg
            prev_boundary = i
            token_list = []
            weights = []
            segment = []
            prev_cp = -1
        if cp != prev_cp:
            token_list.append(cp)
            weights.append(0)
            prev_cp = cp
        segment.append(cp)
    segment_dict[prev_cg][(prev_boundary, i)] = (token_list, weights, segment)
    return segment_dict


def relabel_clustering(temporal_clustering):
    relabeled = []
    last_item = temporal_clustering[0]
    counter = 0
    for item in temporal_clustering:
        if item != last_item:
            counter += 1
        relabeled.append(counter)
        last_item = item
    return relabeled


# Relabel the cluster labels in the secondary clustering based on the correspondences with the primary clustering.
# The primary use-case is ease of visualization.
def relabel_clustering_with_munkres_correspondences(primary_temporal_clustering, secondary_temporal_clustering):
    # First make sure we relabel everything with 0-indexed, continuous cluster labels
    le = LabelEncoder()
    secondary_temporal_clustering = le.fit_transform(secondary_temporal_clustering)

    # Build out the contingency matrix
    mat = contingency_matrix(primary_temporal_clustering, secondary_temporal_clustering)

    # Create the cost matrix
    cost_mat = make_cost_matrix(mat, lambda x: len(primary_temporal_clustering) - x)

    # Apply the Hungarian method to find the optimal cluster correspondence
    m = Munkres()
    indexes = m.compute(cost_mat)

    # Create the correspondences between secondary clusters and primary labels
    correspondences = {b: a for a, b in indexes}

    # What're the labels in the primary and secondary clusterings
    primary_labels, secondary_labels = set(np.unique(primary_temporal_clustering)), set(np.unique(secondary_temporal_clustering))

    proposed_label = max(primary_labels) + 1
    for label in secondary_labels:
        if label not in correspondences:
            correspondences[label] = proposed_label
            proposed_label += 1

    # Relabel the temporal clustering
    relabeled_secondary_temporal_clustering = [correspondences[e] for e in secondary_temporal_clustering]

    return relabeled_secondary_temporal_clustering


def heaviest_common_subsequence_with_alignment(l1, l2, w1, w2):
    h1, h2 = float(sum(w1)), float(sum(w2))
    if len(l1) == 0 or len(l2) == 0 or h1 == 0 or h2 == 0:
        return 0.
    dp = np.zeros((len(l1), len(l2)))
    last_match = np.zeros((len(l1), len(l2)))
    w1_sum = 0.
    for i in range(len(l1)):
        w2_sum = 0.
        for j in range(len(l2)):
            if i == 0 and j == 0:
                emd = abs(w1[0] / h1 - w2[0] / h2) * (h1 + h2)
                dp[0, 0] = 0 if l1[0] != l2[0] else max(w1[0] + w2[0] - emd, 0)
                last_match[0, 0] = 1 if l1[0] == l2[0] else 0
            elif i == 0:
                emd1 = abs(0 / h1 - w2_sum / h2) * (h1 + h2)
                emd2 = abs(w1[0] / h1 - (w2_sum + w2[j]) / h2) * (h1 + h2)
                dp[0, j] = dp[0, j - 1] if l1[0] != l2[j] else max(dp[0, j - 1], w1[0] + w2[j] - (emd1 + emd2))
                last_match[0, j] = 1 if l1[0] == l2[j] and w1[0] + w2[j] - (emd1 + emd2) >= dp[0, j - 1] else 0
            elif j == 0:
                emd1 = abs(w1_sum / h1 - 0 / h2) * (h1 + h2)
                emd2 = abs((w1_sum + w1[i]) / h1 - w2[0] / h2) * (h1 + h2)
                dp[i, 0] = dp[i - 1, 0] if l1[i] != l2[0] else max(dp[i - 1, 0], w1[i] + w2[0] - (emd1 + emd2))
                last_match[i, 0] = 1 if l1[i] == l2[0] and w1[i] + w2[0] - (emd1 + emd2) >= dp[i - 1, 0] else 0
            else:
                emd1 = abs(w1_sum / h1 - w2_sum / h2) * (h1 + h2) * (1 - last_match[i - 1, j - 1])
                emd2 = abs((w1_sum + w1[i]) / h1 - (w2_sum + w2[j]) / h2) * (h1 + h2)
                dp[i, j] = max(dp[i, j - 1], dp[i - 1, j]) if l1[i] != l2[j] else max(dp[i, j - 1], dp[i - 1, j],
                                                                                      dp[i - 1, j - 1] + w1[i] + w2[
                                                                                          j] - (emd1 + emd2))
                last_match[i, j] = 1 if l1[i] == l2[j] and dp[i - 1, j - 1] + w1[i] + w2[j] - (emd1 + emd2) == dp[
                    i, j] else 0
            w2_sum += w2[j]
        w1_sum += w1[i]

    dp = np.array(dp)

    return dp[-1, -1]


def heaviest_common_subsequence(l1, l2, w1, w2):
    if len(l1) == 0 or len(l2) == 0:
        return 0.

    dp = np.zeros((len(l1), len(l2)))
    for i in range(len(l1)):
        for j in range(len(l2)):
            if i == 0 and j == 0:
                dp[0, 0] = 0 if l1[0] != l2[0] else w1[0] + w2[0]
            elif i == 0:
                dp[0, j] = dp[0, j - 1] if l1[0] != l2[j] else max(dp[0, j - 1], w1[0] + w2[j])
            elif j == 0:
                dp[i, 0] = dp[i - 1, 0] if l1[i] != l2[0] else max(dp[i - 1, 0], w1[i] + w2[0])
            else:
                dp[i, j] = max(dp[i, j - 1], dp[i - 1, j]) if l1[i] != l2[j] else max(dp[i, j - 1], dp[i - 1, j],
                                                                                      dp[i - 1, j - 1] + w1[i] + w2[j])

    dp = np.array(dp)

    return dp[-1, -1]

def heaviest_common_substring(l1, l2, w1, w2):
    if len(l1) == 0 or len(l2) == 0:
        return 0.

    dp = np.zeros((len(l1), len(l2)))
    for i in range(len(l1)):
        for j in range(len(l2)):
            if i == 0 and j == 0:
                dp[0, 0] = 0 if l1[0] != l2[0] else w1[0] + w2[0]
            elif i == 0:
                dp[0, j] = 0 if l1[0] != l2[j] else w1[0] + w2[j]
            elif j == 0:
                dp[i, 0] = 0 if l1[i] != l2[0] else w1[i] + w2[0]
            else:
                dp[i, j] = 0 if l1[i] != l2[j] else dp[i - 1, j - 1] + w1[i] + w2[j]

    dp = np.array(dp)
    return np.max(dp)