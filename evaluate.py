import argparse
import datetime
import os
import platform
import warnings

import matplotlib

if not platform.system() == 'Darwin':
    matplotlib.use('agg')
from sklearn.metrics import homogeneity_completeness_v_measure, normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import spearmanr, weightedtau
from viz import *
from metrics import *
from plot import *
from data_loaders import *


def generate_eval_dict(gt, pred):
    # Put all the metrics values in a dictionary and return them
    eval_dict = {}
    # Compute all the traditional metrics
    eval_dict['homogeneity'], eval_dict['completeness'], eval_dict['v_measure'] = \
        homogeneity_completeness_v_measure(gt, pred)
    eval_dict['nmi'] = normalized_mutual_info_score(gt, pred)
    eval_dict['rand'] = adjusted_rand_score(gt, pred)
    eval_dict['munkres'] = munkres_score([gt], [pred])
    eval_dict['ari'] = adjusted_rand_score(gt, pred)

    # Compute all the new metrics
    eval_dict['rss_substring'] = repeated_structure_score(gt, pred, with_purity=True, substring=True)
    eval_dict['transs'] = transition_structure_score(gt, pred)
    eval_dict['transs_flip'] = transition_structure_score(pred, gt)
    eval_dict['lass'] = label_agnostic_segmentation_score(gt, pred)
    eval_dict['sss_combined'] = segment_structure_score_new(gt, pred)
    eval_dict['tss_combined'] = temporal_structure_score_new(gt, pred)
    eval_dict['tss_combined-10'] = temporal_structure_score_new(gt, pred, beta=10.)
    eval_dict['tss_combined-0,1'] = temporal_structure_score_new(gt, pred, beta=0.1)
    eval_dict['tss_combined-5'] = temporal_structure_score_new(gt, pred, beta=5.)
    eval_dict['tss_combined-0,5'] = temporal_structure_score_new(gt, pred, beta=0.5)
    eval_dict['tss_combined-2'] = temporal_structure_score_new(gt, pred, beta=2.)
    eval_dict['tss_combined-0,2'] = temporal_structure_score_new(gt, pred, beta=0.2)

    return eval_dict


# Both gt and pred are list of lists
def evaluate_a_prediction(gt, pred):
    # Concatenate ground-truth and predictions into a single sequence
    gt_combined, pred_combined = np.concatenate(gt), np.concatenate(pred)

    # Make sure they have the same shape
    assert (gt_combined.shape == pred_combined.shape)

    # Generate the evaluation results
    eval_dict = generate_eval_dict(gt_combined, pred_combined)

    return eval_dict


def read_single_run(run_path):
    # Run path needs to exist (and will always contain a temporal_clusterings.npy file)
    if os.path.exists(run_path + '/temporal_clusterings.npy'):
        # Load the temporal clusterings (each temporal_clusterings.npy file contains multiple repetitions of the method)
        # This means we have a list of repetitions, each repetition being a list of lists containing temporal clusterings
        temporal_clusterings = np.load(run_path + '/temporal_clusterings.npy')

        # We only use the first repetition of the method
        pred = temporal_clusterings[0]

        # Return it
        return pred

    # If we can't find the run then return None
    return None


def evaluate_single_run(gt, run_path, **kwargs):
    # Read in the temporal clustering for the run
    pred = read_single_run(run_path)

    # Make sure we actually got something
    if pred is not None:
        # How many clusters in the predicted temporal clustering?
        num_pred_clusters = np.unique(np.concatenate(pred).flatten()).shape[0]

        # Log and return None if we have a degenerate temporal clustering
        if num_pred_clusters == 1:
            kwargs['logger'].write('>> %s has a degenerate temporal clustering.' % run_path)
            return None, None

        # Get out the eval dict
        eval_dict = evaluate_a_prediction(gt, pred)

        # Return both the results of evaluation and the predictions
        return eval_dict, pred

    # Log and return None if we can't find the run
    kwargs['logger'].write('>> %s not found.' % run_path)
    return None, None


def restrict_eval_dict(eval_dict, relevant_metrics):
    return {m: eval_dict[m] for m in eval_dict if m in relevant_metrics}


def evaluate_single_method(gt, method_path, **kwargs):
    # Get all the runs that we did for this method
    run_paths = glob(method_path + '/*')

    # We'll store each run's information in these
    run_ids = []
    method_eval_dicts = {}
    method_preds = {}

    # Loop over each run we did
    for run_p in run_paths:
        # Get the evaluation of this run and the raw predictions
        eval_dict, pred = evaluate_single_run(gt, run_p, **kwargs)

        # Skip if we didn't actually find this run
        if pred is None:
            continue

        # Use the last 4 characters of the run's hash as the id and add the id
        run_id = run_p[-4:]
        run_ids.append(run_id)

        # Store the eval_dict for this run
        method_eval_dicts[run_id] = eval_dict

        # Store the run's predictions
        method_preds[run_id] = pred

    return run_ids, method_eval_dicts, method_preds


def analyze_single_method(gt, method_path, **kwargs):
    # Figure out the method name
    method = method_path.split("/")[-1]

    # Load and evaluate all the runs associated with this method
    run_ids, method_eval_dicts, method_preds = evaluate_single_method(gt, method_path, **kwargs)

    # Build a pandas data-frame to help with analysis
    method_frame = pd.DataFrame({'metric': kwargs['metrics'] * len(run_ids),
                                 'val': [method_eval_dicts[id][m] for id in run_ids for m in kwargs['metrics']],
                                 'run': np.repeat(run_ids, len(kwargs['metrics'])),
                                 'method': [method_lookup[method]] * len(kwargs['metrics']) * len(run_ids)})

    # If the frame is empty, return None
    if method_frame.shape[0] == 0:
        return None, None, None, None

    # Return all information related to this method
    return run_ids, method_eval_dicts, method_preds, method_frame


def evaluate_all_methods(gt, **kwargs):
    # We'll store each method's information in these
    all_run_ids = {}
    all_eval_dicts = {}
    all_preds = {}

    # Create a pandas data-frame to store the evaluation results of all methods
    evaluation_frame = pd.DataFrame({'metric': [], 'val': [], 'run': [], 'method': []})

    # Run through the methods one by one in lexicographic order
    for m in natsorted(kwargs['methods']):

        # Generate the path to the method
        p = kwargs['checkpoint_path'] + m

        # Check if it's a valid directory
        if not os.path.isdir(p):
            continue

        print("Evaluating and analyzing %s." % (m))

        # Run evaluation and analysis for the method
        run_ids, method_eval_dicts, method_preds, method_frame = analyze_single_method(gt, p, **kwargs)

        # Store all the info associated with this method
        all_run_ids[m] = run_ids
        all_eval_dicts[m] = method_eval_dicts
        all_preds[m] = method_preds

        # Add the method's frame to the evaluation frame
        evaluation_frame = evaluation_frame.append(method_frame, ignore_index=True)

    return all_run_ids, all_eval_dicts, all_preds, evaluation_frame


def viz_best_runs_across_methods(gt, frame, all_preds, method_list, **kwargs):
    # Concatenate out the ground truths
    gt = np.concatenate(gt)

    # Grab predictions that correspond to the best run for each method
    temporal_clusterings = [gt]

    # Loop over the methods
    for m in method_list:
        # Find the name of the best run for this method
        best_run = frame[frame['method'] == method_lookup[m]]['run'].unique()[0]
        # Relabel the run using the Munkres correspondeces with ground truth
        best_run_pred = relabel_clustering_with_munkres_correspondences(gt, np.concatenate(all_preds[m][best_run]))
        # Append the prediction for this run
        temporal_clusterings.append(best_run_pred)

    # Stack up the predictions to create a single (giant) matrix
    temporal_clusterings = np.vstack(temporal_clusterings)

    # Create labels corresponding to each temporal clustering
    viz_labels = ['Ground Truth'] + ["Prediction by %s" % method_lookup[m] for m in method_list]

    # Set up paths
    store_path = kwargs['plot_path'] + 'viz_best_runs_across_methods/'
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    store_path += 'best_runs_by_%s_methods_%s' % (kwargs['extension'], "_".join(method_list))

    # Do all the visualization
    viz_temporal_clusterings(temporal_clusterings, store_path + '_viz_temporal_clusterings', labels=viz_labels)
    viz_temporal_clusterings_by_segments(gt, temporal_clusterings[1:],
                                         store_path + '_viz_temporal_clusterings_by_segments', labels=viz_labels)
    viz_temporal_clusterings_with_segment_spacing(gt, temporal_clusterings[1:],
                                                  store_path + '_viz_temporal_clusterings_with_segment_spacing',
                                                  labels=viz_labels)


# Given a frame, keep only the best run for each method, as measured by metric
def select_best_run_per_method_by_metric(frame, metric):
    return frame[frame['run'].isin(
        frame[frame['val'].isin(frame[frame['metric'] == metric].groupby(['metric', 'method'])['val'].max())]['run'])]


# Given a frame and a metric such that the frame contains only the best run for each method, get the list of methods
# sorted by scores on the metric in increasing order.
def get_methods_sorted_by_best_runs_on_metric(frame, metric):
    return list(frame.loc[frame['metric'] == metric].sort_values(by=['val'], ascending=True)['method'].values)


# Given a frame and a list of metrics, restrict the frame to only include the metrics of interest.
def restrict_frame_to_metrics(frame, metrics):
    return frame.loc[frame['metric'].isin(metrics)].sort_values(by=['metric', 'val'], ascending=[True, True])


# Given a frame and a pair of metrics, such that the frame contains only the best run for each method, compare all the
# methods on both metrics in a bar plot.
def analyze_best_runs_across_methods_for_metric_pair(frame, metric_pair, **kwargs):
    # Sort all the methods based on the first metric
    method_order = get_methods_sorted_by_best_runs_on_metric(frame, metric_pair[0])

    # Restrict the frame to the metrics of interest
    restricted_frame = restrict_frame_to_metrics(frame, metric_pair)

    # Generate a bar plot to display all the methods grouped by metrics and store it
    store_path = kwargs['plot_path'] + 'analyze_best_runs_across_methods_for_metric_pair/'
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    store_path += 'best_runs_by_%s_metric_pair_%s_%s' % (kwargs['extension'], metric_pair[0], metric_pair[1])
    barplot_methods_grouped_by_metrics(restricted_frame, store_path, method_order, metric_pair)


# Given a frame and a pair of metrics, such that the frame contains only the best run for each method, compare all the
# methods on both metrics in a bar plot.
def analyze_best_runs_across_methods_for_metric(gt, frame, metric, all_preds, **kwargs):
    # Sort all the methods based on the first metric
    method_order = get_methods_sorted_by_best_runs_on_metric(frame, metric)

    # Pick out the top and bottom method
    worst_method = inverse_method_lookup[method_order[0]]
    best_method = inverse_method_lookup[method_order[-1]]

    # Restrict the frame to the metric
    restricted_frame = restrict_frame_to_metrics(frame, [metric])

    kwargs['extension'] += '_best_worst_%s' % (metric)
    viz_best_runs_across_methods(gt, frame, all_preds, [worst_method, best_method], **kwargs)


# Generate a plot where methods are displayed in a factorplot grouped by metrics
def analyze_best_runs_across_method_pairs_by_metrics(frame, metric_list, **kwargs):
    # Restrict the frame to only the metrics
    restricted_frame = restrict_frame_to_metrics(frame, metric_list)

    # Set up the store path
    store_path = kwargs['plot_path'] + 'analyze_best_runs_across_method_pairs_by_metrics/best_runs_by_%s_metrics_%s/' % \
                                       (kwargs['extension'], ("_".join(metric_list)).lower())
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # Pick out every pair of methods
    for i, m1 in enumerate(kwargs['methods']):
        for m2 in kwargs['methods'][i + 1:]:
            # Method list of the pair of methods being considered, in the order we want
            method_list = [method_lookup[m1], method_lookup[m2]]

            # Restrict the data frame to the methods in this list
            pair_frame = restricted_frame.loc[restricted_frame['method'].isin(method_list)]

            # Specify the store path
            pair_store_path = store_path + "_".join(method_list)

            # Create and store the factorplot
            factorplot_methods_grouped_by_metrics(pair_frame, pair_store_path, method_list, metric_list, method_list)


def analyze_all_methods(gt, **kwargs):
    # Load, evaluate and analyze each method individually
    print("Loading, evaluating and analyzing each method individually.")
    all_run_ids, all_eval_dicts, all_preds, evaluation_frame = evaluate_all_methods(gt, **kwargs)

    # Call methods that do analysis
    # Figure out the best runs for every method based on the tss score
    evaluation_frame_best_runs_by_tss_combined = select_best_run_per_method_by_metric(evaluation_frame, 'tss_combined')

    # Carry out all the visualization
    viz_best_runs_across_methods(gt, evaluation_frame_best_runs_by_tss_combined, all_preds, kwargs['methods'],
                                 extension='tss_combined', **kwargs)

    # Print out the evaluation matrix as a latex table
    latex_df = evaluation_frame_best_runs_by_tss_combined.drop('run', 1)
    latex_df['metric'] = latex_df['metric'].map(metric_lookup)
    latex_df['val'] = latex_df['val'].round(2)
    latex_df = latex_df.pivot_table('val', ['metric'], 'method')
    print("\n")
    print("Latex: Evaluation Matrix")
    print((latex_df.to_latex()))

    # Scatter plot on varying beta
    for metrics in [['rss_substring', 'tss_combined-0,1', 'tss_combined-0,2', 'tss_combined-0,5',
                     'tss_combined', 'tss_combined-2', 'tss_combined-5', 'tss_combined-10', 'sss_combined'],
                    ['tss_combined-0,1', 'tss_combined', 'tss_combined-10'],
                    ['tss_combined', 'nmi', 'munkres', 'ari']]:
        store_path = kwargs['plot_path'] + 'scatterplot_methods_varying_beta/'
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        store_path += 'best_runs_by_%s_metrics_%s' % ('tss_combined', "__".join(metrics))
        scatterplot_methods_varying_beta(restrict_frame_to_metrics(evaluation_frame_best_runs_by_tss_combined, metrics),
                                         store_path, metrics)

    for metric in kwargs['metrics']:
        analyze_best_runs_across_methods_for_metric(gt, evaluation_frame_best_runs_by_tss_combined, metric, all_preds,
                                                    extension='tss_combined', **kwargs)

    # For each metric pair, analyze and plot
    for metric_pair in [('nmi', 'tss_combined'), ('munkres', 'tss_combined'),
                        ('tss_combined', 'rss_substring'), ('tss_combined', 'lass'), ('tss_combined', 'sss_combined')]:
        analyze_best_runs_across_methods_for_metric_pair(evaluation_frame_best_runs_by_tss_combined, metric_pair,
                                                         extension='tss_combined', **kwargs)

    # For each metric combination we analyze and compare all pairs of methods
    for metrics in [['tss_combined', 'rss_substring', 'sss_combined', 'nmi',
                     'homogeneity', 'completeness', 'munkres', 'ari']]:
        analyze_best_runs_across_method_pairs_by_metrics(evaluation_frame_best_runs_by_tss_combined, metrics,
                                                         extension='tss_combined', **kwargs)

    return evaluation_frame_best_runs_by_tss_combined


if __name__ == '__main__':
    # Set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to use.", required=True,
                        choices=['mocap6', 'bees'] + ['bees_%d' % i for i in range(6)])
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--log_path', type=str, default='/logs/', help='Relative path to logging directory.')
    parser.add_argument('--plot_path', type=str, default='/plots/', help='Relative path to plotting directory.')

    # Parse the input args
    args = parser.parse_args()
    kwargs = vars(args)

    # List of metrics being analyzed
    kwargs['metrics'] = natsorted(metric_lookup.keys())

    # List of methods being analyzed
    kwargs['methods'] = natsorted(method_lookup.keys())

    # Set random seed
    np.random.seed(args.seed)

    # Set other options
    np.set_printoptions(precision=2, suppress=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Set up seaborn
    setup_sns()

    # Set up logging directory
    root_path = os.getcwd()
    kwargs['checkpoint_path'] = root_path + args.log_path + args.dataset + '/'
    kwargs['plot_path'] = root_path + args.plot_path + args.dataset + '/'
    kwargs['logger'] = open(kwargs['checkpoint_path'] + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'w')

    # Check if the plotting directory exists and create it if not
    if not os.path.exists(kwargs['plot_path']):
        os.makedirs(kwargs['plot_path'])

    print("Checkpoint path: %s" % kwargs['checkpoint_path'])

    # Load the dataset
    print("Loading dataset.")
    if args.dataset == 'mocap6':
        _, gt = load_mocap6_dataset()
    elif 'bees' in args.dataset:
        _, gt = load_bees_dataset()
        # Check if the dataset is a specific bees sequence
        if len(args.dataset.split("_")) == 2:
            idx = int(args.dataset.split("_")[1])
            # Restrict the dataset to be the sequence of interest
            gt = [gt[idx]]
    else:
        raise NotImplementedError
    print("Dataset loaded.")

    # Run the evaluation
    print("Analyzing all methods.")
    analyze_all_methods(gt, **kwargs)
