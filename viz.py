from utils import get_segment_dict
import numpy as np
import platform
import matplotlib
if not platform.system() == 'Darwin':
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns

colors = ['b', 'r', 'y', 'g', 'purple']
cmap = get_cmap("jet")

colorings = []
for i, (name, hex) in enumerate(matplotlib.colors.cnames.items()):
    colorings.append(np.array(tuple(int(hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)))/255.0)
colorings = np.array(colorings)

colorings_1 = sns.color_palette("PiYG", 12)
colorings_2 = sns.color_palette("PuOr", 10)

from matplotlib.gridspec import GridSpec


# Take a subplot and clear out all tick labels and spines
def clear_labels(ax):
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


# Adds a single row of the plot
def _add_subplot(fig, gridspec, location, segments, label):
    # Create the subplot
    ax = fig.add_subplot(gridspec[location, 0])

    # Display the segments
    ax.imshow(segments, interpolation="nearest", aspect="auto")

    # Clear out all the labels
    clear_labels(ax)

    # Set up the title
    ax.set_title(label, fontsize=14)

    return ax


# Visualize temporal clusterings in a grid
def viz_temporal_clusterings(temporal_clusterings, store_path, labels=None):

    # Get the number of temporal clusterings
    num_seqs = len(temporal_clusterings)

    # Parameter for the grid we're constructing
    grid_spec_factor = num_seqs

    # Setup the figure
    fig = plt.figure(figsize=(7.5, 0.65 * grid_spec_factor))
    gs = GridSpec(grid_spec_factor, 1, height_ratios=[1. / grid_spec_factor] * grid_spec_factor)

    # Run through the temporal clusterings
    for i, tc in enumerate(temporal_clusterings):
        _add_subplot(fig, gs, i, colorings[tc][None, :, :], labels[i])

    plt.tight_layout()
    plt.savefig(store_path, bbox_inches='tight')
    plt.close()


# Visualize segments of a primary temporal clustering in the order in which they occur, and the corresponding
# predictions for these segments in other temporal clusterings.
# This is a great visualization tool to determine how well an algorithm is able to capture the segment structure
# of ground-truth.
def viz_temporal_clusterings_with_segment_spacing(primary_temporal_clustering, temporal_clusterings, store_path, labels=None):

    # Get the number of temporal clusterings
    num_seqs = len(temporal_clusterings)

    # Parameter for the grid we're constructing
    grid_spec_factor = 1 + num_seqs

    # Setup the figure
    fig = plt.figure(figsize=(7.5, 0.65 * grid_spec_factor))
    gs = GridSpec(grid_spec_factor, 1, height_ratios=[1. / grid_spec_factor] * grid_spec_factor)

    # Extend colorings to include white
    local_colorings = np.array([(1., 1., 1.)] + list(colorings))

    # Run through the temporal clusterings
    for i, tc in enumerate(temporal_clusterings):
        # Create the segment dict
        segment_dict = get_segment_dict(primary_temporal_clustering, tc)

        # Create lists that will correctly format segments by grouping labels from the primary temporal clustering and
        # spacing them out
        primary_tc_spaced_segments, tc_spaced_segments = [], []

        # Loop over the primary temporal clustering's labels
        for a, b, label in sorted([(e, f, l) for l in segment_dict for e, f in segment_dict[l]]):
            # Add in the primary temporal clustering's segment as one contiguous chunk
            primary_tc_spaced_segments.extend([label + 1] * (b - a + 1))
            # Add in the corresponding segment in the other temporal clustering
            tc_spaced_segments.extend(list(np.array(segment_dict[label][(a, b)][-1]) + 1))
            # Add some spacing before including the next segment (18 is a good visual separation)
            primary_tc_spaced_segments.extend([0] * 18)
            tc_spaced_segments.extend([0] * 18)

        _add_subplot(fig, gs, i+1, local_colorings[tc_spaced_segments][None, :, :], labels[i+1])

    _add_subplot(fig, gs, 0, local_colorings[primary_tc_spaced_segments][None, :, :], labels[0])

    plt.tight_layout()
    plt.savefig(store_path, bbox_inches='tight')
    plt.close()


# Visualize segments of a primary temporal clustering, grouped by labels and corresponding predictions
# in other temporal clusterings.
# This is a great visualization to understand whether an algorithm is being able to recover repeated structure that
# is present in the ground-truth.
def viz_temporal_clusterings_by_segments(primary_temporal_clustering, temporal_clusterings, store_path, labels=None):
    # Get the number of temporal clusterings
    num_seqs = len(temporal_clusterings)

    # Parameter for the grid we're constructing
    grid_spec_factor = 1 + num_seqs

    # Setup the figure
    fig = plt.figure(figsize=(7.5, 0.65 * grid_spec_factor))
    gs = GridSpec(grid_spec_factor, 1, height_ratios=[1. / grid_spec_factor] * grid_spec_factor)

    # Extend colorings to include white
    local_colorings = np.array([(1., 1., 1.)] + list(colorings))

    # Run through the temporal clusterings
    for i, tc in enumerate(temporal_clusterings):
        # Create the segment dict
        segment_dict = get_segment_dict(primary_temporal_clustering, tc)

        # Create lists that will correctly format segments by grouping labels from the primary temporal clustering and spacing them out
        primary_tc_spaced_segments, tc_spaced_segments = [], []

        # Loop over the primary temporal clustering's labels
        for label in segment_dict:
            # Loop over each segment for the label
            for a, b in sorted(segment_dict[label]):
                # Add in the primary temporal clustering's segment as one contiguous chunk
                primary_tc_spaced_segments.extend([label + 1] * (b - a + 1))
                # Add in the corresponding segment in the other temporal clustering
                tc_spaced_segments.extend(list(np.array(segment_dict[label][(a, b)][-1]) + 1))
                # Add some spacing before including the next segment (18 is a good visual separation)
                primary_tc_spaced_segments.extend([0] * 18)
                tc_spaced_segments.extend([0] * 18)

            # Add in a little more spacing before segments of the next label (25 is a good separation)
            primary_tc_spaced_segments.extend([0] * 25)
            tc_spaced_segments.extend([0] * 25)

        _add_subplot(fig, gs, i + 1, local_colorings[tc_spaced_segments][None, :, :], labels[i+1])

    _add_subplot(fig, gs, 0, local_colorings[primary_tc_spaced_segments][None, :, :], labels[0])

    plt.tight_layout()
    plt.savefig(store_path, bbox_inches='tight')
    plt.close()
