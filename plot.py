import platform
import matplotlib
if not platform.system() == 'Darwin':
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from config import *


def setup_sns(font_scale=1.):
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=font_scale)

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def scatterplot_methods_varying_beta(frame, store_path, metric_order):
    # Number of methods being plotted
    n = frame['method'].unique().shape[0]

    # Create the figure
    _ = plt.figure(figsize=(6, 3))

    frame['metric'] = pd.Categorical(frame['metric'], metric_order)

    f1 = frame.loc[frame['metric'].isin(['tss_combined'])].sort('method')['val'].values.flatten()
    vals = []
    for metric in metric_order:
        f2 = frame.loc[frame['metric'].isin([metric])].sort('method')['val'].values.flatten()
        # vals.append(weightedtau(f1, f2))
        vals.append(ndcg_score((f2 - min(f2))/(max(f2) - min(f2)), f1))
        # print(("WT", weightedtau(f1.flatten(), f2.flatten())))
        # print(("Sp", spearmanr(f1.flatten(), f2.flatten())))

    plt.plot(range(len(vals)), vals, marker='o', linewidth=5, markersize=12)

    # Fix the y axis extent and labels
    plt.ylim([0, 1.05])
    plt.yticks(fontsize=14)
    # plt.ylabel('Weighted Kendall-Tau', fontsize=16)
    plt.ylabel('NDCG', fontsize=16)

    plt.xlabel(r'$\beta$', fontsize=16)
    plt.xticks(range(len(vals)), [r'$%s$' % e for e in ['0.0', '0.1', '0.2', '0.5', r'{\bf 1.0}', '2.0', '5.0', '10.0', '\infty']], fontsize=14)

    # Add the legend
    # plt.legend(ncol=3)

    # Save the plot
    plt.savefig(store_path, bbox_inches='tight')
    plt.close()


def factor_plot_single_method(data_frame, method_path, order=None, fig_size=None, metric_order=None, x_labels=None, **kwargs):
    setup_sns(1.)
    nruns = data_frame['run'].unique().shape[0]
    fig = plt.figure(figsize=(6, 3)) if fig_size is None else plt.figure(figsize=fig_size)
    fp = sns.factorplot(x='run', y="val", col = "metric", data = data_frame, kind = "bar", legend=True, legend_out=True,
                        palette=sns.color_palette("Blues", n_colors=nruns), order=order, col_order=metric_order,
                        size=3,aspect=0.5)

    if metric_order is not None:
        for ax, title in zip(fp.axes.flat, list(metric_order)):
            ax.set_title(title)
            # Set the x-axis ticklabels
            if not x_labels:
                ax.set_xticklabels(range(1,nruns+1))
            else:
                ax.set_xticklabels(x_labels)

            # Set the label for each subplot
            ax.set_xlabel('')
            ax.set_ylim([0, 1])

    fp.set_ylabels('Score')

    # plt.ylim([0,1])
    plt.suptitle(method_lookup[method_path.split("/")[-1]],y=1.05)
    # plt.tight_layout()
    plt.legend(loc='best', ncol=2)
    plt.savefig(method_path + '_factorplot' + kwargs['extension'] + '.png',bbox_inches='tight')
    plt.close()
    setup_sns(1.)


def bar_plot_single_method(data_frame, method_path, **kwargs):
    nruns = data_frame['run'].unique().shape[0]
    fig = plt.figure(figsize=(6,3))
    ax = sns.barplot(x='metric', y="val", hue="run", data = data_frame, palette=sns.color_palette("Blues", n_colors=nruns))
    ax.set_xticks(list((np.arange(nruns)-nruns/2.)/(nruns*1.25) + 0.04) + list((np.arange(nruns)-nruns/2.)/(nruns*1.25) + 1.04))
    ax.set_xticklabels(range(1,nruns+1)*2)
    plt.ylim([0,1])
    plt.title(method_lookup[method_path.split("/")[-1]] + ' (%d hyperparameter settings)' % nruns)
    plt.ylabel('Score')
    plt.xlabel('NMI                                              TCS')
    ax.legend_.remove()
    plt.savefig(method_path + '_barplot' + kwargs['extension'] + '.png',bbox_inches='tight')
    plt.close()


def factor_plot_combined(data_frame, path, method_order=None, fig_size=None, metric_order=None, x_labels=None, **kwargs):
    print( x_labels)

    nmethods = data_frame['method'].unique().shape[0]
    aspect = 0.7 if nmethods == 2 else 0.9
    rotation = 90 if nmethods == 2 else 90
    size = 2 if nmethods == 2 else 3
    sns_size = 1. if nmethods == 2 else 1.15

    setup_sns(sns_size)
    fig = plt.figure(figsize=(4, 3))# if fig_size is None else plt.figure(figsize=fig_size)
    fp = sns.factorplot(x='method', y="val", col="metric", data=data_frame, kind="bar", legend=True, legend_out=True,
                        palette=sns.color_palette("Blues", n_colors=nmethods), order=method_order, col_order=metric_order,
                        size=size, aspect=aspect)

    fp.fig.subplots_adjust(wspace=.05, hspace=.05)

    if metric_order is not None:
        for ax, title in zip(fp.axes.flat, list(metric_order)):
            ax.set_title(title)
            # Set the x-axis ticklabels
            if x_labels is None:
                ax.set_xticklabels(range(1, nmethods + 1), rotation=30)
            else:
                ax.set_xticklabels(x_labels, rotation=rotation)

            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels([0, 0.5, 1])

            # Set the label for each subplot
            ax.set_xlabel('')
            ax.set_ylim([0, 1])

    fp.set_ylabels('Score')


    # plt.ylim([0,1])
    # plt.suptitle('Dataset: ' + dataset_lookup[path.split("/")[-2]], y=1.05)
    # plt.tight_layout()
    # plt.legend(loc='best', ncol=2)
    # fp.legend_.remove()
    plt.savefig(path + 'all_methods_factorplot' + kwargs['extension'] + '.png', bbox_inches='tight')
    plt.close()
    setup_sns(1.)


def bar_plot_combined(data_frame, path, method_list, metric_list, **kwargs):
    nmethods = data_frame['method'].unique().shape[0]
    fig = plt.figure(figsize=(6, 3))
    ax = sns.barplot(x='metric', y="val", hue="method", data=data_frame, palette=sns.color_palette("Blues", n_colors=nmethods), order=metric_list, hue_order=method_list)

    ax.set_xticks(list((np.arange(nmethods) - nmethods / 2.) / (nmethods * 1.25) + 0.06) + list((np.arange(nmethods) - nmethods / 2.) / (nmethods * 1.25) + 1.06))
    ax.set_xticklabels(list(method_list)*2, rotation=90)#range(1, nmethods + 1) * 2)
    plt.ylim([0, 1])
    print( path)
    # plt.title('Dataset: ' + dataset_lookup[path.split("/")[-2]])
    plt.ylabel('Score')
    plt.xlabel('%s                                              %s' % (metric_list[0], metric_list[1]),labelpad=10)
    ax.legend_.remove()
    plt.savefig(path + 'all_methods_barplot' + kwargs['extension'] + '.png', bbox_inches='tight')
    plt.close()


def bar_plot_combined_2(data_frame, path, **kwargs):
    plt.figure()
    # ax = sns.barplot(x='metrics', y='val', data=data_frame)
    ax = sns.barplot(x='method', y="val", hue="metric", data = data_frame)
    # ax.set(xticklabels=METRICS)
    plt.ylim([0,1])
    plt.title(path.split("/")[-1])
    plt.tight_layout()
    plt.savefig(path + 'method_wise_scores_all_methods' + kwargs['extension'] + '.png')
    plt.close()


# Generate a plot where methods are displayed in a factorplot grouped by metrics.
def factorplot_methods_grouped_by_metrics(frame, store_path, method_order=None,
                                          metric_order=None, x_labels=None, fig_size=(4, 3)):

    # Number of methods being plotted
    n = frame['method'].unique().shape[0]

    # Figure out parameters for plot
    aspect = 0.7 if n == 2 else 0.9
    rotation = 90 if n == 2 else 90
    size = 2 if n == 2 else 3
    sns_size = 1. if n == 2 else 1.15

    # Change the setup for seaborn
    setup_sns(sns_size)

    # Set up the figure
    fig = plt.figure(figsize=fig_size)

    # Generate the factorplot
    fp = sns.factorplot(x='method', y="val", col="metric",
                        data=frame, kind="bar", legend=True, legend_out=True,
                        palette=sns.color_palette("Blues", n_colors=n),
                        order=method_order, col_order=metric_order,
                        size=size, aspect=aspect)

    # Adjust the subplots for spacing
    fp.fig.subplots_adjust(wspace=.05, hspace=.05)

    if metric_order is not None:
        for ax, title in zip(fp.axes.flat, list(metric_order)):

            # Write the name of the metric for the particular subplot
            ax.set_title(metric_lookup[title], fontsize=18)

            # Set the x-axis tick labels
            if x_labels is None:
                ax.set_xticklabels(range(1, n + 1), rotation=30, fontsize=18)
            else:
                ax.set_xticklabels(x_labels, rotation=rotation, fontsize=18)

            # Set the y-axis tick labels and limits
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels([0, 0.5, 1], fontsize=18)
            ax.set_ylim([0, 1])

            # Set the label for each subplot
            ax.set_xlabel('')

    # Set the y-axis label
    fp.set_ylabels('Score', fontsize=18)

    # Save the plot
    plt.savefig(store_path, bbox_inches='tight')
    plt.close(fig)

    # Reset seaborn settings
    setup_sns(1.)