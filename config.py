import platform

import matplotlib
import numpy as np

if not platform.system() == 'Darwin':
    matplotlib.use('agg')
from matplotlib.cm import get_cmap
import seaborn as sns

# Setup colors and colorings for plotting
colors = ['b', 'r', 'y', 'g', 'purple']
cmap = get_cmap("jet")

colorings = []
for i, (name, hex) in enumerate(matplotlib.colors.cnames.items()):
    colorings.append(np.array(tuple(int(hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))) / 255.0)

colorings = np.array(colorings)

colorings_1 = sns.color_palette("PiYG", 12)
colorings_2 = sns.color_palette("PuOr", 10)

# Map the metric shorthands to names for pretty printing
metric_lookup = {'rss_substring': 'RSS',
                 'sss_combined': 'SSS',
                 'lass': 'LASS',
                 'tss_combined': 'TSS',
                 'tss_combined-10': 'TSS-10',
                 'tss_combined-0,1': 'TSS-0.1',
                 'tss_combined-5': 'TSS-5',
                 'tss_combined-0,2': 'TSS-0.2',
                 'tss_combined-2': 'TSS-2',
                 'tss_combined-0,5': 'TSS-0.5',
                 'munkres': 'Munk',
                 'transs': 'LASS-O',
                 'transs_flip': 'LASS-U',
                 'nmi': 'NMI',
                 'homogeneity': 'Hom',
                 'completeness': 'Com',
                 'ari': 'ARI'}

# Map the method shorthands to names for pretty printing
method_lookup = {'hdphsmm': 'HDP-HSMM',
                 'hmm': 'HMM',
                 'agg': 'Agg',
                 'hdphmm': 'HDP-HMM',
                 'shdphmm': 'SHDP-HMM',
                 'slds': 'SLDS',
                 'hdpslds': 'HDP-SLDS',
                 'shdpslds': 'SHDP-SLDS',
                 'lrr_relaxed': 'LRSC',
                 'osc_relaxed': 'OSC',
                 'spatsc_relaxed': 'SpatSC',
                 'ssc_relaxed': 'SSC',
                 'tsc': 'TSC'}

# Map the dataset shorthands to names for pretty printing
dataset_lookup = {'mocap6': 'Mocap6',
                  'bees_0': 'Bees-1',
                  'bees_1': 'Bees-2',
                  'bees_2': 'Bees-3',
                  'bees_3': 'Bees-4',
                  'bees_4': 'Bees-5',
                  'bees_5': 'Bees-6'}

inverse_method_lookup = {v: k for k, v in method_lookup.items()}
