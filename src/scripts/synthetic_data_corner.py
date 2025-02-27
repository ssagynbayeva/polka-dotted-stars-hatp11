import starry
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pymc3.math as pmm
import pymc3_ext as pmx
import exoplanet
from starry_process import StarryProcess, MCMCInterface
from starry_process.math import cho_factor, cho_solve
import starry_process 
import theano
theano.config.gcc__cxxflags += " -fexceptions"
theano.config.on_opt_error = "raise"
theano.tensor.opt.constant_folding
theano.graph.opt.EquilibriumOptimizer
import aesara_theano_fallback.tensor as tt
from theano.tensor.slinalg import cholesky
from corner import corner
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from tqdm import tqdm
from theano.tensor.random.utils import RandomStream
import scipy.linalg as sl
import scipy.stats as ss
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import paths

from StarryStarryProcess import *
from DistributionFunctions import *

starry.config.quiet = True
np.random.seed(1)

# Set some free params & get the model
p = dict(params)

p['star.prot']['free'] = True
p['star.obl']['free'] = True
p['star.inc']['free'] = True

p['planet.porb']['free'] = True
p['planet.t0']['free'] = True
p['planet.r']['free'] = True
p['planet.inc']['free'] = True

p['gp.c']['free'] = True
p['gp.mu']['free'] = True
p['gp.sigma']['free'] = True
p['gp.r']['free'] = True
p['gp.n']['free'] = True

free = [x for x in p.keys() if p[x].get("free", False)]

samples_fromfile = az.from_netcdf(paths.data / "SSP-organized-planet-star-gp.nc")
samples = samples_fromfile.posterior.to_dataframe()
samples = np.array(samples).T
labels = samples_fromfile.posterior.to_dataframe().columns
# Find the indices of the elements in "labels" that are also present in "free"
indices_to_keep = [i for i, label in enumerate(labels) if label in free]

# Use the indices to filter the rows in "samples"
filtered_samples = samples[indices_to_keep]

free = labels[indices_to_keep]
# saving the true values in a list
truths=[params[x]['truth'] for x in free]

import pandas as pd
import seaborn as sns
df = pd.DataFrame(filtered_samples.T, columns=free)
sns.set(style='ticks')

# Create the pair grid
g = sns.PairGrid(df, diag_sharey=False)

# Map the histograms to the diagonal
g.map_diag(sns.histplot, kde=True, color='orchid')

# Map the 2D contour plots to the lower triangle
g.map_lower(sns.kdeplot, cmap='plasma', levels=6, fill=True)

# Custom function to remove upper triangle
def remove_upper(*args, **kwargs):
    plt.gca().axis('off')

# Remove the scatter plots and empty axes in the upper triangle
g.map_upper(remove_upper)

for i, var in enumerate(free):
    truth = params[var]['truth']
    g.axes[i, i].axvline(truth, color='k', linestyle='-')
    g.axes[i, i].axhline(truth, color='k', linestyle='-')

    std = np.std(df[var])
    mean = np.mean(df[var])
    sigma_plus = mean + 2 * std
    sigma_minus = mean - 2 * std
    g.axes[i, i].axvline(sigma_plus, color='k', linestyle='--')
    g.axes[i, i].axvline(sigma_minus, color='k', linestyle='--')

# Add truth values as lines to the contour plots
for i, y_var in enumerate(free):
    for j, x_var in enumerate(free):
        if j < i:
            truth_x = params[x_var]['truth']
            truth_y = params[y_var]['truth']
            g.axes[i, j].axvline(truth_x, color='k', linestyle='-')
            g.axes[i, j].axhline(truth_y, color='k', linestyle='-')

plt.savefig(paths.figures / "SyntheticDataCorner.pdf", bbox_inches="tight", dpi=300)
