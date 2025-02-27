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

def generate(t, params, nt, error=1e-4, visualize=True):
    """
    Generate a synthetic light curve.
    
    """
    rng = np.random.default_rng(302592346730275158208684369480422136411)
    # Set up the system
    map = starry.Map(15, 2)
    map.inc = params["star.inc"]["truth"]
    map.obl = params["star.obl"]["truth"]
    map[1] = params["star.u1"]["truth"]
    map[2] = params["star.u2"]["truth"]
    star = starry.Primary(map, r=params["star.r"]["truth"], m=params["star.m"]["truth"], prot=params["star.prot"]["truth"]) 
    planet = starry.Secondary(
        starry.Map(0,0),
        inc=params["planet.inc"]["truth"],
        ecc=params["planet.ecc"]["truth"],
        Omega=params["planet.Omega"]["truth"],
        porb=params["planet.porb"]["truth"],
        t0=params["planet.t0"]["truth"],
        r=params["planet.r"]["truth"],
        m=params["planet.m"]["truth"],
        prot=1.0
    )

    sys = starry.System(star, planet)
    xo, yo, zo = sys.position(t)
    xo = xo.eval()[1]
    yo = yo.eval()[1]
    zo = zo.eval()[1]
    theta = (360 * t / params["star.prot"]["truth"]) % 360
    
    # *** Draw 1 sample from the GP
    sp = StarryProcess(
        mu=params["gp.mu"]["truth"],
        sigma=params["gp.sigma"]["truth"],
        r=params["gp.r"]["truth"],
        dr=params["gp.dr"]["truth"],
        c=params["gp.c"]["truth"],
        n=params["gp.n"]["truth"],
    )

    nt = len(t)
    ssp = StarryStarryProcess(sys, sp, nt, 256)
    
    y_true = sp.sample_ylm().eval().reshape(-1)
    y_true[0] += 1
    
    # Compute the light curve
    flux_true = sys.design_matrix(t).eval()[:, :-1] @ y_true
    sigma_flux = error*np.ones_like(flux_true)

    flux_obs = flux_true + sigma_flux*rng.normal(size=nt)

    star.map[:,:] = y_true

    
    # Visualize the system
    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        idx = (zo > 0) | (xo ** 2 + yo ** 2 > 1)
        ax[0].plot(xo[idx], yo[idx], "k.", ms=1);
        ax[0].set_aspect(1)
        ax[0].axis("off")
        map[:, :] = y_true
        map.show(ax=ax[0])
        ax[0].set_ylim(-3, 3)
        idx = (zo > 0) & (xo ** 2 + yo ** 2 < 1)
        lat = 180 / np.pi * np.arcsin(yo[idx])
        lon = 180 / np.pi * np.arcsin(xo[idx] / np.sqrt(1 - yo[idx] ** 2)) - theta[idx]
        lon = (lon + 180) % 360 - 180
        mask = np.append(np.array(lon[1:] > lon[:-1], dtype=float), 0)
        mask[mask == 1] = np.nan
        map.show(ax=ax[1], projection="rect")
        ax[1].plot(lon, lat + mask, "k-");
        plt.tight_layout()
        plt.savefig(paths.figures / "SyntheticDataMap.pdf", bbox_inches="tight", dpi=300)
    
    return flux_obs, flux_true, sigma_flux, ssp, sp, y_true, sys


# True parameters & priors
params = {
    "planet.inc": {"truth": 90.0, "dist": Planet_Inc},
    "planet.ecc": {"truth": 0.0, "dist": Uniform, "args": [0.0, 0.4]},
    "planet.Omega": {"truth": 0.0, "dist": Angle},
    "planet.porb": {"truth": 4.887802443, "dist": Period},
    "planet.t0": {"truth": 0.0, "dist": Uniform, "args": [-0.2, 0.2]},
    "planet.r": {"truth": 0.04*0.683, "dist": Logarithmic},
    "planet.m": {"truth": 1e-3*0.81, "dist": Uniform, "args": [0.0001, 0.001]},

    "star.inc": {"truth": 90, "dist": Stellar_Ang},
    "star.m": {"truth": 0.81, "dist": Uniform, "args": [0.4, 1]},
    "star.u1": {"truth": 0.4, "dist": Uniform, "args": [0.0, 0.6]},
    "star.u2": {"truth": 0.26, "dist": Uniform, "args": [0.0, 0.4]},
    "star.prot": {"truth": 30, "dist": Period},
    "star.obl": {"truth": -30.0, "dist": Stellar_Ang},
    "star.r": {"truth": 0.683, "dist": Uniform, "args": [0.1, 1]},

    "gp.r": {"truth": 20, "dist": Uniform, "args": [5.0, 25.0]},
    "gp.dr": {"truth": 5.0, "dist": Uniform, "args": [1.0, 10.0]},
    "gp.c": {"truth": 0.5, "dist": Uniform, "args": [0.01, 1]},
    "gp.n": {"truth": 1, "dist": Uniform, "args": [0, 10]},
    "gp.mu": {"truth": 30, "dist": Uniform, "args": [0.0, 80.0]},
    "gp.sigma": {"truth": 5, "dist": Uniform, "args": [1.0, 10]}
}

# Initializing the time
Ttr_half = 0.1
dt_in_transit = Ttr_half / 20.0
dt_out_transit = params['star.prot']['truth'] / 20.0

T = 3*params['star.prot']['truth']
t_in = np.arange(-T, T, dt_in_transit)
t_out = np.arange(-T, T, dt_out_transit)

t_in_transit = (t_in % params['planet.porb']['truth'] < Ttr_half) | (t_in % params['planet.porb']['truth'] > params['planet.porb']['truth']-Ttr_half)
t_out_transit = (t_out % params['planet.porb']['truth'] < Ttr_half) | (t_out % params['planet.porb']['truth'] > params['planet.porb']['truth']-Ttr_half)

t = np.sort(np.concatenate((
    t_in[t_in_transit],
    t_out[~t_out_transit],
)))
nt = len(t)

Ttotal = t[-1] - t[0]

# Setting the priors
# fraction bounds for period priors
prot_frac_bounds = min(params['star.prot']['truth']/Ttotal/2, 0.25)
porb_frac_bounds = min(params['planet.porb']['truth']/Ttotal/2, 0.25)

# Let's add this parameter to the parameter dictionary as a prior
params['star.prot']['frac_bounds'] = prot_frac_bounds
params['planet.porb']['frac_bounds'] = porb_frac_bounds

# semi-major axis
a = (params['star.m']['truth']*np.square(params['planet.porb']['truth']/365.25))**(1/3) * 215.03 # Solar radii
# impact parameter
bmax = params['star.r']['truth'] / a

# Let's add this parameter to the parameter dictionary as a prior
params['planet.inc']['bmax'] = bmax

# Get the light curve
flux_obs, flux_true, sigma_flux, ssp, sp, y_true, sys_true = generate(t, params, nt=nt, error=1e-4)

# Plot the data
fig, ax = plt.subplots(1, figsize=(16, 4))
ax.plot(t, flux_true, color='k')
ax.errorbar(t, flux_obs, yerr=sigma_flux, fmt='.', color='magenta')
ax.set_ylabel("flux [relative]", fontsize=20)
ax.set_xlabel("time [days]", fontsize=20)
ax.legend(fontsize=16);
plt.savefig(paths.figures / "SyntheticDataLightCurve.pdf", bbox_inches="tight", dpi=300)


# Set some free params 
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

