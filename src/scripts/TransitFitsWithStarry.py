import lightkurve as lk
import starry
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import pymc3 as pm
import pymc3_ext as pmx
import theano
theano.config.gcc__cxxflags += " -fexceptions"
theano.config.on_opt_error = "raise"
theano.tensor.opt.constant_folding
theano.graph.opt.EquilibriumOptimizer
import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import paths

starry.config.lazy = False
starry.config.quiet = True

search_result = lk.search_lightcurve('HAT-P-11', author='Kepler', cadence='short')

# Download all available Kepler light curves
lc_collection = search_result.download_all()

# Separate them by year
lc1 = search_result[search_result.year == 2009].download_all()
lc3 = search_result[search_result.year == 2010].download_all()
lc4 = search_result[search_result.year == 2012].download_all()
lc2 = search_result[search_result.year == 2010].download_all() 


lc1 = lc1.stitch()
lc3 = lc3.stitch()
lc4 = lc4.stitch()
lc2 = lc2.stitch()
all_lc = lc_collection.stitch()

# Get the preliminary planetary oebital period for now using lightkurve's bls
period = np.linspace(2, 20, 10000)
bls = all_lc.to_periodogram(method='bls', period=period, frequency_factor=500);
planet_period = bls.period_at_max_power
planet_t0 = bls.transit_time_at_max_power
planet_dur = bls.duration_at_max_power*1.5

# Cleaning the data
all_lc=all_lc.remove_nans()


tranmask = all_lc.create_transit_mask(transit_time=planet_t0.value, period=np.float64(planet_period), duration=np.float64(planet_dur))

all_lc = all_lc[all_lc.quality==0]

clean_mask = (all_lc.time.value >= 201) & (all_lc.time.value <=231) & (all_lc.time.value >= 847) & (all_lc.time.value <=905) & (all_lc.time.value >= 1337) & (all_lc.time.value <=1372) 

all_lc = all_lc[~clean_mask]
all_lc.flux.value[88000:129000] = np.nan
all_lc.flux.value[615000:690000] = np.nan
all_lc = all_lc.remove_nans()
clean_mask1 = (all_lc.time.value>1319) & (all_lc.time.value<1325)
all_lc = all_lc[~clean_mask1]
clean_mask2 = (all_lc.time.value>1347) & (all_lc.time.value<1353)
all_lc = all_lc[~clean_mask2]
clean_mask3 = (all_lc.time.value>331.5) & (all_lc.time.value<334)
all_lc = all_lc[~clean_mask3]
clean_mask4 = (all_lc.time.value>860) & (all_lc.time.value<880)
all_lc = all_lc[~clean_mask4]
tranmask = all_lc.create_transit_mask(transit_time=planet_t0.value, period=np.float64(planet_period), duration=np.float64(planet_dur))

# Set up the starry model to fit the transits -- no GPs here yet!
starry.config.lazy = True
with pm.Model() as model:

    # These are the variables we're solving for;
    # here we're placing wide Gaussian priors on them.
    u1 = pm.Uniform("u1", lower=0.638,upper=0.7)
    u2 = pm.Uniform("u2", lower=0.033,upper=0.064) 
    # rp = pm.Uniform("rp", lower=0.03232321,upper=0.04632)
    # mass_p = pm.Uniform("mp", lower=7.0000e-5,upper=7.5e-5) 
    # The log period; also tracking the period itself
    # logP = pm.Normal("logP", mu=np.log(np.random.uniform(4, 5)), sd=0.1, testval=np.log(4.888))
    period = pm.Uniform("porb", lower=4,upper=5, testval=4.888)
    t0 = pm.Uniform("t0", lower=124,upper=125, testval=124.85) 

    # Instantiate the star; all its parameters are assumed
    # to be known exactly
    A = starry.Primary(
        starry.Map(ydeg=0, udeg=2, amp=1.0), m=0.809, r=0.683, prot=1.0
    )
    A.map[1] = u1
    A.map[2] = u2

    # Instantiate the planet. Everything is fixed except for
    # its luminosity and the hot spot offset.
    b = starry.Secondary(
        starry.Map(ydeg=1, udeg=0, obl=0.0, amp=0),
        m=7.0257708e-5,  # mass in solar masses
        r=0.039974684,   # radius in solar radii
        inc=88.99, # orbital inclination
        porb=period,  # orbital period in days
        prot=1,
        w=-162.149,  # Argument of periastron (little omega)
        ecc=0.265,  # eccentricity
        Omega=106, 
        t0=t0, # 
    )
    
    # Instantiate the system as before
    sys = starry.System(A, b)

    # Our model for the flux
    flux_model = pm.Deterministic("flux_model", sys.flux(all_lc.time.value))

    # This is how we tell `pymc3` about our observations;
    # we are assuming they are ampally distributed about
    # the true model. This line effectively defines our
    # likelihood function.
    # pm.Normal("obs", flux_model, sd=all_lc.flux_err, observed=np.array(all_lc.flatten(mask=tranmask).flux))
    pm.Normal("obs", flux_model, sd=np.array(all_lc.flux_err), observed=np.array(all_lc.flatten(mask=tranmask).flux))

pmx.eval_in_model(flux_model, model=model)

with model:
    map_soln = pmx.optimize()


starry.config.lazy = False
starry.config.quiet = True
A = starry.Primary(starry.Map(ydeg=0, udeg=2, amp=1.0), m=0.809, r=0.683, prot=1.0)
trueu1 = 0.646
trueu2 = 0.048
A.map[1] = map_soln['u1']
A.map[2] = map_soln['u2']

b = starry.Secondary(
    starry.Map(ydeg=1, udeg=0, obl=0.0, amp=0),
    m=7.0257708e-5,  # mass in solar masses
    r=0.039974684,  # radius in solar radii
    inc=88.99, # orbital inclination
    porb=map_soln['porb'],  # orbital period in days
    prot=1,
    w=-162.149,  # Argument of periastron (little omega)
    ecc=0.265,  # eccentricity
    Omega=106, # I think it's lambda in Morris 2017
    t0=map_soln['t0'],
)


all_lc['time'].format = 'iso'

all_lc['time'].value[0].split('-')[0]

mask3 = [i.split('-')[0]=='2010' and i.split('-')[1]=='04' and i.split('-')[2].split(' ')[0]=='18' for i in all_lc['time'].value]
mask1 = [i.split('-')[0]=='2009' and i.split('-')[1]=='12' and i.split('-')[2].split(' ')[0]=='02' for i in all_lc['time'].value]
mask2 = [i.split('-')[0]=='2009' and i.split('-')[1]=='12' and i.split('-')[2].split(' ')[0]=='31' for i in all_lc['time'].value]
mask4 = [i.split('-')[0]=='2012' and i.split('-')[1]=='01' and i.split('-')[2].split(' ')[0]=='08' for i in all_lc['time'].value]

all_lc['time'].format = 'bkjd'

lc1 = all_lc[mask1]
lc2 = all_lc[mask2]
lc3 = all_lc[mask3]
lc4 = all_lc[mask4]

mask3 = np.zeros(len(lc3.flux), dtype=bool)
mask3[370:550] = 1

mask1 = np.zeros(len(lc1.flux), dtype=bool)
mask1[600:800] = 1

mask2 = np.zeros(len(lc2.flux), dtype=bool)
mask2[950:1150] = 1

sys = starry.System(A, b)

fig = plt.subplots(figsize=(6, 12), sharex=True)
cmap = plt.get_cmap("plasma")

plt.plot(lc4.fold(period=map_soln['porb'], normalize_phase=True, epoch_time=map_soln['t0']).time.value, lc4.normalize().flux, "k.", alpha=0.6, ms=3)
plt.plot(lc4.fold(period=map_soln['porb'], normalize_phase=True, epoch_time=map_soln['t0']).time.value, sys.flux(lc4.time.value), color=cmap(1/4), alpha=0.5, label="2012-01-08")

plt.plot(lc3.fold(period=map_soln['porb'], normalize_phase=True, epoch_time=map_soln['t0']).time.value, lc3.flatten(mask=mask3).flux+0.0038, "k.", alpha=0.6, ms=3)
plt.plot(lc3.fold(period=map_soln['porb'], normalize_phase=True, epoch_time=map_soln['t0']).time.value, sys.flux(lc3.time.value)+0.0038, color=cmap(2/4), alpha=0.5, label='2010-04-18')

plt.plot(lc2.fold(period=map_soln['porb'], normalize_phase=True, epoch_time=map_soln['t0']).time.value, lc2.flatten(mask=mask2).flux+0.007, "k.", alpha=0.6, ms=3)
plt.plot(lc2.fold(period=map_soln['porb'], normalize_phase=True, epoch_time=map_soln['t0']).time.value, sys.flux(lc2.time.value)+0.007, color=cmap(3/4),alpha=0.5, label='2009-12-31')

plt.plot(lc1.fold(period=map_soln['porb'], normalize_phase=True, epoch_time=map_soln['t0']).time.value, lc1.flatten(mask=mask1).flux+0.010, "k.", alpha=0.6, ms=3)
plt.plot(lc1.fold(period=map_soln['porb'], normalize_phase=True, epoch_time=map_soln['t0']-0.002).time.value, sys.flux(lc1.time.value)+0.010, color=cmap(4/4),alpha=0.5, label='2009-12-02')

plt.xlim(-0.02,0.02)
plt.xlabel("Phase")
plt.ylabel("Normalized flux");
plt.legend()
plt.savefig(paths.figures / "TransitFitsWithStarry.pdf", bbox_inches="tight", dpi=300)





