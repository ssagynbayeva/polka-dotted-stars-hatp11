import numpy as np
import theano.tensor as tt
from theano.tensor.random.utils import RandomStream

def _cho_solve(L, b):
    """Utility method for solving `A @ x = b` for `x` when `A = L @ L.T` efficiently."""
    return tt.slinalg.solve_upper_triangular(L.T, tt.slinalg.solve_lower_triangular(L, b))

class StarryStarryProcess(object):
    """Combines a STARRY Keplerian system with a STARRY process prior for the
    star's surface map to model the light curve of a star with a planet with a
    physical prior."""

    def __init__(self, sys, sp):
        """Instantiate a `StarryStarryProcess` object.
        
        :param sys: A `starry.System` object representing the Keplerian system.

        :param sp: A `starry_process.StarryProcess` object representing the
            STARRY process prior on the star's map.
        """
        self._sys = sys
        self._sp = sp

        self._mu = sp.mean_ylm
        self._Lambda = sp.cov_ylm

    @property
    def sys(self):
        """The associated `starry.System` object representing the Keplerian
        system"""
        return self._sys
    @property
    def sp(self):
        """The associated `starry_process.StarryProcess` object representing the
        STARRY process prior on the star's map."""
        return self._sp
    
    @property
    def mu(self):
        """The mean of the STARRY process prior on the star's map."""
        return self._mu
    @property
    def Lambda(self):
        """The covariance matrix of the STARRY process prior on the star's map."""
        return self._Lambda
    
    @property
    def primary(self):
        """The primary star in the system."""
        return self.sys.primary
    @property
    def secondary(self):
        """The secondary star in the system."""
        return self.sys.secondaries[0]

    @property
    def design_matrix(self):
        """The STARRY design matrix for the system.  This will only exist
        following a call to `_compute`, and will be overwritten on each
        successive call to `_compute`."""
        return self._M

    @property
    def logl_marginal(self):
        """The marginal log likelihood of the data given the model.  This will
        only exist following a call to `_compute`, and will be overwritten on
        each successive call to `_compute`."""
        return self._logl_marginal
    @property
    def a(self):
        """The conditional mean map of the star.  This will only exist following
        a call to `_compute`, and will be overwritten on each successive call to
        `_compute`."""
        return self._a
    @property
    def AInv_chol(self):
        """The Cholesky decomposition of the inverse of the conditional
        covariance of the star's map.  This will only exist following a call to
        `_compute`, and will be overwritten on each successive call to
        `_compute`."""
        return self._AInv_chol

    def _compute(self, t, flux, sigma_flux):
        """Utility method that marginalizes over stellar maps and stores the
        marginal likelihood and mean/covariance of maps.  Called by
        `marginal_likelihood` and `sample_ylm_conditional`."""
        # M = self.sys.design_matrix(t)[:,:-1] # We don't use any flux from the secondary, so [:, :-1]
        # M = self._M
        theta = (360 * t / self.sys.bodies[0].prot) % 360
        M = self.sys.bodies[0].map.design_matrix(
            xo = self.sys.position(t)[0][1,:],
            yo = self.sys.position(t)[1][1,:],
            zo = self.sys.position(t)[2][1,:],
            ro = self.sys.bodies[1].r,
            theta = theta
            )

        mu = self.mu
        Lambda = self.Lambda

        nlm = mu.shape[0]
        nt = M.shape[0]

        # Note that these quantities only refer to the lower corner of Lambda
        # (i.e. we ignore the (0,0) component)
        Lambda_chol = tt.slinalg.cholesky(Lambda[1:,1:])
        logdet_Lambda = 2.0*tt.sum(tt.log(tt.diag(Lambda_chol)))

        logdet_C = 2.0*tt.sum(tt.log(sigma_flux))

        CinvM = M / (tt.square(sigma_flux)[:, None])
        MTCinvM = tt.dot(M.T, CinvM)

        AInv = MTCinvM
        AInv = tt.set_subtensor(AInv[1:,1:], AInv[1:,1:] + _cho_solve(Lambda_chol, tt.eye(nlm-1)))
        AInv_chol = tt.slinalg.cholesky(AInv)

        Lmu_term = tt.set_subtensor(tt.zeros_like(mu)[1:], _cho_solve(Lambda_chol, mu[1:]))
        a = _cho_solve(AInv_chol, Lmu_term + tt.dot(M.T, flux / tt.square(sigma_flux)))

        logdet_A = -2.0*tt.sum(tt.log(tt.diag(AInv_chol)))

        # This bears some explanation: here we are keeping only the finite part
        # of logdet_B; in principle, logdet_B -> infinity as Lambda_{00} goes to
        # infinity.  But we don't care about the infinite part (it only reflects
        # the improper flat prior we are putting on the constant flux term),
        # only the finite part, which is captured here.
        logdet_B = logdet_C + logdet_Lambda - logdet_A

        b = tt.dot(M, mu)
        r = flux - b
        Cinvr = r / tt.square(sigma_flux)
        MTCinvr = tt.dot(M.T, Cinvr)
        AMTCinvr = _cho_solve(AInv_chol, MTCinvr)
        rproj = tt.dot(M, AMTCinvr)
        chi2 = tt.sum(r * (r - rproj)/tt.square(sigma_flux))
        
        # nt-1 because of the flat prior on the constant term "consumes" one DOF.
        logl = -0.5*chi2 - 0.5*logdet_B - 0.5*(nt-1)*tt.log(2*np.pi)

        self._M = M
        self._a = a
        self._AInv_chol = AInv_chol
        self._logl_marginal = logl

        self.chi2 = chi2
        self.logdet_B = logdet_B

    def marginal_likelihood(self, t, flux, sigma_flux):
        """Compute the marginal likelihood of the data given the starry and
        starry process models, marginalizing over stellar maps.
        
        :param t: The times of the observations.

        :param flux: The fluxes of the observations.

        :param sigma_flux: The uncertainties on the fluxes of the observations.

        :return: `None`.  Stores the marginal likelihood in `self.logl_marginal`.
        """
        self._compute(t, flux, sigma_flux)
        return self.logl_marginal
    
    def sample_ylm_conditional(self, t, flux, sigma_flux, size=1, rng=None):
        """Sample the conditional distribution of the star's spherical harmonic
        map coefficients given observations and starry and starry process
        parameters.  
        
        :param t: The times of the observations.
        
        :param flux: The fluxes of the observations.

        :param sigma_flux: The uncertainties on the fluxes of the observations.

        :param size: The number of map samples to draw; returned shape will be
            `(size, n_Ylm)`.

        :param rng: A `theano.tensor.random.utils.RandomStream` object to use
            for random numbers.  If `None`, a fresh generator will be created
            and initialized with a random seed.

        :return alm: The samples of the star's spherical harmonic map
            coefficients, of shape `(size, n_Ylm)`.
        """
        if rng is None:
            rng = RandomStream(seed=np.random.randint(1<<32))

        self._compute(t, flux, sigma_flux)
        nylm = self.a.shape[0]
        return self.a[None,:] + tt.slinalg.solve_upper_triangular(self.AInv_chol.T, rng.normal(size=(nylm, size))).T