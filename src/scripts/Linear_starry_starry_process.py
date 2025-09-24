import numpy as np
import theano
import theano.tensor as tt
from theano.tensor.random.utils import RandomStream
from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def _cho_solve(L, b):
    return tt.slinalg.solve_upper_triangular(L.T, tt.slinalg.solve_lower_triangular(L, b))

class LinearInterpolatedStarryStarryProcess(object):
    def __init__(self, sys, sp, num_maps):
        self._sys = sys
        self._sp = sp
        self._num_maps = num_maps

        self._mu = sp.mean_ylm
        self._Lambda = sp.cov_ylm
    
    @property
    def num_maps(self):
        return self._num_maps

    @property
    def sys(self):
        return self._sys
    @property
    def sp(self):
        return self._sp
    
    @property
    def mu(self):
        return self._mu
    @property
    def Lambda(self):
        return self._Lambda
    
    @property
    def primary(self):
        return self.sys.primary
    @property
    def secondary(self):
        return self.sys.secondaries[0]
    
    @property
    def design_matrix(self):
        return self._M

    @property
    def logl_marginal(self):
        return self._logl_marginal
    @property
    def a(self):
        return self._a
    @property
    def AInv_chol(self):
        return self._AInv_chol
    @property
    def AInv(self):
        return self._AInv

    def _compute(self, t, flux, sigma_flux):
        num_maps = self.num_maps
        nt = len(t)
        nlm = 256

        theta = (360 * t / self.sys.bodies[0].prot) % 360
        MM = self.sys.bodies[0].map.design_matrix(
            xo = self.sys.position(t)[0][1,:],
            yo = self.sys.position(t)[1][1,:],
            zo = self.sys.position(t)[2][1,:],
            ro = self.sys.bodies[1].r,
            theta = theta
            )

        total_time = t[-1] - t[0]
        time_interval = total_time / (num_maps - 1)

        M = tt.zeros((nt, MM.shape[1] * num_maps))

        # Create sections based on actual time values
        sections = [t[0] + i * time_interval for i in range(num_maps)]


        for i, (t_lo, t_hi) in enumerate(pairwise(sections)):
            # Find indices corresponding to the current time section
            idx = (t >= t_lo) & (t < t_hi)
            if i == num_maps - 2:  # For the last section, include the last point
                idx = (t >= t_lo) & (t <= t_hi)
            
            t_section = t[idx]
            MM_section = MM[idx]
            
            phi = (t_section - t_hi) / (t_lo - t_hi)
            n = phi.shape[0]
            coeffs = tt.zeros((n, num_maps))
            coeffs = tt.set_subtensor(coeffs[:, i], phi)
            coeffs = tt.set_subtensor(coeffs[:, i + 1], 1 - phi)
            
            M = tt.set_subtensor(M[idx, :], tt.reshape(MM_section[:, None, :] * coeffs[:, :, None], (n, -1)))

        self._M = M

        # We want to enforce that the prior on the constant term in the map is
        # completely flat, so we set the first row and column of the precision
        # matrix to zero we do this manually here.  First we Cholesky decompose:
        #
        # Lambda[1:,1:] = L L^T
        #
        # Then, because of a quirk of theano (no gradients for `cho_solve!`,
        # WTF), we can compute Lambda[1:,1:]^{-1} via
        #
        # Lambda[1:,1:]^{-1} = tt.slinalg.solve_triangular(L.T, tt.slinalg.solve_triangular(L, tt.eye(nlm-1), lower=True), lower=False)
        #
        # encapsulated in our _cho_solve(...) function above

        # Also: the covariance matrix contains a "double-copy" of the starry
        # process covariance: CC = [[C, 0], [0, C]], and we only have to solve
        # one of them, and then can reconstruct the inverse as 
        # CC^{-1} = [[C^{-1}, 0], [0, C^{-1}]].
        
        logl = 0.

        mu1 = self.sp.mean_ylm
        mu2 = self.sp.mean_ylm
        Lambda1 = self.sp.cov_ylm[1:,1:]
        Lambda2 = self.sp.cov_ylm[1:,1:]

        a_matrix = []
        AInv_chol_matrix = []

        mu_full = []
        Lambda = tt.zeros((num_maps*nlm, num_maps*nlm))
        for i in range(num_maps):
            start_idx = i*nlm
            end_idx = (i+1)*nlm
            Lambda = tt.set_subtensor(Lambda[start_idx:end_idx, start_idx:end_idx], self.sp.cov_ylm)

        # Calculate time boundaries
        time_boundaries = [t[0] + i * time_interval for i in range(num_maps)]
        # time_boundaries.append(t[-1])

        for j in range(0, num_maps-1):
            start_idx = np.searchsorted(t, time_boundaries[j])
            end_idx = np.searchsorted(t, time_boundaries[j+1])
            Mnew = M[start_idx:end_idx, j*nlm:(j+2)*nlm]

            # epsilon = 1e-6
            # Lambda1 += tt.eye(Lambda1.shape[0]) * epsilon

            theano.printing.Print('iteration')(tt.as_tensor(j))

            Lambda_chol1 = tt.slinalg.cholesky(Lambda1)
            Lambda_chol2 = tt.slinalg.cholesky(Lambda2)

            logdet_Lambda1 = 2.0 * tt.sum(tt.log(tt.diag(Lambda_chol1)))
            logdet_Lambda2 = 2.0 * tt.sum(tt.log(tt.diag(Lambda_chol2)))
            logdet_Lambda = logdet_Lambda1 + logdet_Lambda2

            logdet_C = 2.0*tt.sum(tt.log(sigma_flux[start_idx:end_idx]))

            CinvM = Mnew / (tt.square(sigma_flux[start_idx:end_idx])[:, None])
            MTCinvM = tt.dot(Mnew.T, CinvM)

            AInv = MTCinvM
            if j == 0:
                AInv = tt.set_subtensor(AInv[1:nlm,1:nlm], AInv[1:nlm,1:nlm] + _cho_solve(Lambda_chol1, tt.eye(nlm-1)))
            else:
                AInv = tt.set_subtensor(AInv[:nlm,:nlm], AInv[:nlm,:nlm] + _cho_solve(Lambda_chol1, tt.eye(nlm)))
            AInv = tt.set_subtensor(AInv[nlm+1:,nlm+1:], AInv[nlm+1:,nlm+1:] + _cho_solve(Lambda_chol2, tt.eye(nlm-1)))
            AInv_chol = tt.slinalg.cholesky(AInv)

            Lmu_term = tt.zeros(2*nlm)
            if j==0:
                Lmu_term = tt.set_subtensor(Lmu_term[1:nlm], _cho_solve(Lambda_chol1, mu1[1:]))
            else:
                Lmu_term = tt.set_subtensor(Lmu_term[:nlm], _cho_solve(Lambda_chol1, mu1))
            Lmu_term = tt.set_subtensor(Lmu_term[nlm+1:], _cho_solve(Lambda_chol2, mu2[1:]))

            a = _cho_solve(AInv_chol, Lmu_term + tt.dot(Mnew.T, flux[start_idx:end_idx] / tt.square(sigma_flux[start_idx:end_idx])))

            logdet_A = -2.0*tt.sum(tt.log(tt.diag(AInv_chol)))

            logdet_B = logdet_C - logdet_A + logdet_Lambda

            mu = tt.concatenate([mu1,mu2])
            mu_full.append(mu)
            b = tt.dot(Mnew, mu)
            r = flux[start_idx:end_idx] - b
            Cinvr = r / tt.square(sigma_flux[start_idx:end_idx])
            MTCinvr = tt.dot(Mnew.T, Cinvr)
            AMTCinvr = _cho_solve(AInv_chol, MTCinvr)
            rproj = tt.dot(Mnew, AMTCinvr)
            chi2 = tt.sum(r * (r - rproj)/tt.square(sigma_flux[start_idx:end_idx]))
            
            n_points = end_idx - start_idx
            logl += -0.5*chi2 - 0.5*logdet_B - 0.5*(n_points-1)*tt.log(2*np.pi)

            a_matrix.append(a[:nlm])
            AInv_chol_matrix.append(AInv_chol[:nlm,:nlm])

            A = tt.slinalg.solve(AInv, tt.eye(2*nlm))
            mu1 = a[nlm:]
            Lambda1 = A[nlm:,nlm:]
            mu2 = self.sp.mean_ylm
            Lambda2 = self.sp.cov_ylm[1:,1:]

        n_points = nt - end_idx
        a_matrix.append(a[nlm:])
        AInv_chol_matrix.append(AInv_chol[nlm:,nlm:])

        self._a = tt.concatenate(a_matrix)
        self._AInv_chol = AInv_chol_matrix
        self._logl_marginal = logl
        self._mu = tt.concatenate(mu_full)
        # self._Lambda = Lambda

    def marginal_likelihood(self, t, flux, sigma_flux):
        self._compute(t, flux, sigma_flux)
        return self._logl_marginal
    
    def sample_ylm_conditional(self, t, flux, sigma_flux, size=1, rng=None):
        if rng is None:
            rng = RandomStream(seed=np.random.randint(1<<32))

        nylm = self.nlm

        self._compute(t, flux, sigma_flux)
        ylms = []
        for j in range(len(self.a)):
            a = self.a[j]
            AInv_chol = self.AInv_chol[j]
            ylm_cond = a[None,:] + tt.slinalg.solve_upper_triangular(AInv_chol.T, rng.normal(size=(nylm, size))).T
            ylms.append(ylm_cond)
        return tt.concatenate(ylms)