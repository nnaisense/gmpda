import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class FitClass:
    """
    This class is used for sigma optimization via curve_fit.
    """

    def __init__(self):
        self.mu = []

    def multi_modal(self, *args):
        """
        This is a multi-modal Gaussian with fixed mean values mu.

        :param args: contains the x value and the parameters
        :return:
        """
        idx_offset = 2
        n = int((len(args) - 1) / idx_offset)

        x = args[0]
        gauss_sum = 0
        for i in range(n):
            mu = self.mu[i]
            sigma = args[1 + (idx_offset * i)]
            a = args[2 + (idx_offset * i)]
            gauss_sum += a * np.exp(-((x - mu) ** 2.0) / 2.0 / sigma ** 2.0)
        return gauss_sum


class GMPDA:
    """
    GMPDA (Gaussian Mixture Periodicity Detection Algorithm) class for periodicity detection.
    """

    def __init__(
        self,
        ts,
        max_depth=2,
        max_periods=2,
        max_candidates=None,
        mu_range_min=1,
        mu_range_max=None,
        noise_range=None,
        sigma=None,
        sigma_curvefit=False,
        sigma_log_init=False,
        random_walk=1,
        loss_length=None,
        loss_change_tol=0.001,
        ref_loss_n=0,
        gmpda_seed=10,
        report_short=True,
    ):
        """
        :param ts: (np 1-d array)
        :param max_depth: (int)
        :param max_periods:(int)
        :param max_candidates:(int)
        :param mu_range_min:(int)
        :param mu_range_max:(int)
        :param noise_range:'NaN' or (int)
        :param sigma: 'NaN' or (int)
        :param sigma_curvefit:(boolean)
        :param sigma_log_init:(boolean)
        :param random_walk:{0,1}
        :param loss_length:(int)
        :param loss_change_tol:(float) in [0, 1)
        :param ref_loss_n:(int)
        :param gmpda_seed:(int)
        """
        np.random.seed(gmpda_seed)
        random.seed(gmpda_seed)

        self.max_depth = max_depth
        self.max_periods = max_periods
        self.max_candidates = max_candidates
        if self.max_candidates >= mu_range_max - mu_range_min:
            print(
                "Warning: The set of candidate periodicities is [mu_range_min, mu_range_max], "
                "as max_candidates >= mu_range_max - mu_range_min."
            )

        self.mu_range_max = mu_range_max
        self.mu_range_min = mu_range_min
        if mu_range_min < 5:
            self.mu_range_min = 5
            print("Require mu_range_min >= 5. Set mu_range_min = 5.")

        if not sigma:
            self.sigma = int(np.ceil(np.log(mu_range_min)))
        else:
            self.sigma = sigma

        if not noise_range or noise_range > mu_range_min:
            self.noise_range = mu_range_min
        else:
            self.noise_range = noise_range

        self.random_walk = random_walk
        self.loss_change_tol = loss_change_tol

        self.sigma_curvefit = sigma_curvefit
        self.sigma_log_init = sigma_log_init

        # init ts
        if np.shape(ts.shape)[0] < 2:
            self.ts = ts.reshape((1, ts.shape[0]))
        else:
            n, m = ts.shape
            if n > m:
                self.ts = ts.reshape((m, n))
            elif n < m:
                self.ts = ts

        self.len_ts = self.ts.shape[1]
        self.event_set = np.where((self.ts == 1))[1]

        """Account for time segment of length > mu_range_max, i.e no events"""
        self.len_es_ob = sum(np.diff(self.event_set)[np.diff(self.event_set) <= self.mu_range_max])
        assert self.len_es_ob > 0, Warning("There are no intervals <= mu_range_max in the data.")

        """The loss_length is extended, in order to slice wrt loss_length inclusively."""
        if loss_length <= self.mu_range_max + 3 * self.sigma:
            loss_length = self.mu_range_max + 3 * self.sigma
            print("Require loss_length > mu_range_max. Set loss_length = mu_range_max + 3*max(self.sigma)")
        if loss_length + 1 > self.len_ts:
            self.loss_length = self.len_ts
            print(
                f"Waring: loss_length={loss_length}, len_ts={self.len_ts}"
                f"\nRequire loss_length + 1 < len(ts). Set loss_length = len(ts)."
            )
        else:
            self.loss_length = loss_length + 1

        self.ref_loss_n = ref_loss_n
        self.report_short = report_short

    @staticmethod
    def round_down(n, decimals=6):
        """
        Deterministic rounding by truncation.

        :param n: float or array, numbers to be rounded
        :param decimals:
        :return: rounded float up to decimals
        """
        multiplier = 10 ** decimals
        if isinstance(n, np.ndarray):
            return (n * multiplier).astype("int") / multiplier
        elif isinstance(n, float):
            return int(n * multiplier) / multiplier
        elif isinstance(n, int):
            return n
        else:
            print(f"Error: round_down() for type{type(n)} is not implemented.")
            raise NotImplementedError

    def reinit_ts(self, ts_):
        """
        Reinit self configuration if a new time series is considered, this is the case for the local loss.

        :param ts_:(np 1-d array)
        :return: None
        """
        if np.shape(ts_.shape)[0] < 2:
            self.ts = ts_.reshape((1, ts_.shape[0]))
        else:
            n, m = ts_.shape
            if n > m:
                self.ts = ts_.reshape((m, n))
            elif n < m:
                self.ts = ts_

        self.len_ts = self.ts.shape[1]
        self.event_set = np.where((self.ts == 1))[1]

        # Account for time segment of length > mu_range_max, i.e no events
        self.len_es_ob = sum(np.diff(self.event_set)[np.diff(self.event_set) <= self.mu_range_max])

    def get_ref_loss(self, n=100):
        """
        Estimates the reference loss for the initialized event series.

        :param n: number of samples
        :return: array of losses
        """
        self.ref_loss_n = 0

        ts_origin = self.ts
        sc_orig = self.sigma_curvefit
        sigma_orig = self.sigma
        self.sigma_curvefit = False
        ref_loss = []
        for i in range(0, n):
            # Create a noise only time series
            ts_noise = np.zeros_like(self.ts).astype("int")
            idx_events = np.random.choice(range(max(ts_noise.shape)), int(self.ts.sum()), replace=False)
            ts_noise[:, idx_events] = 1
            self.sigma = sigma_orig

             _, _, loss_, _, _, _, _, _ = self.extract_periods(ts=ts_noise, verbose=False)
            ref_loss.append(loss_[0])

        # Set self to origin ts
        self.sigma_curvefit = sc_orig
        self.sigma = sigma_orig
        self.reinit_ts(ts_=ts_origin)

        return ref_loss

    def get_intervals(self):
        """
        This function estimates the 1st, 2nd, 3rd, ..., Kth order intervals between positive observations.

        :return: numpy ndarray (1,loss_length), frequencies
        """
        dmu = np.zeros((1, self.len_ts))
        for i in range(len(self.event_set) - 1):
            pos_loc = self.event_set[i + 1 : :] - self.event_set[i]
            dmu[:, pos_loc] += 1

        # Restrict Dmu to a predefined length
        dmu = dmu[:, : self.loss_length]

        # Estimate noise contribution
        z = np.median(dmu[:, 1 : self.noise_range])
        idx = np.arange(0, self.loss_length)
        zeta_mu = 0
        if self.len_es_ob != 0 and z != 0 and ~np.isnan(z):
            zeta_mu = z * (1 - (idx / self.len_es_ob))
            idx_ = np.where(zeta_mu < 0)[0]
            zeta_mu[idx_] = 0

        # Remove noise
        dmu_init = dmu.copy()
        dmu[0, :] -= zeta_mu

        # Contributions from the noise range should be neglected
        dmu[0, 0 : self.noise_range] = 0

        # With no noise in data, there are cases of dmu[:, idx] -= zeta_mu with negative results, set to 0
        idx = np.where(dmu[0, :] < 0)[0]
        dmu[0, idx] = 0

        return dmu, dmu_init

    def integral_convolution(self, dmu):
        """
        Smooth Dmu to account for randomness in the process.

        :param dmu: numpy ndarray, frequency table
        :return: tau: numpy ndarray, smoothed/rolled frequency table
        """
        len_dmu = dmu.shape[1]
        tau_mu = np.zeros((1, len_dmu))

        if self.sigma >= 1:
            sigma = self.sigma  # keep just one sigma, in order not to over-smooth
            weight_ = 1 / np.arange(2, sigma + 2)

            for k in np.arange(1, len_dmu, 1):
                total_weight = np.concatenate([weight_[::-1], np.array([1]), weight_])
                a = int(k - sigma)
                if a < 1:
                    a = 1
                if a == 1:
                    total_weight = total_weight[sigma - k + 1 :]

                b = int(k + sigma + 1)
                if b > len_dmu:
                    total_weight = total_weight[: len_dmu - b]
                    b = len_dmu

                r = np.zeros((len_dmu, 1))
                r[np.arange(a, b, 1)] = total_weight.reshape(-1, 1)
                tau_mu[:, int(k)] = np.dot(dmu, r)
        else:
            tau_mu = dmu

        return tau_mu

    def explain_data(self, tau_mu, mu):
        """
        Count the number of events, defined in tam_mu that are covered by the period and sigma.

        :param tau_mu: numpy ndarray (1,loss_length), smoothed frequencies of intervals
        :param mu: int, current periodicity
        :return: float, score of how much of the process is explained by the periodicity
        """
        len_tau = tau_mu.shape[1]
        number_events = int(len_tau / mu)

        # This is the integral-sum over all possible events associated with periodicity mu and sigma
        conf_int = np.arange(-self.sigma, self.sigma + 1, 1)
        mu_multp = (np.arange(1, number_events + 1) * mu).reshape(-1, 1)  # +1 to have number of events, inclusive
        mu_ci = (mu_multp - conf_int).reshape(-1, 1)

        idx = np.where((mu_ci > 0) & (mu_ci < len_tau))[0]
        domain = np.unique(mu_ci[idx].astype("int"))  # Overlapping regions are counted ones.

        return np.sum(tau_mu[:, domain])

    def remove_explained(self, dmu, mu):
        """
        Set dmu at mu +- sigma to zero.

        :param dmu: np ndarray (1,loss_length), frequencies of intervals
        :param mu: int, periodicity
        :return: dmu: np ndarray (1,loss_length), frequencies of intervals remained after dropping frequency mu
        """
        len_dmu = dmu.shape[1]
        number_events = int(len_dmu / mu)

        conf_int = np.arange(-self.sigma, self.sigma + 1, 1)
        mu_multp = (np.arange(1, number_events + 1) * mu).reshape(-1, 1)
        mu_ci = (mu_multp - conf_int).reshape(-1, 1)

        idx = np.where((mu_ci > 0) & (mu_ci < len_dmu))[0]
        domain = np.unique(mu_ci[idx].astype("int"))
        dmu[:, domain] = 0

        return dmu

    def gauss_pdf_explain(self, mu, sigma):
        """
        Evaluate the generative  function.

        :param mu: int
        :param sigma: int
        :return: gmu: np ndarray (len(mu),loss_length)  - Implied Gaussians wrt mu/sigma
        """
        gmu = np.zeros((1, self.loss_length))

        # The number of events during the observed period, needs to be adjusted with respect to long intervals with no
        # events (longer than  mu + 1.96 * sigma). These intervals are disregarded when computing the maximum possible
        # number of intervals.
        len_es_mu = sum(np.diff(self.event_set)[np.diff(self.event_set) <= mu + 1.96 * sigma])  # here 1.96* correct
        number_events_sn = int(len_es_mu / mu)
        if number_events_sn < 2:
            return np.nan * gmu

        x_range = np.arange(1, number_events_sn, 1)
        mu_x = mu * x_range
        b_x = number_events_sn - (x_range - 1)

        # Estimate gmu
        sigma_x = np.sqrt(self.random_walk * (x_range - 1) + 1) * sigma
        sigma_x2 = sigma_x ** 2
        a_x = 1 / np.sqrt(2 * np.pi * sigma_x2)
        for i in range(len(mu_x)):
            conf_ = np.ceil(3 * sigma_x[i])  # multiplication with 3 ensures covering of 99% of the gauss pdf.
            x = np.arange(int(max(1, mu_x[i] - conf_)), int(mu_x[i] + conf_ + 1), 1, dtype=np.int)
            x = x[np.where(x < self.loss_length)]
            gmu[:, x] += (a_x[i] * np.exp((-0.5 * (x - mu_x[i]) ** 2) / sigma_x2[i])) * b_x[i]

        return gmu

    def get_loss(self, dmu, gmu=[], mu_list=[], sigma_list=[]):
        """
        Estimate loss for a a list of mu/sigma.

        :param mu_list: list of int
        :param sigma_list: list of int
        :param gmu: np array of dim=(len(mu), dmu.shape[1])
        :return: float
        """
        if len(gmu) == 0:
            gmu = self.get_gmu(mu_list, sigma_list)
        diff_dmu_pmu = dmu - gmu.sum(axis=0)
        loss = np.sum(np.abs(diff_dmu_pmu), axis=1) / np.sum(np.abs(dmu), axis=1)

        return loss

    def get_gmu(self, mu_list, sigma_list):
        """
        Estimate gmu for a a list of mu/sigma.

        :param mu_list: list of int
        :param sigma_list: list of int
        :return: tuple:
             np array of dim=(len(mu), dmu.shape[1]),
             float
        """
        gmu = np.zeros((len(mu_list), self.loss_length))
        for i in range(len(mu_list)):
            gmu[i, :] = self.gauss_pdf_explain(mu_list[i], sigma_list[i])

        return gmu

    def get_sigma_init(self, dmu, mu_list, sigma_list):
        """
        Optimize sigma for every single mu.

        :param dmu:
        :param mu_list:
        :return: list
        """
        sigma_new = []
        for mu, sigma in zip(mu_list, sigma_list):
            sigma = self.get_sigma(dmu, [mu], [sigma])
            sigma_new.extend(sigma)

        return sigma_new

    def get_sigma(self, dmu, mu_list, sigma_list):
        """
        Use curvefit to improve the guess of sigma.

        :param dmu: np array of dim (1, self.loss_length)
        :param mu_list: sorted list
        :param sigma_list: list
        :return: list
        """
        mu_max = mu_list[-1]
        mu_min = mu_list[0]

        # 3 * self.sigma to cover 99 of pdf, +1 to have loss_length_ inclusive
        loss_length_ = int(min(mu_max + np.ceil(3 * sigma_list[-1] + 1), dmu.shape[1]))

        st = int(max((mu_min - 0.75 * mu_min), 0))
        end = int(min((mu_max + 0.75 * mu_max + 1), dmu.shape[1]))

        y = dmu[:, 0:loss_length_][0]
        y = dmu[:, st:end][0]
        x = np.arange(st, end)

        init_params = []
        mu_fc = []
        b_up = []
        for mu, sigma in zip(mu_list, sigma_list):
            off_set = np.floor(mu_max / mu).astype("int")
            for i in range(off_set):
                mu_fc.append(float(mu * (i + 1)))
                init_params.append(max(min(float(sigma), np.ceil(np.log(mu_max)) - 1), 1))
                init_params.append(1.0)
                b_up.append(0.25 * mu * (i + 1))
                b_up.append(np.inf)

        # Lower bounds
        b_low = np.ones((1, len(init_params)))
        b_low[0, 1::2] = 0.0  # for a, set lower bound to zero, no upper bound as dmu is not normalized.

        # Fit dmu curve
        fc = FitClass()
        fc.mu = mu_fc
        params, _ = curve_fit(fc.multi_modal, x, y, init_params, bounds=([b_low.tolist()[0], b_up]))
        results = np.zeros((int(len(params) / 2), 2))
        for i in range(int(len(params) / 2)):
            row = i * 2
            results[i, :] = [params[row], params[row + 1]]

        off_set = 0
        sigma_new = []
        for mu in mu_list:
            sigma_new.append(np.round(results[off_set, 0]))
            off_set += np.floor(mu_max / mu).astype("int")

        return sigma_new

    @staticmethod
    def get_mod_mu(mu_list, pmu):
        """
        Remove all multiples of a periodicity mu, except the multiples with a higher pmu.

        :param mu_list, list of mus
        :param pmu
        :return: mu_list: sorted periodicities without their multiples
                 multiples: list of multiples
        """
        mu_list, pmu = (list(t) for t in zip(*sorted(zip(mu_list, pmu), reverse=False)))
        mu_pmu = {k: l for k, l in zip(mu_list, pmu)}

        mu_pmu_red = mu_pmu.copy()
        mu_list = []

        for k in mu_pmu.keys():
            mu_mod = {i: v for i, v in mu_pmu_red.items() if not i % k}
            if mu_mod:
                for key in mu_mod:
                    mu_pmu_red.pop(key)

                # keep mus that have pmu >= pmu of the min mu_list
                values = list(mu_mod.values())
                mu_keep = {k: v for k, v in mu_mod.items() if v >= values[0]}
                mu_list.extend(list(mu_keep.keys()))

                if not mu_pmu_red:
                    break

        return mu_list

    def initialize_periods(self, dmu):
        """
        Use step function support instead of Gaussian to pre-compute the set of periodicities.

        :param dmu:
        :return: sorted tuple of numpy ndarray: mu_list - contains all periodicities, sorted
        """

        mu_list = []
        pmu_set = []
        for iteration in range(self.max_depth):

            # Integrate across dmu
            tau_mu = self.integral_convolution(dmu.copy())
            if iteration == 0:
                tau_mu_init = tau_mu

            # Sort and get the indices of max_candidates values from tau_mu
            tau_mu = self.round_down(tau_mu, 6)
            top_mu = np.argsort(-1 * tau_mu[0, : self.mu_range_max + 1], kind="mergesort")  # sort
            top_mu = top_mu[np.array(top_mu) >= self.mu_range_min][: self.max_candidates]  # truncate

            top_mu = np.intersect1d(top_mu, np.where(tau_mu[0, :] > 0)[0])  # exclude mu with tau zero

            mu_list.extend(top_mu)

            # Estimate how much top_mu can explain, top_mu[p]/loss_length is used to get absolute Pmu for all mu
            len_top_mu = len(top_mu)
            pmu = np.zeros((1, len_top_mu))
            if len_top_mu > 0:
                for p in range(len_top_mu):
                    pmu[:, p] = self.explain_data(tau_mu.copy(), top_mu[p]) * (
                        top_mu[p] / self.loss_length
                    )  # ATTENTION, keep () otherwise rounding issues.
                    pmu_set.extend(pmu[:, p])

                # Remove which can be explained by the best
                idx_max = np.argmax(pmu[0, :])
                dmu = self.remove_explained(dmu, top_mu[idx_max])
            else:
                print("Reduce max_depth or sigma")

        mu_list = self.get_mod_mu(mu_list, pmu_set)
        mu_list.sort()

        return mu_list, tau_mu_init

    def get_best_combination(self, dmu, gmu, mu_list, sigma_list, loss_best):
        """
        Get the  optimal set of combinations of all periodicities.

        :param dmu: numpy ndarray, (1,loss_length), frequencies of intervals
        :param gmu: numpy ndarray, (1,loss_length), frequencies of intervals wrt generative model
        :param mu_list: list of top mu
        :param loss_best: float
        :return: list, float
        """
        if self.max_periods > len(mu_list):
            self.max_periods = len(mu_list)
        mu_tmp = si_tmp = gmu_tmp = []
        mu_best = si_best = gmu_best = []

        for num_of_periods in range(1, self.max_periods + 1):
            loss_origin = loss_best

            for idx in itertools.combinations(range(len(mu_list)), num_of_periods):
                # test for overlapping mu/sigmas
                mu_idx = [mu_list[i] for i in idx]
                si_idx = [sigma_list[i] for i in idx]
                sigma_plus = [s1 + s2 for s1, s2 in zip(si_idx[:-1], si_idx[1:])]

                if (np.diff(mu_idx) > sigma_plus).all():
                    loss = self.get_loss(dmu, gmu=gmu[idx, :])

                    if loss < loss_best:
                        loss_best = loss
                        mu_tmp, si_tmp, gmu_tmp = mu_idx, si_idx, gmu[idx, :]

            if loss_best < loss_origin - self.loss_change_tol:
                loss_origin = loss_best
                mu_best, si_best, gmu_best = mu_tmp, si_tmp, gmu_tmp

        return gmu_best, mu_best, loss_origin, si_best

    def extract_periods(self, ts=None, verbose=True):
        """
        GMPDA Algo for periodicity extraction.

        :param self:
        :param ts, numpy ndarray, series of events
        :param verbose, boolean
        :return:
            mu_list: list of top extracted periodicities
            dmu: numpy ndarray, (1,loss_length), frequencies of intervals
            tau_mu: numpy ndarray, (1,loss_length), smoothed frequencies of intervals
            loss_best: float, loss obtained wrt mu_list
            self.sigma: list, sigmas for mu_list
        """

        def printv(*txt):
            if verbose:
                print(txt[0])

        printv("==" * 25)
        printv("GMPDA STARTS")
        ###################################################################
        # 0. Calculate reference loss if requested.
        ###################################################################
        ref_loss = np.NaN
        if self.ref_loss_n > 0:
            ref_loss = self.get_ref_loss(self.ref_loss_n)
            printv(
                f"Reference Loss: min={min(ref_loss)}, 0.01 quantile={np.quantile(ref_loss,0.01)}, "
                f"0.05 quantile={np.quantile(ref_loss,0.05)}"
            )

        ###################################################################
        # 1. If ts is not none update initialization
        ###################################################################
        if ts is not None:
            self.reinit_ts(ts_=ts)

        ###################################################################
        # 2. Compute Intervals
        ###################################################################
        dmu, dmu_init = self.get_intervals()
        loss_best = np.array(np.finfo("f").max)
        gmu_best = []

        ###################################################################
        # 3. Initialize periods
        ###################################################################
        mu_list, tau_mu = self.initialize_periods(dmu.copy())

        ###################################################################
        # 4. Initialize sigma
        ###################################################################
        if self.sigma_log_init:
            sigma_list = np.ceil(np.log(mu_list))
        else:
            sigma_list = [self.sigma] * len(mu_list)
            idx = np.where(sigma_list > mu_list)[0]
            for i in idx:
                sigma_list[i] = int(np.ceil(np.log(mu_list[i])))

        # Check if sigma==0, replace by 1
        sigma_list = [1 if x == 0 else x for x in sigma_list]

        ###################################################################
        # 5. Check if the data has variation, else GMPDA is not appropriate
        ###################################################################
        if len(mu_list) == 0:
            printv("No periods could be found")
            return mu_list, dmu, tau_mu, gmu_best, loss_best, [sigma_list]
        elif len(mu_list) == 1 and (sum(dmu - tau_mu) == 0).all():
            printv("There is one period, with sigma=0")
            printv("GMPDA FINISHED")
            printv("Best obtained mu {}, loss {}, sigma {}".format(mu_list, 0, 0))
            printv("==" * 25)
            return mu_list, dmu, tau_mu, gmu_best, loss_best, [sigma_list]
        elif (sum(dmu - tau_mu) == 0).all():
            printv("Warning: It seems there is no randomness in the process, i.e, sigma=0")
            printv("Top selected mu {}".format(mu_list[0 : self.max_candidates]))

        ###################################################################
        # 6. Optimize sigma for all candidate mu
        ###################################################################
        if self.sigma_curvefit:
            try:
                sigma_list = self.get_sigma_init(dmu, mu_list, sigma_list)
                printv("Sigma optimized via Trust Region Reflective Algo.")
            except Exception as e:
                printv(f"Could not find optimal sigma using TRF, Error message: {str(e)}")

        ###################################################################
        # 7. Compute gaussian mixture gmu for each candidate periodicity
        ###################################################################
        gmu = self.get_gmu(mu_list, sigma_list)

        ###################################################################
        # 8. Find combination of periodicities which minimize loss
        ###################################################################
        gmu_best, mu_list, loss_best, sigma_list = self.get_best_combination(dmu, gmu, mu_list, sigma_list, loss_best)

        ###################################################################
        # 9. Update loss and sigma wrt optimal periodicities
        ###################################################################
        if self.sigma_curvefit and len(mu_list) > 1:
            try:
                sigma_new = self.get_sigma(dmu.copy(), mu_list, sigma_list)
                gmu = self.get_gmu(mu_list, sigma_new)
                loss_new = self.get_loss(dmu, gmu, mu_list, sigma_new)

                if loss_best > loss_new:
                    loss_best, gmu_best = loss_new, gmu
                    sigma_list = sigma_new
                    printv(f"Updated sigma via Trust Region Reflective Algo., new sigma={sigma_list}")
                else:
                    printv("Sigma update does not improve loss.")
            except Exception as e:
                printv("Could not find optimal sigma using TRF, Error message:" + str(e))

        printv("GMPDA FINISHED")
        printv("Best obtained mu {}, loss {}, sigma {}".format(mu_list, loss_best, sigma_list))
        printv("==" * 25)

        # return mu_list, sigma_list, loss_best, ('None') if self.report_short \
        #     else mu_list, sigma_list, loss_best, (dmu, dmu_init, tau_mu, gmu_best, ref_loss)

        return mu_list, sigma_list, loss_best, dmu, dmu_init, tau_mu, gmu_best, ref_loss

    ###################################################################
    # Plot function
    ###################################################################
    def plot_results(self, gmu=[], tau_mu=[], mu_best=[], sigma_best=[], plot_dmu=True, figsize=(10, 5)):
        """
        Plotting the final results. Default also dmu is plotted in an extra plot.

        :param dmu: numpy ndarray, (1,loss_length)
        :param gmu: numpy ndarray, (len(mu_best),loss_length)
        :param tau_mu: numpy ndarray, (1,loss_length)
        :param mu_best: list
        :param sigma_best: list
        :param plot_dmu: boolean
        :return:
        """
        dmu, dmu_in = self.get_intervals()
        z = np.median(dmu_in[0, 1 : self.noise_range])

        fig = plt.figure(figsize=figsize)
        if plot_dmu:
            plt.plot(np.arange(0, dmu_in.shape[1]), dmu_in[0, :], linewidth=2, label="dmu", color="grey", alpha=0.5)
            plt.plot(
                np.arange(0, dmu.shape[1]),
                dmu[0, :],
                "--",
                linewidth=2,
                label=f"dmu without noise (={z})",
                color="grey",
            )
            plt.grid()
            plt.legend()
            plt.title("Plot of all order intervals between events (with and without noise).")
            plt.show()

        if len(gmu) > 0 and len(tau_mu) > 0 and len(mu_best) > 0 and len(sigma_best) > 0:
            plt.plot(np.arange(0, dmu.shape[1]), dmu[0, :], linewidth=2, label="dmu without noise", color="grey")
            plt.plot(tau_mu[0, :], "--", label="tau_mu", alpha=0.7)
            for i, (mu, si) in enumerate(zip(mu_best, sigma_best)):
                plt.plot(gmu[i, :].T, label=f"gmu for mu={mu}, sigma={si}")

            plt.grid()
            plt.legend()
            plt.title("GMPDA results: All order intervals (dmu) without noise,\n smoothed dmu, and, generative models.")
            plt.show()

        return dmu, dmu_in, z, fig
