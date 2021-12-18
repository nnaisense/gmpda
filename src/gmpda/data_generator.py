import numpy as np


class DataGenerator:
    def __init__(self, nrevents, periods, sigmas, fp_beta):
        """
        Generates test cases for the clock and the random walk model.

        :param nrevents: int, Number of events to be generated.
        :param periods: list of integers
        :param sigmas: list of integers
        :param fp_beta: float in [0,1]:  percentage of additional white noise
        """

        assert len(periods) == len(sigmas), "length of periods and sigmas should be the same"

        self.periods = periods
        self.N = np.max(self.periods) * nrevents
        self.sigmas = sigmas
        self.fp_beta = fp_beta

        # fixme
        self.nrevents = nrevents
        idx_max = np.argmax(self.periods)
        self.Nsigma = self.periods[idx_max] * nrevents + self.sigmas[idx_max] * nrevents

    def gen_clock_model(self):
        """
        Generates events wrt clock model.

        :return: (np array, dict): time series of length N, dictionary of pairs: (event: period)
        """
        S = dict()
        ts = np.zeros((1, self.N))
        for period, sigma in zip(self.periods, self.sigmas):
            for i in np.arange(1, self.N, 1):
                if i % period == 0:
                    loc = i + int(np.round(np.random.normal(0, sigma, 1)))
                    if 0 < loc < self.N:
                        ts[:, loc] = 1
                        if loc in S.keys():
                            S[loc].append(period)
                        else:
                            S[loc] = [period]

        if self.fp_beta > 0:
            noise_count = int(np.round(np.sum(ts) * self.fp_beta))
            noise = np.random.randint(self.N, size=noise_count)
            ts[:, noise] = 1
        return ts, S

    def gen_rw_model(self):
        """
        Generates events wrt random walk model.

        :return: (np array, dict): time series of length N, dictionary containing events: period
        """
        S = dict()
        ts = np.zeros((1, self.N))
        for period, sigma in zip(self.periods, self.sigmas):
            loc = 0
            for i in range(np.floor(self.N / period).astype("int")):
                loc += int(np.round(np.random.normal(period, sigma, 1)))
                if 0 < loc < self.N:
                    ts[:, loc] = 1
                    if loc in S.keys():
                        S[loc].append(period)
                    else:
                        S[loc] = [period]

        if self.fp_beta > 0:
            noise_count = int(np.round(np.sum(ts) * self.fp_beta))
            noise = np.random.randint(self.N, size=noise_count)
            ts[:, noise] = 1

        return ts, S

    def gen_rw_model_different(self):
        """
        Generates events wrt random walk model.

        :return: (np array, dict): time series of length N, dictionary containing events: period
        """
        S = dict()
        ts = np.zeros((1, self.Nsigma))
        for period, sigma in zip(self.periods, self.sigmas):
            loc = 0
            for i in range(np.floor(self.nrevents * period).astype("int")):
                loc += int(np.round(np.random.normal(period, sigma, 1)))
                if 0 < loc < self.N + sigma:
                    ts[:, loc] = 1
                    if loc in S.keys():
                        S[loc].append(period)
                    else:
                        S[loc] = [period]

        if self.fp_beta > 0:
            noise_count = int(np.round(np.sum(ts) * self.fp_beta))
            noise = np.random.randint(self.N, size=noise_count)
            ts[:, noise] = 1

        return ts, S
