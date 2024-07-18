import random

import numpy as np

from gmpda import GMPDA, DataGenerator

np.random.seed(3)
random.seed(3)


true_periodicity = [167, 101]
sigma_true = [2, 3]
nr_event = 500
dg = DataGenerator(nrevents=nr_event, periods=true_periodicity, sigmas=sigma_true, fp_beta=0)

ts_rw, _ = dg.gen_rw_model()

mu_range_max = 350
mu_range_min = 10
loss_length = 2 * 350

gmpda = GMPDA(
    ts_rw,
    gmpda_seed=3,
    random_walk=1,
    sigma=2,
    ref_loss_n=10,
    sigma_curvefit=True,
    sigma_log_init=True,
    mu_range_min=mu_range_min,
    mu_range_max=mu_range_max,
    noise_range=mu_range_min,
    max_depth=5,
    max_candidates=15,
    max_periods=3,
    loss_length=loss_length,
    loss_change_tol=0.01,
)


print("True parameters: mu {}, sigma {}".format(true_periodicity, sigma_true))
mu_best, sigma, loss, D_tau, dmu_init, tau_mu, gmu_best, ref_loss = gmpda.extract_periods()


gmpda.plot_results(gmu_best, tau_mu, mu_best, sigma, plot_dmu=False)
