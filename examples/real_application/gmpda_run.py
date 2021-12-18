import argparse
import logging
import os
import pickle
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import yaml

from gmpda import GMPDA

if __name__ == "__main__":

    # parse for running configurations
    parser = argparse.ArgumentParser(description="GMPDA Experiments")
    parser.add_argument("--exp-config", type=str, metavar="CFG", help="experiment configuration file")
    parser.add_argument("--logdir", type=str, default="logs", help="Relative folder for save/log results of this exp")
    args = parser.parse_args()

    ##################################################
    # Load the model configuration yml file for the experiment
    ##################################################
    with open(args.exp_config) as f:
        cfg = yaml.safe_load(f)

    ##################################################
    # Set random seed
    ##################################################
    np.random.seed(cfg["gmpda"]["seed"])
    random.seed(cfg["gmpda"]["seed"])

    ##################################################
    # Logger and output folder for experiment
    ##################################################
    time_now_ = time.strftime("%Y%m%d_%H%M%S")
    exp_name = cfg["experiment"]["name"] + "_{}".format(time_now_)
    res_name = cfg["experiment"]["name"] + "_result_{}".format(time_now_)
    exp_root = "{}/{}".format(cfg["experiment"]["results_path"], exp_name)

    if not os.path.exists(exp_root):
        os.makedirs(exp_root)
    else:
        print("Folder {} for exp already exists!".format(exp_root))
        sys.exit(-1)

    # fix that
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # if y want to have print in the log file use log.info instead
    logging.basicConfig(filename="{}/INFO.log".format(exp_root), filemode="w", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.info("Created folder {} for experiment.".format(exp_name))
    logger.info("Command line: {}".format(" ".join(sys.argv)))
    logger.info("Copy the model configuration file to experiment folder.")
    shutil.copy(src="{}".format(args.exp_config), dst="{}".format(exp_root))

    #########################
    # load csv file in the preprocess folder
    #########################
    ts = pd.read_csv(cfg["experiment"]["data_path"]).values.reshape(-1, 1).T
    logger.info("Event time series loaded from {}".format(cfg["experiment"]["data_path"]))
    logger.info("The loaded event time series has shape {}".format(ts.shape))

    #########################
    # check if there are events, and number of events > 1
    #########################
    logger.info("Event time series has in total {} events".format(ts.sum()))
    assert ts.sum() > 1, logger.error("Only 1 event in ts")
    assert ts.sum() > 30, logger.warning("Number of events in ts is less than 30. Results might be not reliable.")

    #########################
    # init  gmpda
    #########################
    gmpda = GMPDA(
        ts=ts,
        # model
        random_walk=cfg["gmpda"]["random_walk"],
        # gaussian pdf settings
        sigma=cfg["gmpda"]["sigma"][0],
        sigma_curvefit=cfg["gmpda"]["sigma_curvefit"],
        sigma_log_init=cfg["gmpda"]["sigma_log_init"],
        # hierarchical algo settings
        max_depth=cfg["gmpda"]["max_depth"],
        max_periods=cfg["gmpda"]["max_periods"],
        mu_range_min=cfg["gmpda"]["mu_range_min"],
        mu_range_max=cfg["gmpda"]["mu_range_max"],
        max_candidates=cfg["gmpda"]["max_candidates"],
        # dealing with noise
        noise_range=cfg["gmpda"]["noise_range"],
        loss_length=cfg["gmpda"]["loss_length"],
        loss_change_tol=cfg["gmpda"]["loss_change_tol"],
    )

    #########################
    # estimate reference loss
    #########################
    if cfg["experiment"]["ref_loss"]:
        print("==" * 25)
        print("Estimating reference loss")
        ref_loss = gmpda.get_ref_loss(cfg["experiment"]["ref_loss_samples"])
        print("Reference loss is {} ".format(np.asarray(ref_loss).mean(axis=0)))

    #########################
    # Extract periodicities
    #########################
    mu_best, sigma_best, loss, dmu, dmu_init, tau_mu, gmu_best, ref_loss = gmpda.extract_periods()

    #########################
    # Save results
    #########################
    results_dict = {}
    results_dict = {
        "loss": loss,
        "mu_best": mu_best,
        "sigma_best": sigma_best,
        "ref_loss": ref_loss,
        "Dmu": dmu,
        "tau_mu": tau_mu,
    }

    result_name_full = "{}/results_dic.{}".format(exp_root, cfg["experiment"]["results_format"])

    with open(result_name_full, "wb") as fp:
        pickle.dump(results_dict, fp, pickle.HIGHEST_PROTOCOL)

    #########################
    # Experiment finished
    #########################
    logger.info("Final results are stored in {}".format(result_name_full))
