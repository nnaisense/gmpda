# GMPDA

[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Table of Contents

<!-- vim-markdown-toc GFM -->

* [Gaussian Mixture Periodicity Detection Algorithm (GMPDA)](#gaussian-mixture-periodicity-detection-algorithm-gmpda)
* [Installation](#installation)
    * [Install directly from GitHub](#install-directly-from-github)
    * [Or install from source in editable mode (to run examples or to modify code)](#or-install-from-source-in-editable-mode-to-run-examples-or-to-modify-code)
    * [Usage](#usage)
* [GMPDA parameters](#gmpda-parameters)
    * [Practical Hints on Setting the GMPDA Parameters](#practical-hints-on-setting-the-gmpda-parameters)
* [Examples](#examples)
    * [Test-case](#test-case)
    * [Real-Application](#real-application)
    * [GMPDA with R](#gmpda-with-r)

<!-- vim-markdown-toc -->

## Gaussian Mixture Periodicity Detection Algorithm (GMPDA)
This repository contains the code for the GMPDA Algorithm presented in the "Generative Models for Periodicity Detection in Noisy Signals" [paper](https://arxiv.org/abs/2201.07896).


GMPDA is a Gaussian-Mixture based **periodicity detection** algorithm for **event time series**.
The algorithm learns the parameters of a **generative function** of the time series, which implicates
**simultaneous periodicities**, **interval variance**, as well as false-positive and negative **noise**.

![Clockm Model](/figures/demo_intervals4_small.png)



The algorithm addresses two possible generative models: the Clock and Random Walk models.

The **Clock Model** describes periodic processes which are governed by an external pacemaker.

![Clockm Model](/figures/model_clock_small.png)


The **Random Walk Model** describes processes like heartbeats, breaths,
and other time sequences when the next event depends only on the location
of the current event but not on any external pacemaker. Therefore the variance terms add.

![RW Model](/figures/model_rw_small.png)



## Installation
You can install gmpda using following commands:

### Install directly from GitHub
```bash
pip install git+https://github.com/nnaisense/gmpda
```

### Or install from source in editable mode (to run examples or to modify code)
```bash
git clone https://github.com/nnaisense/gmpda.git
pip install -e gmpda
```

### Usage
```python
from gmpda import GMPDA
```

See Section [Examples](./README.md#examples) for more details.

## GMPDA parameters
GMPDA uses a variety of hyper-parameters. These hyper-parameters are primarily designed to narrow the search space for periodicities.

Important GMPDA model parameters:

- **random_walk** *(int)* - 1 or 0 for random walk or clock model.
- **mu_range_min** *(int)* - Defines lower bound of identifiable periodicities. Acts as high pass filter on the frequency detection algorithm.
- **mu_range_max** *(int)* - Defines upper bound of identifiable periodicities. Acts as a low pass filter on the frequency detection algorithm.
- **sigma_curvefit** *(boolean)* - True for non-linear least squares curve fitting for variance, default False.
- **sigma** *(list of ints)* - Initial guess for variance of the underlying Gaussian model. If the sigma is fitted, i.e., *sigma_curvefit = True*, one should select a higher value of *sigma* for optimal results. To find the optimal *sigma*, run GMPDA for different *sigma*, optimal *sigma* is minimizing the loss.
- **sigma_init** *(boolean)* -  If this parameter is set to True, for every extracted periodicity, the variance is initialized as the logarithm of this periodicity.
- **max_candidates** *(int)* - Is used in the fast algorithm to determine how many candidate periodicities to consider in each hierarchical step.
- **max_periods** *(int)* - Maximal number of interwoven periodicities to search for of multiple/overlapping periodicities. Larger numbers will lead to higher computational cost, but if there are multiple periods in the data, it should be equivalently large.
- **max_depth** *(int)* - Determines the maximal number of hierarchical steps in the fast algorithm.
- **loss_length** *(int)* - Defines the range over which the loss function will be computed. The minimum is *mu_range_max*, the maximum is the length of the time series. Please note, for very noisy, or short time series *loss_length* should be set relatively low, e.g., *2mu_range_max*.
- **loss_change_tol** *(float)* - Minimal magnitude of the loss decrease a value in `[0,1)`.
- **ref_loss_n** *(int)* - If *ref_loss_n*>0 then the loss is computed for *n* random time series of the same length as the input and returned as the averaged loss, i.e., the reference loss.
- **noise_range** *(int)* - Is used to approximate the noise.


### Practical Hints on Setting the GMPDA Parameters

In general, if no prior information is available about the underlying process/periodicities, treat all the above GMPDA parameters as hyper-parameters and run GMPDA for different combinations configurations of the parameters. The optimal combination is selected for minimal loss.
However, we would like to share some practical insides on how to confine the parameter space. Let's denote in the following the considered time series of events by ts:

- Parameters *mu_range_min*, *mu_range_max* are problem dependent and should be set by the user. If no prior information is available, set *mu_range_min = 5* and *mu_range_max = round(len(ts)/2)*.
- Test-cases and some real application validated to initialize variance, i.e.,
    - *sigma_init = True*,
    - *sigma = []*,
- *max_candidates = 15*,
- *max_periods = 3*, Increase if you expect to have more overlapping periodicities, but please note, for *max_nr_period > 3*  GMPDA was not tested in very detail.
- *max_depth = 5*,
- *loss_length = 2mu_range_max*,
- *loss_change_tol = 0.001*,
- *noise_range = 5*,

## Examples

### Test-case

Open a terminal and run:
```bash
$ python examples/testcase.py
```

The output should be:
```
True parameters: mu [167, 101], sigma [2, 3]
==================================================
GMPDA STARTS
Reference Loss: min=1.021058252479727, 0.01 quantile=1.0301487506017142, 0.05 quantile=1.0665107430896632
Sigma optimized via Trust Region Reflective Algo.
Sigma update does not improve loss.
GMPDA FINISHED
Best obtained mu [np.int64(101), np.int64(167)], loss [0.22895101], sigma [np.float64(3.0), np.float64(2.0)]
==================================================
```

### Real-Application

Folder *examples/real_application* contains `gmpda_cfg.yml` file for setting the path to the time series to be analyzed, and the parameter configuration for GMPDA.
For real data application run `gmpda_run.py`, this call creates a folder, default *logs*, copies the configuration used for gmpda, and stores the results there.

Open a terminal, cd to `examples/real_application` and run:

```bash
$ python gmpda_run.py --exp-config gmpda_cfg.yml
```

The output should be:

```
Created folder test_real_ts_{current_data} for experiment.
Command line: run_gmpda.py --exp-config gmpda_cfg
Copy the model configuration file to experiment folder.
Event time series loaded from ../data/some_real_ts.csv
The loaded event time series has shape (1, 13199)
Event time series has in total 392 events
Require loss_length > mu_range_max. Set loss_length = mu_range_max + 3*max(self.sigma)
==================================================
Estimating reference loss
Reference loss is 1.0222583806514751 
==================================================
GMPDA STARTS
Sigma optimized via Trust Region Reflective Algo.
Sigma update does not improve loss.
GMPDA FINISHED
Best obtained mu [np.int64(19), np.int64(75)], loss [0.20541469], sigma [np.float64(5.0), np.float64(19.0)]
==================================================
Final results are stored in logs/test_real_ts_20220103_113055/results_dic.pkl
```

The file `results_dic.pkl` contains:
* loss - the final loss
* mu_best - a list of extracted periodicities
* sigma_best - a list of corresponding variances
* ref_loss - the reference loss for this time series
* d_mu - a np array, contains the all order forward differences, as defined in the paper
* tau_mu - convoluted d_mu


### GMPDA with R
You can use gmpda whithin R. The file `gmpda_with_r.Rmd` contains an example.
