# GPALexp
GPALexp is a Python package implementation of Gaussian Process Active Learning (GPAL, Chang et al., 2021).  
GPAL is a nonparametric Bayesian optimization technique that can approximate a wide range of underlying continuous functions.  
GPALexp can be readily incorporated in existing Python experiment codes, thereby efficiently gather varying patterns of individual data.

GPALexp is built upon Python 3.10.18 and other libraries including numpy, pandas, scipy, and scikit-learn.

## Features
- **Adaptive Design Selection with internal functions:** `GPRInstance()`, `argsConstructor()`, and `gpal_optimize()`
- **Various built-in plotting functions for visualization**
- **Supports GPAL optimization for arbitrary number of feature stimuli**
- **Example code for 1D GPAL optimization with 1D Number-Line Task** (Lee et al., 2022)

## Installation
```
# Installing from PyPI
pip install gpalexp

# Installing directly from github (developmental version)
TBD
```

## GPALexp Wiki
We've provided explanatory materials in the github Wiki of this repository.  
Please refer to this [Wiki page](https://github.com/KAIST-PAI-lab/GPALexp/wiki) for further details.
