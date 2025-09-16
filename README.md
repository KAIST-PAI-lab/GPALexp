# GPALexp
GPALexp is a Python package implementation of Gaussian Process Active Learning (GPAL, Chang et al., 2021).  

## What is GPAL?
GPAL is a nonparametric Bayesian optimization technique that can approximate a wide range of underlying continuous functions.  

GPAL enables us to optimize experimental stimuli and obtain maximal information regarding each participant, in the most efficient way.  

## What are the benefits of using GPALexp?
GPALexp built-in functions can readily be incorporated in existing Python experiment codes, thereby efficiently capture varying patterns of individual data.

Since GPALexp has integrated a long sequence of executions required to conduct GPAL into 3 functions, we can easily apply GPAL optimization in the existing experiment codes.

This will help us effectively discover underlying functions of individual data in a concise manner.


## Features of GPALexp
- **Adaptive Design Selection with internal functions:** `GPRInstance()`, `argsConstructor()`, and `gpal_optimize()`
- **Various built-in plotting functions for visualization**
- **Supports GPAL optimization for arbitrary number of feature stimuli**
- **Example code for 1D GPAL optimization with 1D Number-Line Task** (Lee et al., 2022)

## Installation
GPALexp is built upon Python 3.10.18 and other libraries including numpy, pandas, scipy, and scikit-learn.  
Note that the only thing requried in advance is Python 3.10, since other libraries will automatically be installed during the installation process.
```
# Installing from PyPI
pip install gpalexp

# Installing directly from github (developmental version)
TBD
```

## GPALexp Wiki
We've provided explanatory materials in the github Wiki of this repository.  
Please refer to this [Wiki page](https://github.com/KAIST-PAI-lab/GPALexp/wiki) for further details.  

## Contacts
If there are any things that the maintainer should be noticed (bug reports, update requests, questions, future suggestions, etc),  
please contact Junyup Kim (ytrewq271828@alumni.kaist.ac.kr).
