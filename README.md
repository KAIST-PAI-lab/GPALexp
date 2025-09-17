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
<br>

# Tutorial  
This tutorial is written for the researcher who wishes to use GPAL in experiments, providing practical guidance on implementation and data analysis and evaluation. It assumes a working knowledge of Python programming and provides a step-by-step guide for embedding GPAL in code using GPALexp to reduce the programming required to a few function calls. Technical details on the implementation are provided in [GPALexp Wiki](https://github.com/KAIST-PAI-lab/GPALexp/wiki)

## Installation  

GPALexp is built upon Python 3.10.18, so Python 3.10 is recommended. The Anaconda distribution of Python [[link]](https://www.anaconda.com) can help us create a virtual environment for GPALexp, avoiding potential conflicts and providing tailored dependencies.  

Install GPALexp from the Python Package Index (PyPI) via the following pip command in a terminal (or Command Prompt on Windows).
```
# Installing from PyPI
pip install gpalexp
```  
Or we can install the developmental version directly from this github repository.  
```
# Installing directly from github
TBD
```  
To confirm successful installation of GPALexp, run the following command in a terminal.
```
python -m pip show gpalexp
```  
The above command will display the current version of GPALexp if it is installed correctly.  

The tutorial experiment code was implemented using PsychoPy [[link]](https://www.psychopy.org/). To reproduce and execute the code, users must install PsychoPy version 2025.1.1, which can be installed via the following command:  
```
pip install psychopy==2025.1.1
```  
Note that additional dependencies, such as a compatible C++ compiler (e.g., Microsoft C++ Build Tools), may be required depending on the system configuration.

## GPALexp Wiki
We've provided explanatory materials in the github Wiki of this repository.  
Please refer to this [Wiki page](https://github.com/KAIST-PAI-lab/GPALexp/wiki) for further details.  

## Contacts
If there are any things that the maintainer should be noticed (bug reports, update requests, questions, future suggestions, etc), please feel free to contact Junyup Kim (ytrewq271828@alumni.kaist.ac.kr).
