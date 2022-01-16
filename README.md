# Table of Contents

1.  [histopathTDA](#org8c6b7ff)
    1.  [Installation](#orgc6e3e06)
    2.  [Anaconda](#orgf97c9df)
        1.  [Linux](#org0b95b61)
    3.  [Virtual Environment](#org2edd6b8)
    4.  [Data Access](#orgf53e583)
        1.  [Configure a persistent network drive mount](#org7967538)
    5.  [Data Description](#org7dd42b5)


<a id="org8c6b7ff"></a>

# histopathTDA

This is a python library for conducting histopathology using TDA methods.

<a id="orgc6e3e06"></a>

## Installation

### Anaconda

Currently the library is setup to build within its provided Anaconda environment. 
Setup to install via pip is currently under construction.  To install
Anaconda3 on MacOS or Windows, visit the install page
[here](https://docs.anaconda.com/anaconda/install/mac-os/) for MacOS or
[here](https://docs.anaconda.com/anaconda/install/windows/) for Windows, and
follow the directions. On Linux installation instructions are provided
[here](https://docs.anaconda.com/anaconda/install/linux/).

Alternatively, if you use [homebrew](https://brew.sh/) on MacOS, you can install with

```bash
brew install --cask anaconda
```

or if you use [chocolatey](https://chocolatey.org) on Windows, you can install with

```bash
choco install anaconda3
```

## Virtual Environment

Once you have Anaconda installed, use it to create a virtual environment
that includes python dependencies by running the following in your shell

```bash
conda env create -f environment.yml
```

Alternatively, if you have make installed on your system, you can run
```bash
make create_environment
```

Once the environment is created, we can activate it with

```bash
conda activate histopathTDA
```

You may need to run `conda init` if this is the first time activating an Anaconda
virtual environment. Now, once you have the anaconda environment activated, we
use pip to install the package to our environment.

```bash
pip install .
```

