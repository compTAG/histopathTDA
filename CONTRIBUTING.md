# Contributing

## Package tooling

Installed in the conda environment are several python packages providing tools
for package development. In particular, we are using
[tox](https://github.com/tox-dev/tox) for automating running tests (using
[pytest](https://github.com/pytest-dev/pytest)), benchmarks (using
[pytest-benchmark](https://github.com/ionelmc/pytest-benchmark/)), coverage
(using and [Coverage.py](https://github.com/nedbat/coveragepy) and
[pytest-cov](https://github.com/pytest-dev/pytest-cov),
[flake8](https://github.com/pycqa/flake8) for linting and documentation (using
[Sphinx](https://github.com/sphinx-doc/sphinx/)).

Everything is automated so that you can run
```bash
tox
```

from within the histopathTDA conda environment.

Alternatively, you can run pytest, or sphinx individually within the
console/command prompt. There is a Makefile that gives commands for this
workflow. If you do not already have Make installed, you can install by
following the instructions
[here](https://www.gnu.org/software/make/). Alternatively, on MacOS, you can
follow the instructions
[here](https://stackoverflow.com/questions/10265742/how-to-install-make-and-gcc-on-a-mac)
or install via homebrew:

```bash
brew install make
```

On windows, you can also install via chocolatey:

```bash
choco install make
```

See the CONTRIBUTING.md document for more instructions on this workflow.

Please, in order of priority:
 - Write code
 - Write docstrings
 - Write unittests

To run the test suite, run

```
make test
```

To run the linter, run
```
make lint
```

To build docs, run
```
make doc
```
