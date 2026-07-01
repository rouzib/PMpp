# Installation

PM++ requires Python 3.10 or newer. For development, install the package in editable mode:

```bash
python -m pip install -e .
```

Documentation dependencies are optional:

```bash
python -m pip install -e ".[docs]"
```

Read the Docs builds use CPU-only infrastructure. The docs configuration disables notebook execution and uses CPU-safe dependencies; multi-GPU examples are presented as source code to run on suitable local or cluster hardware, not during documentation builds.
