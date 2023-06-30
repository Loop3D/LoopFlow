# LoopFlow

Generate graphs from Loop Models and calculate flow parameters based on these graphs

Two notebooks, one derived from Draw your own model example, the other hardwired to loading van der Wielen et al. Emmie Bluff gocad surfaces

Authors Mark Jessell & Guillaume Pirot

## Installation
```
git clone https://github.com/Loop3D/LoopFlow.git
cd LoopFlow
pip install .
```

*Note:* use `pip install . --verbose` or `pip install . -v` for printing (more) messages during the installation.

 **Remove LoopFlow**

`pip uninstall -y loopflow`

*Note: first remove the directory 'loopflow.egg-info' from the current directory (if present).*

## Using LoopFlow

Do not launch python from the directory where the installation has been done (with `pip`), otherwise `import loopflow` will fail.

## Requirements
The following python packages are used by LoopFlow (tested on python 3.10.9):
   - numpy
   - pandas
   - scipy
   - networkx
   - LoopStructural
   