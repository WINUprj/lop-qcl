# Analysis of Loss of Plasticity in Quantum Continual Learning

The minimal implementation to experiment the loss of plasticity in Quantum Continual Laerning.

### Requirements
- Python (>= 3.10.13)
- Otherwise written in `requirements.txt`

### Installation and Setup
1. Clone this repo by running the following command:

```
git clone git@github.com:WINUprj/lop-qcl.git
```

2. Setup the virtualenv with python and activate:
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install all the packages with pip
```
pip install -r requirements.txt
pip install -e .
```

## Instruction to Run the Experiments
### Run Training Loops
All the experiments are managed through the config files stored in `configs/` directory.
By specifying the 

### Some Important Parameters for 

## Run Notebooks
There is a Jupyter notebook which demonstrates the training loop (`notebooks/submission.ipynb`).
If you wish to run this code, activate the Jupyter Lab and then execute the notebook from there.
Jupyter lab can be activated by running the following code:

```
jupyter lab
```