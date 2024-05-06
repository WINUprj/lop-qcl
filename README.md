# Analysis of Loss of Plasticity in Quantum Continual Learning

The codes to investigate the loss of plasticity in Quantum Continual Learning.

### Requirements
- Python (>= 3.10.13)
- Otherwise written in `requirements.txt`

### Installation and Setup
1. Clone this repo by running the following command:

```
git clone git@github.com:WINUprj/lop-qcl.git
```

2. Setup the virtualenv and activate it:
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install all the packages with pip:
```
pip install -r requirements.txt
pip install -e .
```

# Instruction to Run the Experiments
### Run Training Loops and Generate the Plots
All the experiments are managed through the config files stored in `configs/` directory.
Upon training, the selected config file will be used to generate the directory `results/['exp_name']` to save all the results and to specify the parameters of the components such as model, tasks, etc.
Here, 'exp_name' is one of the parameters in the config file.
If 'search_keys' in the config file is specified as a non-empty list of the valid parameter names in the same config file, the code will automatically detect and conduct the grid-search over the candidate values.

For example, with the `configs/vqc.json`, one can run the training loop by running:
```
python3 main.py --config_file=configs/vqc.json
```
Since `configs/vqc.json` specifies the 'step_size' and 'model_params.n_qnn_layers', it will sweep through all the possible combinations of the step-sizes and the number of ansatz layers.
Consult the code for further details.

Also, the same config file can be used to generate the plots:
```
python3 [plot_vqc.py | plot_classical.py] --config_file=[CONFIG_FILE]
```
The plots will be generated under the directory `plots/['exp_name']`.
Please note that the released plot scripts are specialized for the existing config files, so modification is needed to generate the clean plots if you change the config file. 
Also, the plot scripts are separated for the VQC and classical ANNs due to a reason described below.

### Some Important Notes for Plotting the Results from Classical ANNs
Sweeping over the 'model_params.layer_sizes' for the classical ANN is not supported. To conduct the grid-search over it, one needs to rerun the experiments manually with different 'model_params.layer_sizes' and 'exp_name'. Because this produces the results in separate directories under `results/`, running the plot script above will not capture all the swept results. To overcome this, manually modify the line 20 of `plot_classical.py` as follows:

```python
exp_dirs = [cfg["exp_dir"], cfg["root_dir"] / "results/DIRECTORY_FOR_DIFFERENT_LAYER_SIZES", cfg["root_dir"] / "results/ANOTHER_DIRECTORY_FOR_DIFFERENT_LAYER_SIZES",]
```
In other words, manually add the paths of the result directories.

### Replicating the Results in Report
Simply run the following commands.

VQC (run time ~ 2.5 hrs):
```
python3 main.py --config_file=configs/vqc.json
python3 plot_vqc.py --config_file=configs/vqc.json
```

Classical ANN (run time ~ 2.5 mins):
```
python3 main.py --config_file=configs/classical.json 
python3 plot_classical.py --config_file=configs/classical.json     // The results for the other two experiments are uploaded in this repo (`results/classical3`, `results/classical5`).
```

# Run Notebooks
There is a Jupyter notebook with a minimal demo showing the loss of plasticity in quantum continual learning (`notebooks/submission.ipynb`).
If you wish to run this code, activate the Jupyter Lab and then execute the notebook from there.
Jupyter lab can be activated by running the following code:

```
jupyter lab
```

# Report
The report can be found in `lop_qcl_report.pdf`.

# Special Mention
- Qingfeng Lan. Variational Quantum Soft Actor-Critic. arXiv preprint arXiv:2112.11921, 2021 [[Code](https://github.com/qlan3/QuantumExplorer)]
