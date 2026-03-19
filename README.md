# MOMA

Open-source code for MOMA: Multi-Objective Memetic Attack for Multi-Agent Reinforcement Learning.


## Installation instructions

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda create -n MOMA python=3.8 -y
conda activate MOMA

bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```


## Command Line Tool

**Run an experiment**

```shell
# For SMAC
conda activate MOMA
CUDA_VISIBLE_DEVICES=2 python3 src/main.py --config=MOMA --env-config=sc2 with env_args.map_name=1c3s5z
```


