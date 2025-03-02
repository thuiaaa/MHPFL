# MHPFL：Model Heterogenous Personalized Federated Learning
We create this repo to combine MoE with FL work. We learn some of the code structure from [HTFLLIB-MAIN](https://github.com/TsingZ0/HtFLlib)

## To Simulate our experiment
### Set environments

```bash
conda env create -f env_cuda_latest.yaml
conda activate fl
```

### Generate Dataset

```bash
cd dataset
python generate_Cifar10.py --noniid True --balance True --partition path --num_clients 10 #partition should be pat/dir
```