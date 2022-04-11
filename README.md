Implementation of paper "Federated Collaborative Transfer for Recommendaiton" in SIGIR2021

Full code is still under optimization. Please contact Shuchang for temporary code.

citation:

```
@inproceedings{liu2021fedct,
  title={FedCT: Federated Collaborative Transfer for Recommendation},
  author={Liu, Shuchang and Xu, Shuyuan and Yu, Wenhui and Fu, Zuohui and Zhang, Yongfeng and Marian, Amelie},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={716--725},
  year={2021}
}
```

## 0. Setup

```
conda create -n fedct python=3.9
conda activate fedct
conda install -c anaconda ipykernel
python -m ipykernel install --user --name fedct --display-name "FedCT"
conda install pytorch cudatoolkit=11.3 -c pytorch -c conda-forge 
conda install -c conda-forge scikit-learn
conda install -c conda-forge tqdm 
conda install pandas
conda install setprotitle
pip install matplotlib
```

# 1. Domain Environment

### 1.1 Preprocess

Run notebook 'data/Amazon.ipynb'

Corresponding data will be saved in the specified directory 'XXX/domain_data/' with the following sub contents:
* multicore_data/
* tsv_data/
* meta_data/

Then run:
> bash run_domain_reader.sh

This will save data readers in 'XXX/domain_data/reader/'

### 1.2 Train

> bash run_domain_pretrain.sh

Training logs and models will be saved in:
* XXX/env_logs/
* XXX/env_models/

# 2. Cold-start User Transfer Task

### 2.1 Preprocess

Run notebook 'data/Amazon_FedCT.ipynb' with your specified target_path (e.g. 'XXX/transfer_data/')

Corresponding data will be saved in the specified directory target_path with the following sub contents:
* id_train.tsv
* domain_model_logs.txt
* \$\{target_domain\}_val.tsv
* \$\{target_domain\}_test.tsv


### 2.2 Train

> bash run_emcdr.sh

> bash run_cue.sh

Training logs and models will be saved in:
* XXX/transfer_logs/
* XXX/transfer_models/

# 3. Result Observation

Notebooks:
* DomainModelResult.ipynb: domain-specific model results, standard recommendation, no cold-start
* ColdStartTransferResult.ipynb: EMCDR and DUE results
