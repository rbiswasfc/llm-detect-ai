This repo contains our code and configurations for the **LLM - Detect AI Generated Text** competition. The summary of the solution is posted [here](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121). Please refer to the following sections for details on training and dependencies. 

## Section 1: Setup
### 1.1 Hardware
**Jarvislabs.ai** was our primary source of compute. Specifically, models were trained on the following instance:

Ubuntu 20.04.5 LTS (128 GB boot disk)
Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz (7 vCPUs)
4 x NVIDIA A100 40GB GPU OR 4 x NVIDIA A6000 48GB GPU

### 1.2 Software
I used PyTorch-2.1 image from Jarvislabs.ai, which comes with:

* Python 3.10.11
* CUDA 12.3
* Python packages installation: pip install -r requirements.txt

### 1.3 Datasets
Please make sure Kaggle API is installed. Then run the following script to download the required datasets:

```
chmod +x ./setup.sh
./setup.sh
```

Please note that the above script will create a datasets folder in the directory located one level above the current directory. The external datasets will be downloaded in the datasets folder.

## Section 2: Training

### 2.1 LLM Models
For (Q)LoRA fine-tuning of the LLM models, please run the following commands:
```
accelerate launch ./code/train_r_detect.py \
--config-name conf_r_detect_mix_v16 \
use_wandb=false
```

```
accelerate launch ./code/train_r_detect.py \
--config-name conf_r_detect_mix_v26 \
use_wandb=false
```

Please note that 4xA100 GPUs on a single node was used for LLM fine-tuning. Training takes ~3 hours for mix_v16 and ~4 hours for mix_v26.

### 2.2 DeBERTa Ranking Models

To training the `deberta-v3-large` model with ranking loss, please run the following command:

```
accelerate launch ./code/train_r_ranking.py \
--config-name conf_r_ranking_large \
use_wandb=false
```

### 2.3 Embedding model

We trained an embedding model with supervised contrastive loss to find similar essays (KNN neighbors) for a given essay in the test set.

```
accelerate launch ./code/train_r_embed.py \
--config-name conf_r_embed \
use_wandb=false
```

## Section 3: Text Generation

We fine-tuned a wide variety of LLMs using the CLM objective on [PERSUADE](https://www.kaggle.com/datasets/nbroad/persaude-corpus-2) corpus to produce student like essays. The fine-tuned checkpoints were uploaded as a Kaggle Dataset `conjuring92/detect-ai-persuade-clm-ckpts`. These checkpoints can be used to generate essays using the following commands:

```
accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate.yaml

accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_tiny_llama.yaml

accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_pythia.yaml

accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_bloom.yaml

accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_gpt2.yaml

accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_opt.yaml

accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_falcon.yaml

accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_mpt.yaml

accelerate launch ./detect-ai/code/generate_r_clm.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_llama13b.yaml

accelerate launch ./detect-ai/code/generate_r_clm_from_scratch.py \
--config_path ./detect-ai/conf/r_clm/conf_r_clm_generate_mistral_persuade.yaml
```

Optionally, the fine-tuning of LLMs for text generation can be done using the following commands:

```
accelerate launch ./detect-ai/code/train_r_clm.py \
--config-name conf_r_clm_tiny_llama \
use_wandb=false

accelerate launch ./detect-ai/code/train_r_clm.py \
--config-name conf_r_clm_pythia \
use_wandb=false

accelerate launch ./detect-ai/code/train_r_clm.py \
--config-name conf_r_clm_bloom \
use_wandb=false

accelerate launch ./detect-ai/code/train_r_clm.py \
--config-name conf_r_clm_gpt2 \
use_wandb=false

accelerate launch ./detect-ai/code/train_r_clm.py \
--config-name conf_r_clm_opt \
use_wandb=false

accelerate launch ./detect-ai/code/train_r_clm.py \
--config-name conf_r_clm_falcon \
use_wandb=false

accelerate launch ./detect-ai/code/train_r_clm.py \
--config-name conf_r_clm_mpt \
use_wandb=false

accelerate launch ./detect-ai/code/train_r_clm.py \
--config-name conf_r_clm_llama13b \
use_wandb=false

accelerate launch ./detect-ai/code/train_r_clm_from_scratch.py \
--config-name conf_r_clm_mistral_persuade \
use_wandb=false
```