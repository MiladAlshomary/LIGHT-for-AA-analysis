# LIGHT

This repository accompanies the paper **"Layered Insights: Generalizable Analysis of Authorial Style by Leveraging All Transformer Layers"**. The study introduces a novel approach to the authorship attribution task by utilizing linguistic representations learned at different layers of pre-trained transformer-based models. Our method is tested across three datasets and compared against a state-of-the-art baseline in both in-domain and out-of-domain settings. The results demonstrate that leveraging multiple transformer layers enhances the robustness of authorship attribution models, particularly for out-of-domain data, achieving new state-of-the-art performance.  

## Installation  
Execute the following commands to create a virtual environment and install the necessary dependencies:  

```bash
python3 -m venv multiluar
. ./multiluar/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```  

## Configuring the Python Path  

After cloning the repository, update the Python path to point to the project's directory:  

```bash
./export PYTHONPATH="<path to MultiLUAR>:$PYTHONPATH"
```  

## Downloading Data and Pre-trained Weights  

Once the environment is set up, run the following commands to download the SBERT pre-trained weights and preprocess the datasets:  

### Pre-trained Weights  

First, install Git Large File Storage (LFS) by following the instructions [here](https://git-lfs.github.com). Then, download the weights using:  

```bash
./scripts/download_sbert_weights.sh
```  

### Reddit Dataset  

Reddit has updated its [Data API terms](https://www.redditinc.com/policies/data-api-terms), prohibiting the use of user-generated data for machine learning unless explicit permission is granted. As a result, we provide only the comment identifiers used for training our models:  

| Dataset Name | Download Link |  
|-------------|--------------|  
| [MUD](https://arxiv.org/abs/2105.07263) | [Google Drive Link](https://drive.google.com/file/d/16YgK62cpe0NC7zBvSF_JxosOozG-wxou/view?usp=drive_link) |  

### Amazon Dataset  

The Amazon dataset must be requested from [this source](https://nijianmo.github.io/amazon/index.html#files) under the "raw review data" (34GB) section. After downloading, move the files to `./data/raw_amazon` and execute the preprocessing script:  

```bash
./scripts/preprocess_amazon_data.sh
```  

### Fanfiction Dataset  

The fanfiction dataset can be obtained from [this repository](https://zenodo.org/record/3724096#.YT942y1h1pQ). Once downloaded, place `data.jsonl` and `truth.jsonl` from the large dataset into `./data/pan_paragraph`, then run the preprocessing script:  

```bash
./scripts/preprocess_fanfiction_data.sh
```  

## Path Configuration  

Modify the paths in `file_config.ini` to customize storage locations:  
- **output_path**: Directory for experiment results and model checkpoints (default: `./output`).  
- **data_path**: Location for dataset storage (default: `./data`).  
- **transformer_path**: Path for storing SBERT pre-trained weights (default: `./pretrained_weights`).  

We recommend setting custom paths according to your system.  

## Reproducing Results  

To replicate the results presented in the paper, use the scripts in the `./scripts/reproduce/` directory, with file names corresponding to each table (e.g., `table_N.sh`).  

## Training  

We provide training commands for **single-domain** and **multi-domain** models.  
- **Single-domain models** are trained on a single dataset.  
- **Multi-domain models** utilize two datasets for training.  

### Available Datasets  
- `raw_all` - Reddit Million User Dataset (MUD).  
- `raw_amazon` - Amazon Reviews dataset.  
- `pan_paragraph` - PAN Short Stories dataset.  

### Training Single-Domain Models  

#### Reddit Comments  
```bash
python main.py --dataset_name raw_all --do_learn --validate --gpus 0 --experiment_id reddit_model --approach multiluar
```  

#### Amazon Reviews  
```bash
python main.py --dataset_name raw_amazon --do_learn --validate --gpus 0 --experiment_id amazon_model --approach multiluar
```  

#### Fanfiction Stories  
```bash
python main.py --dataset_name pan_paragraph --do_learn --validate --gpus 0 --experiment_id fanfic_model --approach multiluar
```  

### Training Multi-Domain Models  

#### Reddit Comments + Amazon Reviews  
```bash
python main.py --dataset_name raw_all+raw_amazon --do_learn --validate --gpus 0 --experiment_id reddit_amazon_model --approach multiluar
```  

#### Amazon Reviews + Fanfiction Stories  
```bash
python main.py --dataset_name raw_amazon+pan_paragraph --do_learn --validate --gpus 0 --experiment_id amazon_stories_model --approach multiluar
```  

#### Reddit Comments + Fanfiction Stories  
```bash
python main.py --dataset_name raw_all+pan_paragraph --do_learn --validate --gpus 0 --experiment_id reddit_stories_model --approach multiluar
```  

## Evaluation  

To evaluate a trained model, use the following commands. Replace `<experiment_id>` with the identifier assigned during training (e.g., `reddit_model`, `amazon_model`, or `fanfic_model`).  

### Reddit Comments  
```bash
python main.py --dataset_name raw_all --evaluate --experiment_id <experiment_id> --load_checkpoint --approach multiluar
```  

### Amazon Reviews  
```bash
python main.py --dataset_name raw_amazon --evaluate --experiment_id <experiment_id> --load_checkpoint --approach multiluar
```  

### Fanfiction Stories  
```bash
python main.py --dataset_name pan_paragraph --evaluate --experiment_id <experiment_id> --load_checkpoint --approach multiluar
```  

## Experiments with Baseline  

To run experiments with the **baseline LUAR** model for both single-domain and multi-domain scenarios, simply change the `--approach` flag to `'baseline'`.  

### Training Single-Domain Models with Baseline  

#### Reddit Comments  
```bash
python main.py --dataset_name raw_all --do_learn --validate --gpus 0 --experiment_id reddit_baseline --approach baseline
```  

#### Amazon Reviews  
```bash
python main.py --dataset_name raw_amazon --do_learn --validate --gpus 0 --experiment_id amazon_baseline --approach baseline
```  

#### Fanfiction Stories  
```bash
python main.py --dataset_name pan_paragraph --do_learn --validate --gpus 0 --experiment_id fanfic_baseline --approach baseline
```  

### Training Multi-Domain Models with Baseline  

#### Reddit Comments + Amazon Reviews  
```bash
python main.py --dataset_name raw_all+raw_amazon --do_learn --validate --gpus 0 --experiment_id reddit_amazon_baseline --approach baseline
```  

#### Amazon Reviews + Fanfiction Stories  
```bash
python main.py --dataset_name raw_amazon+pan_paragraph --do_learn --validate --gpus 0 --experiment_id amazon_stories_baseline --approach baseline
```  

#### Reddit Comments + Fanfiction Stories  
```bash
python main.py --dataset_name raw_all+pan_paragraph --do_learn --validate --gpus 0 --experiment_id reddit_stories_baseline --approach baseline
```  

By adjusting the `--approach` flag, you can compare our **MultiLUAR** approach with the baseline model across different datasets and scenarios.