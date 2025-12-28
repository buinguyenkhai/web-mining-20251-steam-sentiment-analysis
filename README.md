# Steam Game Review Sentiment Analysis

A deep learning approach for sentiment classification of Steam game reviews, comparing multiple architectures including BiGRU with Attention and TextCNN, enhanced with metadata features.

## Project Overview

This project implements and compares multiple sentiment analysis models on Steam game reviews:
- **Baseline**: Naive Bayes + TF-IDF
- **BiGRU + Attention + Metadata**: Bidirectional GRU with attention mechanism, incorporating game metadata
- **TextCNN + Metadata**: Convolutional neural network for text, with metadata features

**Best Performance**: BiGRU+Attention+Metadata achieves **90.2% F1-score** on the test set.

## Dataset

- **Source**: Steam API (custom scraping scripts in `data_scrape/`)
- **Total Reviews**: 40,567
- **Train/Test Split**: 34,481 / 6,086 (85% / 15%)
- **Games**: 48 games across 12 genres
- **Labels**: Balanced (50% positive, 50% negative)
- **Location**: `processed_data/train.csv` and `processed_data/test.csv`


### Metadata Features
- **Genre**: 12 game categories (action_fps, rpg_action, horror, etc.)
- **Playtime Tier**: short, medium, long, unknown
- **Review Length Tier**: short, medium, long


## Evidence of Performance

Performance results are documented in:
- `results/` - JSON files with detailed metrics
- Notebook cell outputs - Training logs and evaluation results
- Summary tables below

## Results

### Main Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes + TF-IDF | 0.860 | 0.880 | 0.833 | 0.856 |
| BiGRU+Attention+Metadata | **0.901** | 0.893 | **0.911** | **0.902** |
| TextCNN+Metadata | 0.878 | 0.850 | 0.917 | 0.882 |

## Project Structure

```
├── data_scrape/              # Steam API scraping scripts
├── processed_data/           # Train/test datasets
│   ├── train.csv
│   ├── test.csv
│   └── selected_games.csv
├── notebooks/                # Training notebooks
├── ablation_study/           # Ablation experiments
├── results/                  # Model results and best parameters
├── models/                   # Saved model weights
├── app/                      # Streamlit demo application
```

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```
- Install [PyTorch](https://pytorch.org/get-started/locally/) using the official installer. Make sure your NVIDIA driver supports the CUDA version you choose. 

### Training Models
Run the notebooks in order:
1. `00_exploratory_data_analysis.ipynb` - Data exploration
2. `01_data_preprocessing.ipynb` - Text preprocessing
3. `02_train_fasttext_embeddings.ipynb` - Train FastText embeddings
4. `03_baseline_naive_bayes.ipynb` - Baseline model
5. `04_bigruatt_metadata.ipynb` - BiGRU+Attention model
6. `06_textcnn_metadata.ipynb` - TextCNN model

### Running the Demo App
```bash
cd app
streamlit run app.py
```