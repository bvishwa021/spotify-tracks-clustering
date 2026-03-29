# 🎵 Spotify Tracks Clustering
### Teaching a machine to hear genres it was never told existed — through K-Means clustering on Spotify audio features.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=flat&logo=pandas&logoColor=white)

---

## Overview

End-to-end unsupervised ML pipeline that segments 100K+ Spotify tracks into interpretable audio archetypes — without ever using genre labels during modeling. Clusters are profiled, named, and validated against ground-truth genres to measure how much musical identity lives purely in audio signal.

---

## Pipeline

| # | Notebook | What happens |
|---|---|---|
| 1 | `01_eda.ipynb` | Feature distributions, correlation heatmap, genre radar profiles |
| 2 | `02_preprocessing.ipynb` | Log-transform skewed features, StandardScaler, PCA for visualization |
| 3 | `03_kmeans_clustering.ipynb` | Elbow method + silhouette score → optimal k → final model |
| 4 | `04_cluster_interpretation.ipynb` | Cluster profiling, human-readable naming, genre validation |

---

## Cluster Archetypes (Example)

| Cluster | Name | Signature |
|---|---|---|
| 0 | 🎻 Classical / Orchestral | High acousticness, high instrumentalness, low energy |
| 1 | 💃 High-Energy Dance / EDM | High danceability, high energy, high instrumentalness |
| 2 | 🎤 Hip-Hop / Rap | High speechiness, high danceability, low acousticness |
| 3 | 🌿 Acoustic Chill / Folk | High acousticness, low energy, low speechiness |
| 4 | 🔥 Energetic Rock | High energy, low acousticness, low instrumentalness |
| 5 | ☀️ Upbeat Pop | High valence, moderate danceability, moderate energy |

> These are illustrative.

---

## Key Design Decisions

- **Log1p on `instrumentalness` + `speechiness`** — both are zero-inflated; log-transform spreads the distribution so scaling has real signal to work with
- **StandardScaler over MinMaxScaler** — K-Means uses Euclidean distance; StandardScaler prevents high-range features like `tempo` from dominating
- **PCA for visualization only** — clustering runs on all 8 original features to preserve full signal; PCA is only used to see the clusters in 2D
- **Silhouette + Elbow together** — elbow is visual and subjective; silhouette gives an objective numeric confirmation of the best k

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/spotify-tracks-clustering.git
cd spotify-tracks-clustering
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and place `dataset.csv` in `data/raw/`. Then run notebooks in order.

---

## Results
>This will be finalised once project is completed.

- **Optimal k:** [fill in]
- **Best silhouette score:** [fill in]
- **Cluster–genre alignment:** [fill in — e.g. "Classical and Hip-Hop show near-pure membership; Pop and R&B overlap, reflecting shared audio properties"]

---

## Possible Extensions

- Build a track recommender: given a new song, find its cluster and surface similar tracks

---

## Data Source

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — Maharshi Pandya via Kaggle. Audio features from the Spotify Web API.

---

**Vishwa B.** · [LinkedIn](#) · [GitHub](#) · [Portfolio](#)
