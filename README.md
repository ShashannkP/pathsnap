# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. Noisy GPS tracks are matched to a road network with a candidate classifier plus HMM/Viterbi decoding, served via FastAPI (Laptop 1) and a UI (Laptop 2).

## Repository layout
- `data_processing/` reading, cleaning, candidate generation, feature engineering
- `models/` training and comparison of 5+ models
- `tuning/` Bayesian hyperparameter optimisation
- `hmm/` HMM and Viterbi route reconstruction
- `api/` FastAPI backend
- `ui/` evaluator interface
- `viz/` Folium maps and plots
- `results/` CSVs, logs, configs, summaries
- `report/` technical report
- `data/` raw dataset (not committed; place here locally)

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running
TODO: training, API (Laptop 1) and UI (Laptop 2) instructions.
