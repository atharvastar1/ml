# 🧬 Cinematic Curator: Neural Graph Recommendation System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26+-ff4b4b.svg)

An advanced movie recommendation engine powered by **Graph Neural Networks (GNN)**. This system leverages the **LightGCN** architecture to capture high-order collaborative filtering signals through neural graph propagation, combined with a deep MLP ranking layer for precision.

## 🚀 Key Features

*   **Graph-Native Intelligence**: Uses a 3-layer LightGCN model to propagate embeddings across the user-item graph.
*   **Hybrid Ranking Pipeline**: Combines graph neural scores with metadata-aware MLP ranking (Feature Engineering).
*   **Real-time Simulation**: Built-in "Kafka-style" stream ingestion to see recommendations update dynamically.
*   **Neural Latent Space Explorer**: Interactive 2D/3D visualization of user and movie embeddings.
*   **Dual-Service Architecture**: 
    - **Backend**: High-performance FastAPI serving layer.
    - **Frontend**: Modern, cinematic Streamlit dashboard.

## 🏗️ Architecture

1.  **Retrieval Stage**: ANN search using **FAISS** on the GNN-generated latent space.
2.  **Ranking Stage**: Deep MLP layer that incorporates movie metadata and user profiles.
3.  **Explanation Layer**: Neural propagation weights translated into human-readable explanations.

## 📁 Project Structure

```text
├── api/                # FastAPI serving logic
├── ui/                 # Streamlit dashboard & templates
├── src/                # Core neural logic
│   ├── lightgcn_model.py   # GNN Architecture
│   ├── ranking_layer.py    # MLP Ranking Logic
│   ├── inference.py        # Recommendation Engine
│   └── ...
├── notebooks/          # Training & Research (Jupyter)
├── saved_model/        # Trained weights & embeddings
├── data/               # Processed datasets
└── requirements.txt    # System dependencies
```

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.11+
- Virtual Environment recommended

### 2. Setup Environment
```bash
# Create and activate venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the System
Start both services to experience the full pipeline:

**Start Backend (API):**
```bash
python3 api/api.py
```

**Start Frontend (UI):**
```bash
streamlit run ui/app_streamlit.py
```

Access the UI at: `http://localhost:8501`

## 📊 Training
The model is trained on the MovieLens dataset. Research and training workflows can be found in the `notebooks/` directory:
- `gnn-based-recommendation-system.ipynb`: Full training pipeline with 20M dataset support.

## 🛡️ License
MIT License. Created for advanced machine learning research.
