# Multimodal Sentiment Analysis Project

This is a research project for multimodal sentiment analysis, integrating data preprocessing, model training, evaluation, and analysis.

## Directory Structure

```
.
├── cmff/                 # Code related to the CMFF model
│   ├── mlruns/           # MLFlow experiment logs
│   ├── config.py         # Configuration file
│   ├── lightning_data.py # PyTorch Lightning data module
│   └── lightning_model.py# PyTorch Lightning model
├── datasets/             # Datasets and preprocessing scripts
│   ├── cmff_cache.py     # CMFF data caching
│   ├── split_av.py       # Audio and video splitting
│   ├── extract_face.py   # Face data extraction
│   ├── extract_frames.py # Video frame extraction
│   ├── fix_video_for_qwen.py # Video format fixing for Qwen
│   ├── MOSEI/            # MOSEI dataset
│   ├── MOSI/             # MOSI dataset
│   └── ...
├── cmff/                 # Code related to the CMFF model
├── llm/                  # Scripts for Large Language Model testing and evaluation
├── utils/                # Common utility scripts
├── main.py               # Main entry point for model training
└── readme.md             # Project description
```

## Download Data

The three datasets (CMU-MOSI and CMU-MOSEI) are available from this link: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

## Data Preprocessing

The data preprocessing scripts are mainly located in the `datasets/` directory. The main steps are as follows:

1.  **Download Datasets**: Place the original datasets like `MOSEI`, `MOSI`, `SIMS`, etc., into their corresponding folders under the `datasets/` directory.
2.  **Feature Extraction**:
    *   Run `split_av` to split audio and video files.
3.  **Data Integration and Caching**: Run `cmff_preprocess.py` (or other corresponding preprocessing scripts) to process the extracted features, convert them into the format required for model training, and cache them to speed up subsequent data loading.

Specific preprocessing parameters can be modified within each script.

## How to Run the Model

Model training and evaluation are initiated via the `main.py` script.

1.  **Configure Parameters**:
    *   Depending on the model you want to train (e.g., `cmff` ), modify the `Config` file referenced in `main.py`, such as `cmff/config.py`.
    *   In the `config.py` file, you can set the dataset, model hyperparameters, learning rate, batch size, etc.

2.  **Start Training**:
    Run the main script directly from the terminal:
    
    ```bash
    python main.py
    ```
    
3.  **Experiment Tracking and Results**:
    
    *   This project uses `MLFlow` for experiment tracking. You can find the experiment logs in the `cmff/mlruns` directories.
    *   The trained model weights and records of the best model are saved in the `artifacts/checkpoints` subdirectory of the corresponding experiment in the `mlruns` directory.
    *   Evaluation results will be output after the training is complete.

