# ROGII Wellbore Geology Prediction: Automating Geosteering

This project automates the geosteering process using a Hybrid Deep Learning model (1D-CNN + BiLSTM). It predicts the True Vertical Thickness (TVT) of geological formations in real-time based on sensory data, specifically Gamma Ray (GR) and Absolute Depth (Z).

**🏆 Competition Link:** [ROGII Wellbore Geology Prediction on Kaggle](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
We use `conda` to manage environments and dependencies. You don't need to manually install packages; let Conda figure it out for you!

First, ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed. Then, create the environment from the provided `environment.yml` file:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate geosteering
```
This will automatically install Python 3.10, PyTorch (with CUDA support), Pandas, Numpy, Scikit-Learn, Jupyter, and other necessary libraries.

### 2. Project Structure
- `data/train/` - Contains the horizontal well CSVs used for training.
- `data/test/` - Contains the hidden wells used for inference.
- `src/model.py` - Defines the PyTorch Hybrid 1D-CNN + BiLSTM architecture.
- `src/dataset.py` - Custom PyTorch Dataset for loading sequence windows.
- `src/train_full.py` - The main training loop with early stopping, dynamic learning rates, and dataset splitting.
- `notebooks/inference_all_wells.ipynb` - Jupyter Notebook to run inference on the test set, plot the predictions, and generate a submission file.
- `documentation.tex` - Comprehensive theoretical overview of the project.

### 3. Training the Model
To train the model from scratch, simply run the full training script. The script automatically handles splitting the training data by well IDs to ensure the model validates on completely unseen geology.

```bash
cd src
python train_full.py
```

*Note: The best model weights will be saved automatically to `src/models/best_geosteering_model.pth` along with the scaling statistics (`normalization_stats.json`).*

### 4. Running Inference & Plotting
Once you have trained the model, you can test it on unseen data and visualize the predictions.

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/inference_all_wells.ipynb`.
3. Run all cells. 
   - This notebook will load the trained model and normalization statistics.
   - It will predict the missing TVT gaps ("blind stretches") in the test wells.
   - Post-Processing: It automatically applies a Tie-In Anchoring Algorithm to align the model's 
   relative predictions perfectly with the last known physical TVT ground truth, preventing spatial drift.
   - It will save matplotlib plots for each well in the `notebooks/img/` folder.
   - Finally, it will generate a `submission.csv` file for Kaggle.

## 🧠 Model Architecture
- **1D-CNN Block**: Extracts complex hierarchical spatial features from the raw GR and Z curves across a 100-foot historical window.
- **BiLSTM Block**: Reads the CNN feature sequences bidirectionally to form a robust contextual understanding of the geological stratigraphy.
- **Fully Connected Head**: Outputs the continuous scalar prediction for the True Vertical Thickness.

For an in-depth understanding of the preprocessing pipeline, missing data handling, and normalization strategy, please refer to the `Grand_Challenges.pdf` file!
