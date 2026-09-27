# EN3150 Assignment 03

## Resource-Constrained CNN for Edge Image Classification

This repository contains the group work for **EN3150 - Pattern Recognition, Assignment 03**.

The assignment focuses on designing, training, and evaluating lightweight Convolutional Neural Networks (CNNs) for image classification under resource constraints.

The selected dataset for this project is **OCTMNIST**, a retinal Optical Coherence Tomography (OCT) image dataset from the MedMNIST collection.

> **Current Stage:**  
> The project is currently in the environment setup and dataset preparation stage.  
> Detailed information about the model architectures, experiments, results, and conclusions will be added as the project progresses.

---

# Dataset

## OCTMNIST

The project uses:

**OCTMNIST - Retinal Optical Coherence Tomography Image Dataset**

The **64 × 64** version of the dataset is used:

```text
octmnist_64.npz
```

OCTMNIST contains retinal OCT images belonging to four classes:

- Choroidal Neovascularization (CNV)
- Diabetic Macular Edema (DME)
- Drusen
- Normal

The dataset is obtained from the official **MedMNIST distribution**.

The dataset file is not included in this repository due to its size.

Each contributor should download:

```text
octmnist_64.npz
```

and place it inside the local `data/` directory.

The expected location is:

```text
data/octmnist_64.npz
```

---

# Project Structure

The project is organized as follows:

```text
EN3150_Assignment_03/
│
├── data/
│   └── octmnist_64.npz
│
├── notebooks/
│   └── Experimental and exploratory notebooks
│
├── src/
│   └── Reusable project components
│
├── results/
│   └── Generated plots, figures, tables and results
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Directory Descriptions

### `data/`

Contains dataset files used by the project.

The dataset is stored locally and is not included in the repository.

### `notebooks/`

Contains Jupyter notebooks used for:

- dataset exploration
- preprocessing experiments
- model development
- training
- evaluation
- result analysis

### `src/`

Contains reusable project components that may be developed during the project.

### `results/`

Contains generated outputs such as:

- training and validation curves
- confusion matrices
- evaluation results
- model comparison tables
- figures used in the final report

---

# Environment

The project currently uses:

- Python 3.11
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- Jupyter
- IPython Kernel

The exact Python package versions used in the working development environment are stored in:

```text
requirements.txt
```

The requirements file was generated from a successfully tested environment to make it easier for all contributors to reproduce the same setup.

---

# Getting Started

## 1. Open the Project Directory

After obtaining the project repository, open a terminal inside the project directory.

The terminal should be located at the same level as:

```text
README.md
requirements.txt
data/
notebooks/
src/
results/
```

---

## 2. Check the Python Version

The project is currently developed using **Python 3.11**.

Check the installed Python version using:

```bash
python --version
```

The output should indicate Python 3.11.

---

## 3. Create a Virtual Environment

Each contributor should create their own Python virtual environment.

### Windows

```powershell
python -m venv .venv
```

Activate the environment using:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
```

Activate the environment using:

```bash
source .venv/bin/activate
```

After activation, the terminal should indicate that the `.venv` environment is active.

For example:

```text
(.venv)
```

---

## 4. Upgrade pip

After activating the virtual environment:

```bash
python -m pip install --upgrade pip
```

---

## 5. Install Project Dependencies

All required Python packages and their versions are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

This will recreate the Python environment used for the project.

---

## 6. Verify the Environment

After installation, confirm that the main libraries can be imported successfully.

The environment should support:

- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- Jupyter

If these libraries import without errors, the environment is ready for development.

---

# Dataset Setup

Download the **64 × 64 version of OCTMNIST**.

The required file is:

```text
octmnist_64.npz
```

Place the downloaded file inside:

```text
data/
```

The resulting path should be:

```text
data/octmnist_64.npz
```

The dataset file should remain local and should not be committed to the repository.

---

# Starting Jupyter

Jupyter is included in the project dependencies.

Start Jupyter Notebook using:

```bash
jupyter notebook
```

Alternatively, if JupyterLab is available:

```bash
jupyter lab
```

Project notebooks should be stored inside:

```text
notebooks/
```

---

# Before Starting Work

Before beginning development, make sure that:

- Python 3.11 is installed
- the `.venv` virtual environment has been created
- the virtual environment is active
- all packages in `requirements.txt` have been installed
- `octmnist_64.npz` is available inside `data/`
- Jupyter starts successfully

---

# Dependency Management

The current Python environment is recorded in:

```text
requirements.txt
```

If a new Python package becomes necessary during the project:

1. Install the package inside the active virtual environment.
2. Confirm that it works correctly.
3. Update `requirements.txt`.
4. Inform the other group members that the environment requirements have changed.

The requirements file can be regenerated using:

```bash
pip freeze > requirements.txt
```

Avoid installing unnecessary libraries so that the development environment remains manageable.

---

# Files That Should Remain Local

The following files and directories should not be added to the repository:

- `.venv/`
- dataset files
- `.npz` files
- Python cache files
- Jupyter checkpoint files
- temporary files
- unnecessary IDE-specific files

These should be handled through `.gitignore`.

---

# Assignment Workflow

The project will broadly follow the workflow below:

```text
Dataset Setup
      ↓
Dataset Exploration
      ↓
Data Preparation
      ↓
Training / Validation / Test Split
      ↓
Standard CNN - Model A
      ↓
Lightweight CNN - Model B
      ↓
Optimizer Comparison
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Lightweight Pre-trained Model Comparison
      ↓
Accuracy / Memory / Computational Cost Analysis
      ↓
Final Discussion
```

---

# Project Objectives

The main stages of the assignment include:

1. Dataset preparation and exploration
2. Creation of training, validation, and test datasets
3. Design of a standard CNN
4. Design of a lightweight CNN using depthwise separable convolutions
5. Trainable parameter analysis
6. Activation function selection
7. Optimizer selection and comparison
8. Model training
9. Training and validation loss analysis
10. Test-set evaluation
11. Accuracy analysis
12. Precision and recall analysis
13. Confusion matrix analysis
14. Model-size comparison
15. Computational-cost comparison
16. Comparison with lightweight pre-trained architectures
17. Analysis of the trade-off between accuracy, memory footprint, and computational cost

---

# Planned Project Documentation

As the project progresses, this README may be expanded to include:

- detailed dataset description
- dataset statistics
- class distribution
- preprocessing methodology
- Model A architecture
- Model B architecture
- trainable parameter calculations
- activation function justification
- optimizer selection
- training configuration
- training and validation results
- test-set evaluation
- confusion matrices
- precision and recall results
- model-size comparison
- computational-cost analysis
- lightweight pre-trained model comparison
- final trade-off discussion
- conclusions

---

# Current Development Status

- [x] Dataset selected
- [x] OCTMNIST 64 × 64 downloaded
- [x] Python virtual environment created
- [x] Required libraries installed
- [x] Libraries successfully imported
- [x] `requirements.txt` generated
- [x] Initial project structure created
- [ ] Dataset exploration
- [ ] Dataset statistics and class distribution analysis
- [ ] Data preparation
- [ ] 70 / 15 / 15 data split
- [ ] Model A design
- [ ] Model B design
- [ ] Parameter calculations
- [ ] Optimizer experiments
- [ ] Model training
- [ ] Model evaluation
- [ ] Lightweight pre-trained model comparison
- [ ] Final trade-off analysis
- [ ] Final documentation