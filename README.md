# EN3150 Assignment 03

## Resource-Constrained CNN for Edge Image Classification

Group work for **EN3150 – Pattern Recognition, Assignment 03**: designing, training, and evaluating lightweight convolutional neural networks for image classification under resource constraints.

The project uses **OCTMNIST at 64 × 64 resolution**, a retinal Optical Coherence Tomography (OCT) image dataset from MedMNIST, with four classes: CNV, DME, Drusen, and Normal.

> **Current stage: data preparation and Model A initialization are implemented.**
> Dataset exploration code, a reusable stratified data splitter, and the standard CNN architecture are available. Example notebooks record successful data loading, model construction, and a forward pass. Training, evaluation, and comparisons with lightweight models are still pending.

## Environment setup

Run these commands from the repository root, where `README.md`, `requirements.txt`, and `src/` are located.

### 1. Create and activate a Python 3.11 environment

The local development environment uses **Python 3.11.9**. Create a separate virtual environment for this project.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Verify the active interpreter:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

The executable path should point inside the project's `.venv` directory.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned requirements include TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit-learn, Jupyter Notebook/Lab, and IPython Kernel. `requirements.txt` records the Windows development environment and includes `pywinpty`; installation on Linux/macOS may require a platform-adjusted requirements file. A fresh installation on those platforms has not been verified.

Check the main imports:

```bash
python -c "import tensorflow as tf; import numpy, pandas, matplotlib, sklearn, jupyter; print('TensorFlow:', tf.__version__); print('Environment imports OK')"
```

If PowerShell activation is unavailable, use `.\.venv\Scripts\python.exe` in place of `python` to run commands with the project interpreter.

### 3. Check the dataset

The required file is:

```text
data/octmnist_64.npz
```

The current repository tracks this dataset file. If it is missing from your checkout, obtain the **64 × 64 OCTMNIST** archive from the official MedMNIST distribution and place it at the path above.

The preprocessor expects these archive keys:

```text
train_images, train_labels, val_images, val_labels, test_images, test_labels
```

### 4. Start Jupyter with the project environment

```bash
python -m ipykernel install --user --name en3150-cnn --display-name "Python (EN3150 CNN)"
python -m notebook
```

Alternatively, start JupyterLab with `python -m jupyterlab`. Open a notebook in `notebooks/` and select **Python (EN3150 CNN)** as its kernel. In an IDE, select the project's `.venv` interpreter.

## Using the reusable components in `src/`

There is currently one reusable class, `OCTMNISTDataPreprocessor`, and one model builder function, `build_model`.

### Import paths and dataset location

Run this cell before the examples below. It supports both the repository root and `notebooks/` as the working directory, and makes `src` importable from notebooks:

```python
from pathlib import Path
import sys

project_root = Path.cwd().resolve()
if not (project_root / "src").is_dir():
    project_root = project_root.parent
if not (project_root / "src").is_dir():
    raise RuntimeError("Run from the repository root or its notebooks directory.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

data_path = project_root / "data" / "octmnist_64.npz"
```

### `OCTMNISTDataPreprocessor` — load and split the dataset

Defined in [`src/data_preparation.py`](src/data_preparation.py).

```python
from src.data_preparation import OCTMNISTDataPreprocessor

preprocessor = OCTMNISTDataPreprocessor(data_path=data_path)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.load_dataset()

print(X_train.shape, y_train.shape)
```

`load_dataset(val_size=0.15, test_size=0.15)`:

1. Loads and combines the original MedMNIST training, validation, and test splits.
2. Flattens labels into one-dimensional integer-label arrays.
3. Creates new stratified splits, preserving class proportions approximately. Defaults are **70% training / 15% validation / 15% test**.
4. Adds a channel axis so images have shape `(N, 64, 64, 1)` for TensorFlow.
5. Returns six arrays in the order shown above.

The saved data preparation example records:

| Split | Images | Image shape | Label shape |
| --- | ---: | --- | --- |
| Training | 76,516 | `(76516, 64, 64, 1)` | `(76516,)` |
| Validation | 16,396 | `(16396, 64, 64, 1)` | `(16396,)` |
| Test | 16,397 | `(16397, 64, 64, 1)` | `(16397,)` |

To change the split proportions, pass fractions of the full combined dataset:

```python
# Optional alternative: 80% training / 10% validation / 10% test
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.load_dataset(
    val_size=0.10,
    test_size=0.10,
)
```

Use positive validation and test fractions whose sum is less than one, with enough samples for stratification. This workflow creates a custom evaluation split; it does **not** retain the official MedMNIST test split.

**Current implementation details:**

- Pixel values are returned without normalization. Model A applies its own `1 / 255` rescaling, so do not divide these inputs by 255 again before passing them to that model.
- The constructor accepts `random_state=42`, but both internal split calls currently hardcode `42`. Changing the constructor argument does not change the split.
- Use the returned arrays: the instance attributes such as `preprocessor.X_train` remain `None` after loading.
- Loading and splitting happen in memory and create several arrays; allow sufficient RAM for the full dataset.

### `build_model` — initialize the standard CNN (Model A)

Defined in [`src/ModelA_init.py`](src/ModelA_init.py). This is a function that returns an **uncompiled, untrained** `tf.keras.Model`.

```python
from src.ModelA_init import build_model

model_a = build_model(
    input_shape=(64, 64, 1),
    num_classes=4,
    conv_filters=(32, 64, 128, 256),
    verbose=True,
)
model_a.summary()

# After loading the arrays with the preprocessor above:
predictions = model_a(X_train[:2], training=False)
print(predictions.shape)  # (2, 4)
```

The default architecture is:

```text
Input (64, 64, 1)
  → Rescaling (1 / 255)
  → Conv2D (32, 3×3, ReLU, same padding)  → MaxPooling2D (2×2)
  → Conv2D (64, 3×3, ReLU, same padding)  → MaxPooling2D (2×2)
  → Conv2D (128, 3×3, ReLU, same padding) → MaxPooling2D (2×2)
  → Conv2D (256, 3×3, ReLU, same padding) → MaxPooling2D (2×2)
  → GlobalAveragePooling2D
  → Dense (32, ReLU)
  → Dense (4, softmax)
```

The default model has **396,196 parameters**. `conv_filters` must contain exactly four positive integers. `verbose=True` prints construction details, output shapes, and the parameter count.

Each output row contains four class probabilities in label order: `0 = CNV`, `1 = DME`, `2 = Drusen`, `3 = Normal`. A forward pass checks input/output compatibility; predictions from the initialized model are not trained classification results.

The model still needs an optimizer and loss configured with `compile()` before `fit()`. The preprocessor returns integer labels, so a sparse categorical classification loss is appropriate. Training configuration and optimizer comparisons remain future work.

## Project structure and notebooks

```text
.
├── data/
│   └── octmnist_64.npz
├── notebooks/
│   ├── Dataset_exploration.ipynb
│   ├── data_preparation_example.ipynb
│   ├── ModelA_development.ipynb
│   └── ModelA_initialization_example.ipynb
├── src/
│   ├── data_preparation.py
│   └── ModelA_init.py
├── results/
├── EN3150_Assignment_03_In23.pdf
├── requirements.txt
└── README.md
```

| Notebook | Purpose |
| --- | --- |
| [`1.Dataset_exploration.ipynb`](notebooks/1.Dataset_exploration.ipynb) | Inspect archive contents, class counts/distributions, pixel values, and stratified splits. |
| [`2.data_preparation_example.ipynb`](notebooks/2.data_preparation_example.ipynb) | Demonstrate the reusable preprocessor and returned array shapes. |
| [`3.ModelA_development.ipynb`](notebooks/3.ModelA_development.ipynb) | Develop Model A layer by layer and check a forward pass. |
| [`4.ModelA_initialization_example.ipynb`](notebooks/4.ModelA_initialization_example.ipynb) | Combine the reusable preprocessor and model builder; print the model summary and check predictions. |

`results/` currently contains only a placeholder. It is intended for training curves, evaluation metrics, confusion matrices, and model comparison outputs.

## Current development status

This status reflects the source code and saved notebook outputs currently in the repository.

- [x] Initial project structure and Python environment requirements recorded.
- [x] OCTMNIST 64 × 64 dataset available.
- [x] Dataset exploration and class-distribution analysis code added.
- [x] Reusable data preparation class implemented.
- [x] Custom stratified 70 / 15 / 15 split and TensorFlow input shapes demonstrated.
- [x] Standard CNN (Model A) implemented as a reusable builder.
- [x] Default Model A parameter count reported and sample forward pass demonstrated.
- [ ] Finalize training configuration and activation-function justification.
- [ ] Design Model B using depthwise separable convolutions.
- [ ] Complete analytical parameter calculations and architecture comparisons.
- [ ] Run optimizer experiments and train the models.
- [ ] Analyze training and validation loss/accuracy curves.
- [ ] Evaluate test accuracy, precision, recall, and confusion matrices.
- [ ] Compare with lightweight pre-trained architectures.
- [ ] Compare model size, computational cost, and accuracy/resource trade-offs.
- [ ] Complete the final report and results documentation.

The next phase is to configure and train Model A, develop Model B, and record comparable experiments before final evaluation.

## Contributor notes

- Keep virtual environments, Python caches, notebook checkpoints, and temporary IDE files local; these are covered by `.gitignore`.
- The current `.gitignore` does **not** exclude dataset archives, and the OCTMNIST archive is already tracked. Avoid adding duplicate dataset copies or generated artifacts accidentally.
- Install new dependencies in the project environment, verify them, and update `requirements.txt` deliberately. To regenerate the environment snapshot, use `python -m pip freeze > requirements.txt` and review the diff before committing.
- Store reusable implementation code in `src/`, experiments in `notebooks/`, and generated analysis outputs in `results/`.
