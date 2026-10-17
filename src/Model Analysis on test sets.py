from src.data_preparation import OCTMNISTDataPreprocessor
from src.ModelA_init import build_model as build_model_a
from src.ModelB_init import build_model as build_model_b
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,ConfusionMatrixDisplay,confusion_matrix,classification_report)

# Load the OCTMNIST dataset and create new stratified splits
preprocessor=OCTMNISTDataPreprocessor(data_path="../data/octmnist_64.npz")
X_train, X_val, X_test, y_train, y_val, y_test=preprocessor.load_dataset()

#building the models
ModelA=build_model_a(verbose=False)
ModelB=build_model_b(verbose=False)

#Load the paths for trained weights for both models
modela_weights_path=Path("../notebooks/results/weights/model_a_final.weights.h5")
modelb_weights_path=Path("../notebooks/results/weights/model_b_final.weights.h5")

#Load the trained weights for both models
ModelA.load_weights(modela_weights_path)
ModelB.load_weights(modelb_weights_path)

#predict on the test set using both models
y_pred_a=ModelA.predict(X_test)
y_pred_b=ModelB.predict(X_test)

# True class labels
y_true = np.asarray(y_test).reshape(-1)

# Convert predicted probabilities to class labels
y_pred_a = np.argmax(y_pred_a, axis=1)
y_pred_b = np.argmax(y_pred_b, axis=1)
class_names = ["CNV", "DME", "Drusen", "Normal"]
predictions = {"Model A": y_pred_a,"Model B": y_pred_b}
results = []
confusion_matrices = {}
for name, y_pred in predictions.items():
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,average="macro",zero_division=0)
    recall = recall_score(y_true, y_pred,average="macro",zero_division=0)
    f1 = f1_score(y_true, y_pred,average="macro",zero_division=0)
    cm = confusion_matrix(y_true, y_pred,labels=[0, 1, 2, 3])
    confusion_matrices[name] = cm
    results.append({"Model": name,"Accuracy": accuracy,"Macro Precision": precision,"Macro Recall": recall,"Macro F1": f1})
    print(f"\n{'=' * 50}")
    print(name)
    print(f"{'=' * 50}")
    print(classification_report(y_true,y_pred,labels=[0, 1, 2, 3],target_names=class_names,zero_division=0))
    print("Confusion Matrix:")
    print(cm)

# Final comparison table
results_df = pd.DataFrame(results)
print("\nFinal Test Results:")
print(results_df.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
    disp.plot(ax=ax,cmap="Blues",values_format="d",colorbar=False)
    ax.set_title(f"{name} - Test Confusion Matrix")
plt.tight_layout()
plt.show()

'''
Uncomment the following lines to save the confusion matrix figure
=================================================================================================================================
# Create output directories
results_dir = Path("../results")
metrics_dir = results_dir / "metrics"
figures_dir = results_dir / "figures"
metrics_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
# Save overall test metrics
results_df.to_csv(metrics_dir / "test_metrics.csv",index=False)
# Save confusion matrices and per-class metrics
for name, y_pred in predictions.items():
    filename = name.lower().replace(" ", "_")
    cm = confusion_matrices[name]
    pd.DataFrame(cm,index=class_names,columns=class_names).to_csv(metrics_dir / f"{filename}_confusion_matrix.csv")
    report = classification_report(y_true,y_pred,labels=[0, 1, 2, 3],target_names=class_names,output_dict=True,zero_division=0)
    pd.DataFrame(report).transpose().to_csv(metrics_dir / f"{filename}_classification_report.csv")
# Save the combined confusion matrix figure
fig.savefig(figures_dir / "test_confusion_matrices.png",dpi=300,bbox_inches="tight")
print("Evaluation results saved successfully.")
=================================================================================================================================
'''

