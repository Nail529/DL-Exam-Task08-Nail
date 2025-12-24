# DL Exam Task 8: Food Classification (5 Types)

## 📌 Project Overview
This project demonstrates **Transfer Learning** using the **ResNet18** architecture to classify images from the **Food-101** dataset. The goal was to achieve a target accuracy of ≥85% by fine-tuning only the final classification layer of a pre-trained model.

## 🍔 Chosen Food Classes
For this task, the following 5 classes were selected:
1. Pizza
2. Hamburger
3. Sushi
4. Tacos
5. Ice Cream

## 🏗️ Model Architecture
- **Base Model:** `torchvision.models.resnet18(weights='IMAGENET1K_V1')`
- **Backbone:** Frozen (`requires_grad=False`) to preserve pre-trained features.
- **Classifier Head:** The original fully connected layer was replaced with:
  - `model.fc = nn.Linear(512, 5)`
- **Training:** Only the weights of the final `fc` layer were updated.

## 🧪 Experimental Setup
The project compares two training versions to analyze the impact of different optimization strategies:

| Configuration | Version 1 (Adam) | Version 2 (SGD) |
| :--- | :--- | :--- |
| **Optimizer** | Adam | SGD (with Momentum 0.9) |
| **Learning Rate** | 0.0001 | 0.001 |
| **Batch Size** | 32 | 32 |
| **Epochs** | 10 | 10 |

## 📊 Results & Analysis

### 1. Training Comparison
The plot below compares the validation accuracy over 10 epochs for both versions.

![Comparison Plot](comparison_plot.png)

### 2. Comparison Table<img width="1000" height="500" alt="comparison_plot" src="https://github.com/user-attachments/assets/5af022d0-389e-49e0-8421-fa989f282381" />



| Metric | Version 1 (Adam) | Version 2 (SGD) |
| :--- | :--- | :--- |
| **Best Accuracy** | 86.80% | **89.12%** |
| **Final Loss** | 0.6564 | **0.3905** |

### 3. Analysis
* **Best Version:** **Version 2 (SGD with Momentum)** performed best.
* **Key Finding:** While Adam is often the standard for quick convergence, SGD with a learning rate of 0.001 and momentum of 0.9 achieved a much lower final loss (0.3905 vs 0.6564) and a higher peak accuracy.
* **Target Accuracy:** The project successfully reached the goal! The final accuracy of **89.12%** exceeds the required 85% by over 4%.

## 🖼️ Visualizations

### Confusion Matrix (Best Model)
![Confusion Matrix](confusion_matrix.png)
*Observation: The model shows high precision across the 5 categories, indicating that the pre-trained ResNet18 features are highly effective for food recognition.*
<img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/7a440806-5c44-45c2-983b-c0f6631888b6" />

### Sample Predictions
![Sample Predictions](sample_predictions.png)
*10 random samples showing the true vs. predicted labels from the best-performing model.*
<img width="1500" height="600" alt="sample_predictions" src="https://github.com/user-attachments/assets/a0748865-f2f7-4960-bca5-5de9f0a4622b" />
## ⚙️ Setup & Installation
1.  **Clone the Repository:**
    ```bash
    git clone [Your-GitHub-Link]
    ```
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Notebook:** Open the `.ipynb` file in Google Colab or Jupyter and run all cells.

---
**Seed used for reproducibility:** `20240108`
