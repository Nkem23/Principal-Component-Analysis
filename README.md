## PCA and Logistic Regression on Breast Cancer Dataset

###  Project Description
This project demonstrates **Principal Component Analysis (PCA)** using the Breast Cancer dataset from `sklearn.datasets`.  
The dataset is reduced to **two PCA components**, visualized, and then used to train a **Logistic Regression model** for cancer classification.

### Objectives
- Implement PCA to extract essential variables from the cancer dataset.
- Reduce cancer data into 2 principal components.
- Visualize the reduced dataset.
- Train and evaluate Logistic Regression on the PCA-transformed data.

### Dataset
The dataset used is the **Breast Cancer Wisconsin Diagnostic dataset**, available directly from `sklearn.datasets.load_breast_cancer`.  
It contains 569 rows with 30 columns describing cancer tumor characteristics.

- **Target values:**  
  - `0` = Malignant (cancer present)  
  - `1` = Benign (non-cancerous)

### Modules & Libraries used
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

### Steps Performed
1. **Load dataset** and convert to DataFrame.  
2. **Standardize features** using `StandardScaler`.  
3. **Apply PCA** to reduce features to 2 components.  
4. **Visualize PCA results** with a scatterplot.  
5. **Train Logistic Regression** using the 2 PCA components.  
6. **Evaluate accuracy** on test data.

### Results
- **Visualization:**  
  A scatterplot showing the separation of malignant vs. benign samples .  
  Saved as: `PCA - Breast Cancer Dataset.png`  

- **Logistic Regression Performance:**  
With 99.1% accuracy, the two PCA components clearly kept the important information.
