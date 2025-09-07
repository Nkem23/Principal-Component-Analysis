# Step 1: import necessary libraries and modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2: Load the dataset
cancer = load_breast_cancer()

# convert the data to a dataframe
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target

# print the description and view of your dataframe
print(cancer_df.shape)
print(cancer_df.info())
print(cancer_df.describe())

cancer_df.head()
cancer_df.tail()

# Step 3: Standardize the columns of the cancer dataframe

X = cancer_df.drop('target', axis=1)   
y = cancer_df['target']                

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 4: Apply PCA to reduce to the features to 2 components
pca = PCA(n_components=2)                 
X_pca = pca.fit_transform(X_scaled)       

# Create new DataFrame with PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])  
pca_df['target'] = y                                       
pca_df.head()   


# Step 5: Visualize PCA results

sns.scatterplot(
    x='PC1', y='PC2',              
    hue='target',                  
    data=pca_df,                      
)

plt.title("PCA - Breast Cancer Dataset")
plt.savefig('PCA - Breast Cancer Dataset.png')
plt.show()


# Step 6 we test if a machine learning model- Logistic Regression can predict cancer outcome using our new PCA features.

# We split the data into training to teach the model and testing to check performance (using PC1 and PC2 as features)

X_train, X_test, y_train, y_test = train_test_split(
    pca_df[['PC1', 'PC2']],    
    pca_df['target'],          
    test_size=0.2,             
    random_state=42            
)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate the model, we check accuracy to see how well it did.
print("Accuracy:", log_reg.score(X_test, y_test))
   