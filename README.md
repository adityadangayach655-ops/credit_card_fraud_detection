Credit Card Fraud Detection
This project demonstrates a classic machine learning approach to identifying fraudulent credit card transactions. The primary focus is on handling a highly imbalanced dataset, a common challenge in fraud detection, and using appropriate evaluation metrics to assess model performance accurately.

Project Overview
The goal of this project is to train a model that can distinguish between legitimate and fraudulent credit card transactions. Given the anonymized transaction data, we build a classification model that is optimized to catch rare fraudulent events while minimizing false alarms.

Key Learnings & Concepts:
Handling Imbalanced Data: The dataset contains less than 0.2% fraudulent transactions. This project explores why standard accuracy is a misleading metric and uses techniques like class_weight balancing to address the imbalance.

Data Preprocessing: Scaling features like Time and Amount to ensure they don't disproportionately influence the model.

Model Evaluation: Moving beyond accuracy to use more insightful metrics like the Confusion Matrix, Precision, Recall, and F1-Score to understand the model's real-world performance.

Tech Stack
Python

Pandas: For data loading and manipulation.

Scikit-learn: For data preprocessing, model training (Logistic Regression), and evaluation.

Matplotlib & Seaborn: For data visualization.

Jupyter Notebook: For interactive development and analysis.

Methodology
Exploratory Data Analysis (EDA): The dataset was loaded and analyzed to identify the severe class imbalance.

Preprocessing: The Time and Amount columns were scaled using StandardScaler to normalize their ranges.

Data Splitting: The data was split into training and testing sets, using stratification to maintain the same percentage of fraudulent transactions in both splits.

Model Training: A Logistic Regression model was trained with the class_weight='balanced' parameter. This crucial step penalizes mistakes on the minority (fraud) class more heavily, forcing the model to pay more attention to it.

Evaluation: The model's performance was evaluated on the unseen test set using a confusion matrix and a classification report to analyze its precision and recall for the fraud class.

How to Run
Clone the repository:

git clone https://github.com/YourUsername/Your-Repo-Name.git

Create and activate a conda environment:

conda create --name fraud_project python=3.10 -y
conda activate fraud_project

Install the required dependencies:

pip install -r requirements.txt

Launch the Jupyter Notebook:

jupyter notebook

Open the project notebook file (.ipynb) and run the cells.