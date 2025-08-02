Credit Card Fraud Detection
This project demonstrates a classic machine learning approach to identifying fraudulent credit card transactions. The primary focus is on handling a highly imbalanced dataset, a common challenge in fraud detection, and using appropriate evaluation metrics to assess model performance accurately.

Dataset
This project uses the "Credit Card Fraud Detection" dataset from Kaggle. It contains anonymized transactions made by European cardholders over a two-day period in September 2013.

Download Link: Kaggle: Credit Card Fraud Detection

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

Results
The model's performance was evaluated based on its ability to correctly identify the rare fraud class.

Recall (Fraud Class): The model achieved a high recall (approx. 0.92 or 92%), successfully identifying the vast majority of actual fraudulent transactions in the test set.

Precision (Fraud Class): The precision was lower (approx. 0.06 or 6%), indicating that the model produced a number of false positives.

Interpretation: The results show a successful trade-off for a fraud detection system. By optimizing for high recall, we ensure that most fraudulent activities are caught, which is often the primary business goal. The lower precision is an acceptable consequence, as it is generally preferable to investigate a few legitimate transactions (false positives) than to miss a costly fraudulent one (false negative).

Future Scope & Improvements
While the current model serves as a strong baseline, there are several ways it could be improved:

Advanced Sampling Techniques: Instead of just using class_weight, techniques like SMOTE (Synthetic Minority Over-sampling Technique) could be used to generate synthetic fraud examples, potentially improving the model's ability to learn fraud patterns.

More Complex Models: Experimenting with more complex models like Random Forest or XGBoost could capture more intricate patterns in the data and potentially lead to better performance.

Feature Engineering: Although the features are anonymized, further analysis could be done to see if new features can be created from the existing ones to improve model accuracy.

Threshold Tuning: The classification threshold (defaulting to 0.5) could be tuned to find a better balance between precision and recall, depending on the specific business requirements for minimizing risk vs. customer friction.

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
