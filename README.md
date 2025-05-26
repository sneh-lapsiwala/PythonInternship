# Elevate Labs AI & ML Internship - Task 1: Data Cleaning & Preprocessing

## Overview
This repository contains the solution for "Task 1: Data Cleaning & Preprocessing" as part of the Elevate Labs AI & ML Internship. The objective of this task is to learn how to clean and prepare raw data for machine learning models effectively. This project demonstrates a robust data preprocessing workflow using Python, Pandas, NumPy, Matplotlib/Seaborn, and Scikit-learn, specifically applied to the well-known Titanic dataset.

## Objective:
To learn how to clean and prepare raw data for machine learning (ML) models using robust and reproducible techniques.

## Tools Used:
* **Python:** The primary programming language.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib/Seaborn:** For data visualization (especially for outlier detection).
* **Scikit-learn (sklearn):** For building preprocessing pipelines, imputation, scaling, encoding, and a simple classification model.

## Hints/Mini Guide Implementation:
This project directly addresses each point from the provided "Hints/Mini Guide" in the task document:

1.  **Import the dataset and explore basic info (nulls, data types):**
    * The Titanic dataset is loaded directly from a public URL using Pandas.
    * Initial data exploration is performed using `df.info()` to inspect data types and non-null counts, and `df.isnull().sum()` to explicitly identify and count missing values in each column. `df.head()` provides a quick preview of the data.

2.  **Handle missing values using mean/median/imputation:**
    * Missing numerical values (e.g., in 'Age') are imputed using the **median strategy** via `sklearn.impute.SimpleImputer`.
    * Missing categorical values (e.g., in 'Embarked') are imputed with the **most frequent value (mode)** using `sklearn.impute.SimpleImputer`.
    * The 'Cabin' column, which has a very high percentage of missing values (approx. 77%), is effectively **dropped** by using `remainder='drop'` in the `ColumnTransformer`, as it's often more practical than complex imputation for such a high volume of missing data.

3.  **Convert categorical features into numerical using encoding:**
    * Categorical features ('Sex', 'Embarked') are converted into numerical format using **One-Hot Encoding** (`sklearn.preprocessing.OneHotEncoder`) within the preprocessing pipeline. This creates new binary columns for each category, suitable for machine learning algorithms.

4.  **Normalize/standardize the numerical features:**
    * Numerical features ('Age', 'Fare', 'SibSp', 'Parch', 'Pclass') are **standardized** using `sklearn.preprocessing.StandardScaler`. This transforms the data to have a mean of 0 and a standard deviation of 1, ensuring that features with larger scales do not disproportionately influence the model.

5.  **Visualize outliers using boxplots and remove them:**
    * Outliers in numerical features are visualized using boxplots both **before and after** the preprocessing steps to demonstrate their presence and the effect of handling them.
    * To "remove" (mitigate the impact of) outliers within the preprocessing pipeline, a **custom `OutlierCapper` transformer** is implemented. This transformer caps extreme values at the 1st and 99th percentiles, effectively limiting their range without removing entire rows, which is crucial for maintaining dataset integrity in a pipeline.

## Dataset:
The project uses the publicly available Titanic Dataset, which is loaded directly from:
`https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`

## What You'll Learn (and demonstrated in this project):
* Foundational data cleaning techniques.
* Strategies for handling null values (imputation and dropping).
* Methods for encoding categorical features into numerical representations.
* Techniques for feature scaling (standardization).
* Approaches to visualizing and mitigating the impact of outliers.
* The practical application of `sklearn` Pipelines and `ColumnTransformer` for building a robust, reproducible, and data-leakage-free preprocessing workflow.
* Integration of preprocessing with a basic machine learning model (Logistic Regression) to demonstrate an end-to-end process.

## Repository Contents:
* `data_cleaning_preprocessing_titanic.ipynb`: The main Google Colab notebook containing all the Python code, detailed explanations, and visualizations.
* `README.md`: This file, providing an overview and guide to the project.

## How to Run the Notebook:
1.  **Open in Google Colab:** Click on the `.ipynb` file in this repository. GitHub will provide an option to "Open in Colab". Alternatively, go to [Google Colab](https://colab.research.google.com/), select `File > Open notebook > GitHub`, and paste the URL of this repository.
2.  **Run All Cells:** Once the notebook is open in Colab, go to `Runtime > Run all` to execute all the code cells sequentially. The output, including data info, missing value counts, visualizations, and model evaluation, will be displayed directly in the notebook.

## My Google Colab Notebook Link:
You can access and run the notebook directly via this link:
[https://colab.research.google.com/drive/1aBJnL6buBeqejLTG3j3wc3W20-8ejERw?usp=sharing](https://colab.research.google.com/drive/1aBJnL6buBeqejLTG3j3wc3W20-8ejERw?usp=sharing)
