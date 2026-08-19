# Customer Churn Prediction

## Overview

This project is an end-to-end Customer Churn Prediction system built using Python, Scikit-learn, and Streamlit. The project uses a Logistic Regression pipeline trained on the Telco Customer Churn dataset and provides an interactive web application for predicting whether a customer is likely to churn.

## Project Objectives

The main objectives of this project are:

* Build a machine learning model for customer churn prediction.
* Create a reusable Scikit-learn preprocessing and prediction pipeline.
* Deploy the trained model through a Streamlit web application.
* Provide an interactive interface for entering customer information and viewing churn probability.

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information related to:

* Demographics
* Account information
* Contract details
* Internet and phone services
* Monthly and total charges
* Customer churn status

## Machine Learning Workflow

The project follows these main steps:

1. Data loading and inspection
2. Data cleaning
3. Exploratory Data Analysis
4. Categorical feature encoding
5. Train-test splitting
6. Feature scaling
7. Logistic Regression training
8. Decision Tree training and comparison
9. Feature engineering
10. Scikit-learn Pipeline implementation
11. Model evaluation
12. Model serialization using Joblib
13. Streamlit application development

## Model

The final deployed model is a Logistic Regression model integrated into a Scikit-learn Pipeline.

The pipeline includes:

* StandardScaler for numerical features
* OneHotEncoder for categorical features
* Logistic Regression for classification

The final pipeline achieved approximately 80.53% accuracy on the test set.

## Feature Engineering

Two additional features were tested:

* AverageMonthlySpend
* TotalServices

The feature-engineered pipeline achieved slightly lower performance than the basic pipeline, so the basic pipeline was selected as the final model.

## Streamlit Application

The Streamlit application allows users to enter customer information and receive:

* Churn prediction
* Churn probability
* Stay probability
* Customer profile summary

The application loads the saved machine learning pipeline directly using Joblib.

## Project Structure

```text
Churn-Project/
│
├── Churn.ipynb
├── app.py
├── customer_churn_pipeline.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd Churn-Project
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application with:

```bash
python -m streamlit run app.py
```

The application will open in your browser at the local Streamlit address.

## Live Demo

Live Application:

<Add your Streamlit Cloud URL here>

## Model Performance

| Model                              | Accuracy |
| ---------------------------------- | -------: |
| Manual Logistic Regression         |   80.38% |
| Basic Logistic Regression Pipeline |   80.53% |
| Feature-Engineered Pipeline        |   80.31% |

The basic Logistic Regression pipeline was selected as the final deployed model because it provided the highest test accuracy.

## Future Improvements

Possible improvements include:

* Testing additional machine learning algorithms
* Hyperparameter tuning
* Improved model interpretability
* More advanced feature engineering
* Adding customer retention recommendations
* Deploying the application with additional monitoring

## Author

Maooz Shafaqat

Data Science Intern at NeuroFive Solutions
