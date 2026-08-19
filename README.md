# Customer Churn Prediction

An end-to-end machine learning application that predicts whether a customer is likely to churn. The project uses Logistic Regression with a Scikit-learn preprocessing pipeline and provides an interactive Streamlit web application for generating churn predictions and probability estimates.

## Live Demo

[Customer Churn Prediction Application](https://customer-churn-prediction-057.streamlit.app/)

## Project Overview

Customer churn is an important business problem because identifying customers who are likely to leave can help organizations take preventive retention measures.

This project develops a machine learning solution that analyzes customer information and predicts whether a customer is likely to churn.

The trained model is integrated into a Streamlit web application, allowing users to enter customer information and receive predictions through an interactive interface.

## Project Objectives

The main objectives of this project are:

* Build a machine learning model for customer churn prediction.
* Perform data cleaning and exploratory data analysis.
* Preprocess numerical and categorical features.
* Compare different machine learning approaches.
* Build a reusable Scikit-learn preprocessing and prediction pipeline.
* Save the trained model using Joblib.
* Develop an interactive Streamlit application.
* Deploy the application using Streamlit Community Cloud.

## Dataset

The project uses the Telco Customer Churn dataset, which contains information about customers and their subscription behavior.

The dataset includes information related to:

* Customer demographics
* Account information
* Contract details
* Internet services
* Phone services
* Payment methods
* Monthly charges
* Total charges
* Customer tenure
* Churn status

The target variable is `Churn`, which indicates whether a customer has left the service.

## Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

1. Data loading and inspection
2. Data cleaning
3. Exploratory Data Analysis
4. Handling missing and inconsistent values
5. Categorical feature encoding
6. Numerical feature preprocessing
7. Train-test splitting
8. Feature scaling
9. Logistic Regression model training
10. Decision Tree model comparison
11. Feature engineering
12. Scikit-learn Pipeline implementation
13. Model evaluation
14. Model serialization using Joblib
15. Streamlit application development
16. Deployment using Streamlit Community Cloud

## Machine Learning Model

The final deployed model is a Logistic Regression classifier integrated into a Scikit-learn Pipeline.

The pipeline includes:

* `StandardScaler` for numerical features
* `OneHotEncoder` for categorical features
* `LogisticRegression` for classification

Using a pipeline ensures that the same preprocessing steps used during model training are automatically applied to new customer data submitted through the Streamlit application.

## Feature Engineering

Additional features were explored to determine whether they could improve model performance.

The following features were tested:

* `AverageMonthlySpend`
* `TotalServices`

The feature-engineered pipeline achieved slightly lower performance than the basic pipeline. Therefore, the basic Logistic Regression pipeline was selected as the final deployed model.

## Model Performance

The models were evaluated using test-set accuracy.

| Model                              | Accuracy |
| ---------------------------------- | -------: |
| Manual Logistic Regression         |   80.38% |
| Basic Logistic Regression Pipeline |   80.53% |
| Feature-Engineered Pipeline        |   80.31% |

The Basic Logistic Regression Pipeline achieved the highest test accuracy of approximately 80.53% and was selected as the final model for deployment.

## Streamlit Application

The Streamlit application provides an interactive interface for customer churn prediction.

Users can enter relevant customer information and receive:

* Churn prediction
* Stay prediction
* Churn probability
* Stay probability
* Customer profile summary
* Interactive probability visualization

The application loads the trained machine learning pipeline using Joblib and generates predictions in real time.

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File               | Description                                     |
| ------------------ | ----------------------------------------------- |
| `app.py`           | Streamlit application                           |
| `churn_model.pkl`  | Serialized trained machine learning model       |
| `requirements.txt` | Python dependencies required by the application |
| `README.md`        | Project documentation                           |
| `.gitignore`       | Files excluded from Git version control         |

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn
* Jupyter Notebook
* Git
* GitHub

## Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd customer-churn-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application Locally

Start the Streamlit application using:

```bash
python -m streamlit run app.py
```

The application will open in your browser at the local Streamlit address.

## Deployment

The application is deployed using Streamlit Community Cloud.

### Live Application

https://customer-churn-prediction-057.streamlit.app/

The deployed application loads the serialized machine learning model and performs predictions directly through the web interface.

## Future Improvements

Possible future improvements include:

* Testing additional classification algorithms
* Hyperparameter optimization
* Improving model interpretability
* Exploring advanced feature engineering techniques
* Adding customer retention recommendations
* Providing personalized recommendations for high-risk customers
* Adding model performance monitoring
* Expanding the application with additional analytics and visualizations

## Author

**Maooz Shafaqat**

Data Science Intern
NeuroFive Solutions

## Project Workflow

The project demonstrates the complete process of transforming a machine learning model into a deployed web application:

```text
Dataset
   |
   v
Data Cleaning and EDA
   |
   v
Feature Preprocessing
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Joblib Model Serialization
   |
   v
Streamlit Application
   |
   v
GitHub Repository
   |
   v
Streamlit Community Cloud
   |
   v
Live Machine Learning Application
```

This project demonstrates the complete machine learning lifecycle, from data preprocessing and model development to application development and cloud deployment.
