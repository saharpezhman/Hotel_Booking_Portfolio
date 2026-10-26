Hotel Booking Analytics & Cancellation Prediction

Overview

This project analyzes hotel booking data to identify cancellation patterns, investigate booking behavior, and evaluate machine learning approaches for predicting reservation cancellations.

It includes data cleaning, exploratory data analysis, data quality checks, statistical analysis, clustering, model comparison, and an interactive Streamlit dashboard.

Business Objective

Hotel cancellations create uncertainty in room availability, revenue planning, and operational decisions.

The objective of this project is to explore the factors associated with cancellations and evaluate whether historical booking information can help estimate cancellation risk.

Project Workflow

Data Cleaning and Preparation
Inspected the original dataset.
Investigated duplicate records and missing values.
Analyzed data quality issues.
Prepared the dataset for further analysis.
Exploratory Data Analysis
Explored booking characteristics and cancellation patterns.
Investigated lead time and hotel type.
Created visualizations to identify meaningful patterns.
Data Quality Analysis
Investigated missing-data patterns.
Examined outliers and suspicious records.
Audited feature availability for prediction tasks.
Machine Learning
Compared cancellation prediction models.
Evaluated classification performance.
Investigated feature importance.
Explored decision-threshold optimization and early prediction.
Statistical Analysis and Clustering
Applied clustering to explore groups of bookings.
Investigated differences in cancellation behavior.
Examined cancellation patterns across booking characteristics.
Model Evaluation
Compared models and feature sets.
Evaluated metrics including ROC-AUC and F1-score.
Investigated temporal evaluation.
Interactive Dashboard
The Streamlit dashboard provides:

Total booking count
Canceled booking count
Overall cancellation rate
Booking cancellation visualization
A preview of the prepared dataset
Dataset

The project uses the Hotel Booking Demand dataset.

Original dataset: 119,390 records and 32 columns.
Prepared dataset: 87,396 records and 39 columns.
Cancellation rate in the prepared dataset: approximately 27.49%.
Technology Stack

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Streamlit
Run the Dashboard Locally

Requirements

Python must be installed on your computer.

Install the required libraries:

python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit

Start the Dashboard

Open a terminal in the project directory and run:

python -m streamlit run app.py

Then open the local URL displayed in the terminal, usually:

"http://localhost:8501"

Note: The prepared dataset and any required model files must be available at the paths expected by the application.

Key Learning Outcomes

Data cleaning and preparation
Exploratory data analysis
Data visualization
Feature analysis
Classification and clustering
Model comparison and evaluation
Threshold optimization
Temporal model evaluation
Interactive dashboard development
Limitations

Predictions are estimates based on historical data and are not guarantees of future booking behavior. Model performance should be evaluated carefully before operational use.

Future Improvements

Integrate the saved prediction model into the dashboard.
Add interactive cancellation-risk predictions for individual bookings.
Expand dashboard filters and analytical views.
Improve deployment and reproducibility.
