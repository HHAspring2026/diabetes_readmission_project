# Diabetes Readmission Prediction Project

## Overview
We have been assigned a new project from the Director of Clinical Informatics to use the diabetes dataset provided to determine the leading variables of readmission so the hospital can take corrective action to reduce the 30-daay readmission rate. Our team will clean the dataset and visualize the work done in building the diabetes readmission predictive model. We will also design the infrastructure components necessary to build, test and deploy an algorithm to predict the likelihood of 30-day readmissions in the diabetic population.

### Three Components
1. Python Analytics (HHA550)
2. Data Visualization (HHA 552)
3. Infrastructure Design (HHA 551)

## Dataset
https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008 

## Repo Structure
diabetes_readmission_project/
│
├── data/
│   ├── diabetic_data.csv
│   └── cleaned_diabetic_data.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── analysis.py
│   └── run_all.py
│
├── images/
│   ├── anova_results.csv
│   ├── cluster_summary.csv
│   ├── clusters_scatter.png
│   ├── correlation_heatmap.png
│   ├── top_linear_predictors.png
│   ├── top_logistics_predictors.png
│
├── README.md
├── requirements.txt

## Team Members
- Anita Liu
- Angel Huang 
- Huma Babar
- Aarav Desai
- Tanveer Kaur 

# Part 1: Python Analytics (HHA 550)

## Objective 
Identify the fewest variables required to predict 30-day readmission using statistical and machine learning methods.

## ETL Process
- Loaded dataset into Python
- Replaced missing values ("?" into NaN)
- Removed invalid records (e.g., unknown gender)
- Dropped variables with >50% missing data
- Created binary target variable:
    -  `<30` into 1 (readmitted within 30 days)
    -  `>30` / `NO` into 0
- Saved cleaned dataset for analysis

## Methods Used
1. Correlation Analysis
2. Linear Regression
3. Logistic Regression
4. ANOVA 
5. Clustering

## Final Variables
- number_inpatient
- num_medications
- time_in_hospital
- number_diagnoses

## Key Findings
- Patients with more prior inpatient visits are more likely to be readmitted
- Higher medication counts indicate greater treatment complexity
- Longer hospital stays reflect increased severity
- More diagnoses indicate higher comorbidity burden

## Outputs
- Correlation heatmap
- Logistic regression feature importance
- Linear regression coefficients
- ANOVA results
- Clustering visualization

# Part 2: Data Visualization (HHA 552) 

# Part 3: Infrastructure Design (HHA 551)