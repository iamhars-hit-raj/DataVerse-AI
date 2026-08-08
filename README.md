# 📊 DataVerse AI (-Harshit Raj)

> An End-to-End AI-Powered Data Science Platform and a universe of data.

🌐 Live Application: https://dataverse-ai.streamlit.app/

📦 Repository: https://github.com/iamhars-hit-raj/DataVerse-AI

🚀 Overview

DataVerse AI is a full-stack data analytics workspace designed to take a user from raw dataset to actionable insight without requiring multiple disconnected tools.

The platform combines traditional data-science workflows with AI-assisted analysis:

Upload → Profile → Clean → Explore → Engineer Features → Train ML Models → Ask AI → Generate Insights → Report

It is built for students, analysts, developers, and anyone who wants a practical workflow for working with structured datasets through a single interactive interface.

✨ Key Features

📥 Dataset Upload

Upload structured datasets directly into the application.

Maintain the active dataset through Streamlit session state.

Work with the same dataset across the analytics workflow.

🔎 Dataset Overview

Dataset dimensions and structure.

Column-level information.

Data types and completeness.

Descriptive statistics.

Unique-value analysis.

Early identification of potential data-quality issues.

🧹 Data Cleaning

Missing-value inspection and handling.

Duplicate-data analysis.

Data-type handling.

Cleaning workflow designed for downstream analytics and ML.

📈 Exploratory Data Analysis

Distribution analysis.

Numerical and categorical exploration.

Interactive visualizations.

Relationship and trend analysis.

Plotly-powered charts.

📊 Interactive Dashboard

KPI cards.

Dataset health indicators.

Interactive charts.

High-level dataset summaries.

Visual exploration of important variables.

🛠️ Feature Engineering

Feature transformation workflow.

Preparation of data for machine-learning models.

Handling of categorical and numerical features.

Dataset-aware preprocessing.

🤖 Machine Learning Studio

Supports supervised machine-learning workflows including:

Classification

Regression

Decision Tree

Logistic Regression

Model training and evaluation

Hyperparameter configuration

Automatic handling of categorical/numerical preprocessing

The ML workflow is designed to guard against common real-world dataset problems such as high-cardinality identifier columns, accidental text targets, categorical encoding issues, and inappropriate classification targets.

🧠 AI Analyst

Ask questions about the uploaded dataset using natural language.

Examples:

Which columns have unusual values?

What are the major patterns in this dataset?

Which variables appear important for prediction?

What data-quality issues should I investigate?

The AI layer uses Google's current Google GenAI SDK rather than the deprecated google-generativeai package.

📌 AI Dashboard / AI Copilot

AI-assisted functionality for turning dataset information into useful analytical observations and recommendations.

📑 Reports

Generate analytical reports from the application's workflow and findings.

🧩 Application Workflow

Dataset
   │
   ▼
Upload → Overview → Cleaning → EDA
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            Feature Engineering    Dashboard
                    │
                    ▼
             Machine Learning
                    │
             ┌──────┴──────┐
             ▼             ▼
        Evaluation     AI Analyst
             │             │
             └──────┬──────┘
                    ▼
          AI Dashboard / Copilot
                    │
                    ▼
                 Reports

🏗️ Architecture

DataVerse AI uses a modular frontend/backend structure:

DataVerse-AI/
│
├── app.py
│
├── backend/
│   ├── ai_analyst.py
│   ├── ai_insights.py
│   ├── chart_ai.py
│   ├── ml.py
│   └── ...
│
├── frontend/
│   ├── dashboard.py
│   ├── dataset_overview.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── machine_learning.py
│   ├── ai_analyst.py
│   ├── ai_dashboard.py
│   ├── ai_copilot.py
│   ├── reports.py
│   ├── theme.py
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore

The exact module list may evolve as development continues.

Design principle

Frontend: Streamlit pages and user interaction.

Backend: ML, AI, preprocessing, and analytical utilities.

Session state: Shared dataset and application state across pages.

🛠️ Tech Stack

Area

Technology

Language

Python

Web App

Streamlit

Data Processing

Pandas, NumPy

Visualization

Plotly, Matplotlib

Machine Learning

Scikit-learn

Scientific Computing

SciPy

Explainability

SHAP

Generative AI

Google Gemini / Google GenAI SDK

Reporting

ReportLab

Deployment

Streamlit Community Cloud

Version Control

Git + GitHub

⚙️ Local Installation

1. Clone

git clone https://github.com/iamhars-hit-raj/DataVerse-AI.git
cd DataVerse-AI

2. Create and activate a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Configure Gemini

DataVerse AI uses Google's current GenAI Python SDK.

If needed:

pip install google-genai

Set your API key:

Windows PowerShell:

$env:GEMINI_API_KEY="YOUR_API_KEY"

macOS/Linux:

export GEMINI_API_KEY="YOUR_API_KEY"

For Streamlit Community Cloud, add the key through the app's Secrets configuration:

GEMINI_API_KEY = "YOUR_API_KEY"

Never commit API keys, .env files, credentials, or private datasets to GitHub.

5. Run

streamlit run app.py

Then open:

http://localhost:8501

🤖 Machine Learning Design

DataVerse AI is designed for real-world datasets rather than assuming every column is immediately suitable for modeling.

Predictive features

Examples:

Age

Numerical measurements

Valid categorical attributes

Engineered features

High-cardinality / identifier-like columns

Examples:

Name

Doctor

Hospital

Record IDs

Unique identifiers

These columns can create extremely large feature spaces after encoding and make models unnecessarily slow.

Target validation

A classification target should have a reasonable number of classes relative to the number of observations.

For example, a Name column should not accidentally become a classification target containing thousands of unique values.

This helps avoid:

Extremely slow Logistic Regression training.

Memory-heavy one-hot encoded matrices.

Invalid multiclass solver configurations.

Misleading classification warnings.

🔐 Security

Sensitive configuration should be supplied through environment variables or Streamlit Secrets.

Do not commit:

.env
API keys
tokens
credentials
private datasets

☁️ Deployment

DataVerse AI is deployed using Streamlit Community Cloud.

Live application: https://dataverse-ai.streamlit.app/

Deployment flow:

Local Development
       │
       ▼
      Git
       │
       ▼
    GitHub
       │
       ▼
Streamlit Community Cloud
       │
       ▼
   Live Application

Deployment configuration uses:

GitHub repository

main branch

app.py

Python 3.11

requirements.txt

Streamlit Secrets for API configuration

📌 Current Status

✅ Completed

Dataset Upload

Session State Management

Dataset Overview

Data Cleaning

Exploratory Data Analysis

Dashboard

KPI Cards

Dataset Health Score

Dashboard Charts

Feature Engineering

Machine Learning Studio

AI Analyst

AI Dashboard

AI Copilot

Reports

🚧 In Progress

Advanced Machine Learning

Advanced Model Evaluation

AutoML

Explainable AI / SHAP enhancements

🔜 Coming Soon

AI Dataset Assistant

Automated Data Storytelling

Model Deployment / Prediction API

Advanced Feature Selection

🎯 Why DataVerse AI?

Traditional data workflows often require switching between multiple tools:

CSV → Pandas → Jupyter → Visualization → Scikit-learn → AI Tool → Reporting Tool

DataVerse AI brings these capabilities into one workflow:

                    ┌─────────────────────────┐
                    │       DataVerse AI      │
                    │                         │
Dataset ───────────►│ Explore → Clean → Model │
                    │     ↓           ↓       │
                    │ Visualize       AI       │
                    │        ↓        ↓       │
                    │       Insights          │
                    │           ↓             │
                    │         Report           │
                    └─────────────────────────┘

The goal is to make the complete analytical workflow faster, more accessible, and easier to demonstrate.

🧪 Example Use Cases

Student data-analysis projects

Healthcare dataset exploration

Customer analytics

Sales and business datasets

Financial datasets

Survey analysis

Classification problems

Regression problems

Exploratory data analysis

AI-assisted data interpretation

Rapid ML experimentation

📈 Future Direction

The long-term goal is to evolve DataVerse AI into a more complete AI-assisted data-science workspace, with:

Automated model selection

Automated feature engineering

Advanced explainability

Natural-language dataset querying

AI-generated data stories

Automated anomaly detection

Model comparison

Prediction APIs

Advanced reporting

Reusable analytical pipelines

👨‍💻 Author

Harsh Raj

GitHub: https://github.com/iamhars-hit-raj

⭐ Support the Project

If DataVerse AI is useful or interesting:

⭐ Star the repository

🐛 Report issues

💡 Suggest features

🔀 Submit pull requests

📢 Share the project

📄 Project Status

DataVerse AI — v1.0

Built with Python, Streamlit, Scikit-learn, Plotly, Pandas, and Google Gemini.

Live Demo: https://dataverse-ai.streamlit.app/


Screenshots:

<img width="1917" height="923" alt="image" src="https://github.com/user-attachments/assets/1a9c9da9-a2d7-4d54-b332-9878029744f8" />
<img width="1917" height="892" alt="image" src="https://github.com/user-attachments/assets/0c9c25b4-d6d0-4a91-812f-66932ac39c96" />
<img width="1917" height="886" alt="image" src="https://github.com/user-attachments/assets/8479b042-85b0-41d8-be1a-889a66115519" />
<img width="1917" height="877" alt="image" src="https://github.com/user-attachments/assets/e4a78da0-5ea9-489e-bdbf-d9e50680335b" />
<img width="1912" height="865" alt="image" src="https://github.com/user-attachments/assets/5ad15c90-bca4-4a5d-b7ee-50639e102811" />
<img width="1917" height="883" alt="image" src="https://github.com/user-attachments/assets/46ca74ce-d580-4ba5-8487-c04c97d3ca29" />
<img width="1916" height="867" alt="image" src="https://github.com/user-attachments/assets/f893f908-af07-44ec-b78e-a17eed465c90" />
<img width="1917" height="881" alt="image" src="https://github.com/user-attachments/assets/72c50588-1182-4e27-a82c-3c936effdee2" />





