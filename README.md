# 🏦 AI Loan Risk Assessment System

An end-to-end Machine Learning application that predicts whether a loan applicant is **Low Risk** or **High Risk** using a LightGBM classifier. The application includes custom financial feature engineering, explainable AI (SHAP), and an interactive Streamlit dashboard to support better lending decisions.

---

## 🚀 Features

- Predicts borrower risk using a trained LightGBM model
- Interactive web application built with Streamlit
- Custom financial feature engineering
- SHAP Explainability for model transparency
- Prediction confidence visualization
- Personalized lending recommendations
- Professional dashboard interface

---

## 📊 Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Feature encoding
- Numerical feature preparation

### Feature Engineering

Custom engineered features include:

- Age Bracket
- Credit Capacity Ratio
- Credit History Depth (Credit Age Density)
- Household Stability Score
- Unbacked Principal Exposure

### Model

- LightGBM Classifier

### Explainability

- SHAP (SHapley Additive Explanations)

---

## 🖥️ Application

The Streamlit dashboard allows users to:

- Enter borrower information
- Predict loan risk instantly
- View model confidence
- Understand model decisions using SHAP
- Review engineered features
- Receive lending recommendations

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LightGBM
- SHAP
- Pandas
- NumPy
- Plotly
- Matplotlib
- Joblib

---

## 📂 Project Structure

```text
Loan_Risk_Assessment/
│
├── app.py
├── lightgbm.pkl
├── loan.jpg
├── requirements.txt
├── README.md
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Move into the project directory

```bash
cd <repository-name>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📈 Example Workflow

1. Enter borrower details
2. Click **Predict Risk**
3. Review:
   - Risk Classification
   - Prediction Confidence
   - SHAP Explainability
   - Lending Recommendation

---

## 🎯 Future Improvements

- Multi-class risk prediction
- PDF report generation
- Model monitoring dashboard
- REST API using FastAPI
- Docker deployment
- Cloud deployment on AWS/Azure/GCP
- Continuous model retraining pipeline

---

## 📚 Learning Outcomes

This project demonstrates:

- End-to-end Machine Learning workflow
- Financial feature engineering
- Explainable AI (XAI)
- Model deployment with Streamlit
- Interactive dashboard development
- Credit risk analysis

---

## 👨‍💻 Author

**Abhi Deshmukh**

Machine Learning • Artificial Intelligence • Quantitative Finance • Data Science

---

## ⭐ If you found this project interesting, consider giving it a star!
