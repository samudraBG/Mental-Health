# 🧠 Mansik Santulan — Student Mental Health Score Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application designed to predict a student's **Mental Health Score (1–10)** based on social media usage habits, academic workload, and daily lifestyle patterns. Built with an optimized **Random Forest** regression model, served through a **FastAPI** microservice, and interfaced with an interactive modern web dashboard featuring real-time wellness gauge visualization.

---

## 📌 Table of Contents

- [Key Features](#-key-features)
- [Dataset Overview](#-dataset-overview)
- [Machine Learning Workflow](#-machine-learning-workflow)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
  - [Model Performance Comparison](#model-performance-comparison)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started Locally](#-getting-started-locally)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Backend API](#running-the-backend-api)
  - [Launching the Frontend](#launching-the-frontend)
- [API Reference](#-api-reference)
- [License](#-license)

---

## ✨ Key Features

- **End-to-End Pipeline:** Seamless integration from exploratory data analysis and scikit-learn preprocessing pipelines to a production REST API and interactive web interface.
- **Accurate Predictions:** Trained and evaluated multiple regression algorithms; Random Forest achieved an **$R^2$ score of 87.8%** with a Mean Absolute Error (**MAE**) of **0.35**.
- **Robust Preprocessing (`ColumnTransformer`):** Automatic log transformation for skewed numeric features, ordinal encoding for ranked metrics, standard scaling, and one-hot encoding with unknown-category handling.
- **Production-Ready FastAPI Backend:** Includes strict data validation using **Pydantic** schema definitions, CORS middleware support, and instant inference.
- **Interactive Wellness Dashboard:** Modern, responsive UI equipped with an animated SVG arc gauge, wellness band classification (Flourishing, Moderate, At Risk), and actionable suggestions.

---

## 📊 Dataset Overview

The dataset contains **5,000 student behavioral records** across 13 diverse attributes:

| Feature Name | Type | Description / Range |
| :--- | :--- | :--- |
| `Age` | Numerical | Age of the student (18 – 24) |
| `Gender` | Categorical | `Male`, `Female` |
| `Country` | Categorical | 111 unique countries (engineered into Top 10 + `Other`) |
| `Academic_Level` | Categorical | `High School`, `Undergraduate`, `Graduate` |
| `Most_Used_Platform` | Categorical | Instagram, YouTube, TikTok, WhatsApp, LinkedIn, etc. |
| `Purpose_Of_Use` | Categorical | `Entertainment`, `Education`, `Networking`, `News` |
| `Avg_Daily_Usage_Hours` | Numerical | Screen time on social media (hours/day) |
| `Daily_Unlocks` | Numerical | Total phone unlocks per day |
| `Study_Hours` | Numerical | Academic study hours per day |
| `Physical_Activity_Hours` | Numerical | Exercise and sports hours per day |
| `Sleep_Hours_Per_Night` | Numerical | Average sleep duration (hours/night) |
| `Stress_Level` | Ordinal | `Low`, `Medium`, `High`, `Very High` |
| **`Mental_Health_Score`** | **Target** | Continuous mental health assessment score (**1.0 to 10.0**) |

---

## 🔬 Machine Learning Workflow

### Exploratory Data Analysis (EDA)
- Analyzed target score distribution showing a balanced bell-curve centered at ~6.2.
- Identified strong negative correlation between high screen time / high stress levels and mental wellness scores.
- Discovered strong positive correlation between sleep duration, physical activity, and mental health resilience.
- Evaluated feature skewness: `Study_Hours` exhibited right-skewness ($\approx 0.44$), mitigated using `np.log1p`.

### Preprocessing & Feature Engineering
A composite `ColumnTransformer` pipeline ensures zero data leakage and consistent transformation during training and real-time inference:

1. **High Cardinality Reduction:** Grouped 111 raw countries into the top 10 most represented nations (`India`, `USA`, `Canada`, `Australia`, `UK`, `Germany`, `Mexico`, `Turkey`, `France`, and `Other`).
2. **Skewed Features Pipeline:** `FunctionTransformer(np.log1p)` $\rightarrow$ `StandardScaler()` on `Study_Hours`.
3. **Continuous Features Pipeline:** `StandardScaler()` applied across `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, and `Sleep_Hours_Per_Night`.
4. **Ordinal Pipeline:** `OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']])` preserving relative rank.
5. **Nominal Pipeline:** `OneHotEncoder(handle_unknown='ignore')` applied to `Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, and `Grouped_country`.

### Model Performance Comparison

Multiple regression models were trained using an 80/20 train-test split:

| Model | Test $R^2$ Score | Training $R^2$ | Test MAE | Test RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** (Baseline) | 0.7398 | 0.7237 | 0.5362 | 0.6760 |
| **Random Forest Regressor (Default)** ⭐ | **0.8776** | **0.9808** | **0.3472** | **0.4637** |
| **Random Forest Regressor (Tuned via RandomizedSearchCV)** | 0.8650 | 0.9547 | 0.3689 | 0.4869 |

> The default **Random Forest Regressor** demonstrated the highest testing predictive accuracy ($R^2 \approx 87.8\%$) and lowest MAE ($0.35$), and was serialized as `Mental_Health_Model.pkl` for inference.

---

## 🛠️ Tech Stack

- **Machine Learning & Modeling:** Python, Scikit-learn, NumPy, Pandas, Joblib
- **Data Visualization & Analysis:** Matplotlib, Seaborn, Jupyter Notebook
- **Backend API:** FastAPI, Pydantic, Uvicorn
- **Frontend Interface:** HTML5, CSS3, Modern JavaScript (Fetch API, SVG Gauges)
- **Version Control:** Git

---

## 📁 Project Structure

```bash
├── Student Social Media And Mental Health Impact.csv # Dataset (5,000 rows)
├── ML_Project.ipynb                                  # End-to-end Jupyter Notebook (EDA to export)
├── Mental_Health_Model.pkl                           # Serialized preprocessor + trained model
├── main.py                                           # FastAPI backend microservice
├── requirements.txt                                  # Python dependencies
├── index.html                                        # Web dashboard UI
├── style.css                                         # UI styling & design system
├── script.js                                         # Client-side validation & API integration
└── README.md                                         # Project documentation
```

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.9+ installed on your system
- Git installed

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/samudraBG/Mental-Health-Score.git
   cd Mental-Health-Score
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Backend API

Start the FastAPI local server:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
- API will be live at: `http://127.0.0.1:8000`
- Interactive Swagger documentation: `http://127.0.0.1:8000/docs`

### Launching the Frontend

Serve `index.html` via Live Server or Python's built-in HTTP server:
```bash
python -m http.server 3000
```
Open your browser and navigate to `http://127.0.0.1:3000`.

---

## 📡 API Reference

### Health Check
- **Endpoint:** `GET /`
- **Response:**
  ```json
  {
    "message": "Welcome to Sheryians AI School Guys"
  }
  ```

### Predict Mental Health Score
- **Endpoint:** `POST /predict`
- **Headers:** `Content-Type: application/json`
- **Request Body Example:**
  ```json
  {
    "age": 21,
    "gender": "Male",
    "country": "India",
    "academic_level": "Undergraduate",
    "most_used_platform": "Instagram",
    "purpose_of_use": "Entertainment",
    "avg_daily_usage_hours": 4.5,
    "daily_unlocks": 140,
    "study_hours": 3.5,
    "physical_activity_hours": 1.5,
    "sleep_hours_per_night": 7.0,
    "stress_level": "Medium"
  }
  ```
- **Response Body Example:**
  ```json
  {
    "predicted_mental_health_score": 6.84
  }
  ```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/samudraBG/Mental-Health-Score/issues).

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
