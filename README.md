# Mental Health Score Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project to predict a student's **Mental Health Score (1–10)** based on social media habits, screen time, sleep duration, and daily lifestyle factors. Features a trained **Random Forest** regression model served through a **FastAPI** backend and an interactive web dashboard with real-time gauge visualization.

---

## Overview & Highlights

- **Dataset:** 5,000 student records analyzing screen time, phone unlocks, study hours, sleep, physical activity, and stress levels.
- **Preprocessing Pipeline:** Modular `ColumnTransformer` handling log transformations for skewed features, standard scaling, ordinal encoding for stress levels, and one-hot encoding for categorical variables.
- **Model Performance:** Random Forest Regressor achieving **87.8% $R^2$ score** and a low **MAE of 0.35** (outperforming baseline Linear Regression at 73.9% $R^2$).
- **API & UI:** Lightweight **FastAPI** service with **Pydantic** input validation, paired with a responsive HTML/CSS/JS frontend.

---

## Tech Stack

- **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
- **Visualization & EDA:** Matplotlib, Seaborn, Jupyter Notebook
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** HTML5, CSS3, JavaScript

---

## Project Structure

```bash
├── Student Social Media And Mental Health Impact.csv # Dataset (5,000 rows)
├── ML_Project.ipynb                                  # EDA, preprocessing & model training
├── Mental_Health_Model.pkl                           # Serialized pipeline & trained model
├── main.py                                           # FastAPI backend
├── requirements.txt                                  # Dependencies
├── index.html                                        # Web dashboard
├── style.css                                         # UI styling
├── script.js                                         # Client-side API integration
└── README.md                                         # Project documentation
```

---

## How to Run Locally

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/samudraBG/Mental-Health-Score.git
cd Mental-Health-Score

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # On Windows
# source venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the FastAPI Server
```bash
uvicorn main:app --reload
```
- API running at: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`

### 3. Open the Web Dashboard
Open `index.html` directly in your browser or run a simple local server:
```bash
python -m http.server 3000
```
Then visit `http://127.0.0.1:3000`.

---

## API Usage

### `POST /predict`
Send student behavioral data to receive the predicted score:

**Sample Request:**
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

**Sample Response:**
```json
{
  "predicted_mental_health_score": 6.84
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
