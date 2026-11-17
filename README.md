# 💼 Employee Salary Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-red?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

The Employee Salary Prediction System is a Python-based Machine Learning project that predicts annual employee salaries based on age, work experience, education level, and job role.

It uses Random Forest Regression to learn patterns from sample employee data and estimate salaries for new employees.

## 🎯 Objectives

- Analyze employee salary data.
- Understand salary patterns across job roles.
- Process categorical features using One-Hot Encoding.
- Train a Random Forest Regression model.
- Evaluate prediction performance.
- Predict salaries for new employees.

## ⚙️ Features

- Employee dataset validation.
- Salary statistical analysis.
- Job-role-wise salary analysis.
- Categorical feature encoding.
- Train-test splitting.
- Random Forest Regression.
- MAE, MSE, RMSE and R² evaluation.
- New employee salary prediction.
- Actual vs predicted salary visualization.
- CSV prediction report.

## 🧠 Workflow

1. Load and validate employee data.
2. Analyze salary statistics.
3. Separate input features and target salary.
4. Split the dataset into training and testing sets.
5. Encode categorical features.
6. Train the Random Forest model.
7. Evaluate model performance.
8. Predict a new employee's salary.
9. Export prediction results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Visualization |
| Random Forest | Regression algorithm |

## 📊 Dataset Information

The sample dataset contains 30 synthetic employee records.

| Feature | Description |
|---|---|
| employee_id | Unique employee identifier |
| age | Employee age |
| experience_years | Work experience |
| education | Bachelor or Master |
| job_role | Employee job position |
| salary | Annual salary in INR |

**Target variable:** `salary`

## 📈 Model Evaluation

The model uses:

- **MAE:** Average absolute prediction error.
- **MSE:** Average squared prediction error.
- **RMSE:** Root mean squared prediction error.
- **R² Score:** Explains the variation in salary predictions.

Lower error metrics and an R² score closer to 1 generally indicate better performance.

## 🖥️ Example Prediction

```python
age = 27
experience = 4
education = "Bachelor"
job_role = "Developer"
```

The trained model estimates the employee's annual salary in Indian Rupees.

## 📁 Generated Files

| File | Description |
|---|---|
| salary_comparison.png | Actual vs predicted salary chart |
| salary_predictions.csv | Prediction results and errors |

## 🚀 Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/ayushman-ece/employee-salary-predictor.git
```

**2. Open the project folder**

```bash
cd employee-salary-predictor
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate on Windows**

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the project**

```bash
python main.py
```

## 📚 Key Concepts Learned

- Machine Learning Regression
- Random Forest Regression
- One-Hot Encoding
- Feature Engineering
- Scikit-learn Pipelines
- Train-test splitting
- MAE, MSE and RMSE
- R² Score
- Data validation
- Model prediction

## ⚠️ Limitations

- Uses only 30 synthetic employee records.
- Does not represent actual salary market data.
- Salary depends on many additional factors, including company, location, skills, and industry.
- Model results should not be used for real hiring or compensation decisions.

## 🔮 Future Improvements

- Use a larger real-world salary dataset.
- Add skills, company size, and location features.
- Compare Linear Regression and Gradient Boosting.
- Perform hyperparameter tuning.
- Build a Streamlit dashboard.
- Deploy the model as an API.

## 👨‍💻 Author

**Ayushman Tiwari**

GitHub: [@ayushman-ece](https://github.com/ayushman-ece)

---

⭐ If you find this project useful, consider giving the repository a star!