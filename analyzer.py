
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


NUMERIC_FEATURES = ["age", "experience_years"]
CATEGORICAL_FEATURES = ["education", "job_role"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_employee_data(file_path):
    df = pd.read_csv(file_path)

    required = ["employee_id"] + FEATURES + ["salary"]

    if not all(col in df.columns for col in required):
        raise ValueError("Required columns are missing.")

    if df.empty or df[required].isnull().any().any():
        raise ValueError("Dataset is empty or contains missing values.")

    if df["employee_id"].duplicated().any():
        raise ValueError("Duplicate employee IDs found.")

    for col in ["employee_id"] + NUMERIC_FEATURES + ["salary"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    numeric_values = df[["employee_id"] + NUMERIC_FEATURES + ["salary"]].to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError("Invalid numeric values found.")

    if (df[NUMERIC_FEATURES] < 0).any().any() or (df["salary"] <= 0).any():
        raise ValueError("Invalid negative or zero values found.")

    if not df["education"].isin(["Bachelor", "Master"]).all():
        raise ValueError("Invalid education category.")

    if not df["job_role"].isin(
        ["Developer", "Analyst", "Data Scientist", "Manager", "Tester"]
    ).all():
        raise ValueError("Invalid job role.")

    if (df["experience_years"] > df["age"] - 18).any():
        raise ValueError("Experience cannot exceed plausible working age.")

    print(f"Loaded {len(df)} employee records.")
    return df


def analyze_employees(df):
    print("\n========== EMPLOYEE ANALYSIS ==========")
    print("Total employees:", len(df))
    print(f"Average salary: ₹{df['salary'].mean():,.2f}")
    print(f"Minimum salary: ₹{df['salary'].min():,.2f}")
    print(f"Maximum salary: ₹{df['salary'].max():,.2f}")

    print("\nAverage salary by job role:")
    print(df.groupby("job_role")["salary"].mean().round(2))


def prepare_data(df):
    X = df[FEATURES]
    y = df["salary"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def train_salary_model(X_train, y_train):
    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"),
             CATEGORICAL_FEATURES)
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    print("\nSalary prediction model trained successfully.")
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"MAE:  ₹{mae:,.2f}")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Salary (₹)")
    plt.ylabel("Predicted Salary (₹)")
    plt.title("Actual vs Predicted Employee Salaries")
    plt.tight_layout()
    plt.savefig("salary_comparison.png")
    plt.close()

    return predictions


def predict_salary(model, age, experience, education, job_role):
    employee = pd.DataFrame([{
        "age": age,
        "experience_years": experience,
        "education": education,
        "job_role": job_role
    }])

    if not np.isfinite([age, experience]).all():
        raise ValueError("Age and experience must be finite numbers.")

    if age < 18 or experience < 0 or experience > age - 18:
        raise ValueError("Invalid age or experience.")

    if education not in ["Bachelor", "Master"]:
        raise ValueError("Education must be Bachelor or Master.")

    if job_role not in [
        "Developer", "Analyst", "Data Scientist", "Manager", "Tester"
    ]:
        raise ValueError("Invalid job role.")

    prediction = model.predict(employee)[0]

    print("\n========== SALARY PREDICTION ==========")
    print(f"Predicted annual salary: ₹{prediction:,.2f}")

    return prediction


def export_predictions(X_test, y_test, predictions):
    results = X_test.copy()
    results["actual_salary"] = y_test.to_numpy()
    results["predicted_salary"] = predictions
    results["absolute_error"] = abs(
        results["actual_salary"] - results["predicted_salary"]
    )

    results.to_csv("salary_predictions.csv", index=False)
    print("\nResults exported to salary_predictions.csv")
