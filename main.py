
from analyzer import (
    load_employee_data,
    analyze_employees,
    prepare_data,
    train_salary_model,
    evaluate_model,
    predict_salary,
    export_predictions
)


def main():
    print("=" * 50)
    print("       EMPLOYEE SALARY PREDICTION SYSTEM")
    print("=" * 50)

    try:
        df = load_employee_data("employees.csv")

        analyze_employees(df)

        X_train, X_test, y_train, y_test = prepare_data(df)

        model = train_salary_model(X_train, y_train)

        predictions = evaluate_model(model, X_test, y_test)

        predict_salary(
            model,
            age=27,
            experience=4,
            education="Bachelor",
            job_role="Developer"
        )

        export_predictions(X_test, y_test, predictions)

        print("\nProject completed successfully!")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
