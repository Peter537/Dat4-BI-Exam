import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def plot_linear_regression(df, job_title):
    selected_job_df = df[df['job_title'] == job_title]

    X = selected_job_df['work_year'].values.reshape(-1, 1)
    y = selected_job_df['salary_in_usd'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Print R-squared and RMSE
    print(f"R-squared: {r_squared:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Plot the results
    plt.scatter(X_test, y_test, color='black', label='Actual Data')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
    plt.xlabel('Work Year')
    plt.ylabel('Salary in USD')
    plt.title(f'Linear Regression for {job_title}')
    plt.legend()
    plt.show()

    # Print the linear regression equation
    print("Linear Regression Equation:")
    print(f"y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")

# Example usage:
# plot_linear_regression(df, 'Data Scientist')
