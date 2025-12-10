# Generalised linear model 

# HOW IT WORKS --- 
    # Three components: Response distribution (e.g., Binomial, Poisson, Gamma), linear predictor, and link function connecting them.
    # Link function: Transforms the expected outcome to match the linear predictor (e.g., logit for logistic, log for Poisson, inverse for Gamma).
    # Maximum likelihood estimation: Coefficients are found by maximizing the likelihood function, not minimizing squared errors like ordinary least squares regression.

# WHEN TO USE ---
    # Non-normal outcomes: Counts, binary, proportions, or positive continuous data that violates normality.
    # Need interpretability: When you want p-values, confidence intervals, and explainable coefficients.
    # Avoid for complexity: Use tree-based models or neural networks for highly non-linear patterns instead.



import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# URL to your data (example using a CSV file)
# Replace this with your actual data URL
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Load data from URL
print("Loading data from URL...")
df = pd.read_csv(data_url)

# Display first few rows
print("\nFirst few rows of data:")
print(df.head())

# Example: Preparing data for GLM
# You'll need to modify this based on your actual data structure
# This example uses Titanic dataset to predict Age from other features

# Select features and target
# Remove rows with missing values in key columns
df_clean = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()

# Convert categorical variable to numeric
df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})

# Define features (X) and target (y)
X = df_clean[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]
y = df_clean['Age']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose your GLM model
# Options: PoissonRegressor, GammaRegressor, TweedieRegressor
print("\n" + "="*50)
print("Fitting Generalized Linear Model (Gamma GLM)...")
print("="*50)

# Fit a Gamma GLM (suitable for continuous positive data)
model = GammaRegressor(alpha=0.1, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Display coefficients
print(f"\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('GLM: Actual vs Predicted Values')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('glm_predictions.png', dpi=300)
print("\nPlot saved as 'glm_predictions.png'")
plt.show()

# Display sample predictions
print("\nSample Predictions:")
comparison = pd.DataFrame({
    'Actual': y_test[:10].values,
    'Predicted': y_pred[:10],
    'Difference': y_test[:10].values - y_pred[:10]
})
print(comparison)