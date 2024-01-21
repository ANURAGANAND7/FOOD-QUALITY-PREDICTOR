import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# dataset
data = {
    'Ingredients': [4, 5, 7, 3, 6, 8, 9, 4, 5, 6],
    'CookingTime': [25, 30, 35, 20, 45, 50, 55, 30, 40, 35],
    'FoodQuality': [6.5, 7.0, 7.5, 6.0, 8.0, 8.5, 9.0, 7.0, 7.5, 8.0],
    'Spices': [2, 1, 3, 1, 4, 2, 5, 3, 2, 4],  
    'PreparationTime': [15, 20, 25, 10, 30, 35, 40, 20, 30, 25], 
    'AmbianceScore': [8, 7, 9, 8, 6, 7, 8, 9, 7, 8],  
    'PresentationScore': [7, 8, 9, 8, 7, 6, 9, 8, 7, 8], 
    'SpicinessLevel': [2, 3, 1, 2, 3, 1, 2, 3, 2, 3],  
    'OverallSatisfaction': [8.5, 9.0, 8.0, 8.5, 9.0, 8.5, 9.0, 8.0, 8.5, 9.0]  
}

df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['Ingredients', 'CookingTime', 'Spices', 'PreparationTime', 'AmbianceScore', 'PresentationScore', 'SpicinessLevel']]
y = df['FoodQuality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared (R2) Score:", r2)

# Use the trained model for prediction
X_new = pd.DataFrame({
    'Ingredients': [1],
    'CookingTime': [20],
    'Spices': [15],
    'PreparationTime': [15],
    'AmbianceScore': [4],
    'PresentationScore': [8],
    'SpicinessLevel': [2]
})
predicted_quality = model.predict(X_new)
print("Predicted Food Quality:", predicted_quality[0])

# Create a bar graph comparing actual vs. predicted food quality
bar_width = 0.35
index = np.arange(len(X_test))
plt.bar(index, y_test, width=bar_width, label='Actual Food Quality')
plt.bar(index + bar_width, y_pred, width=bar_width, label='Predicted Food Quality')
plt.xlabel('Sample Index')
plt.ylabel('Food Quality')
plt.title('Actual vs. Predicted Food Quality for Test Set')
plt.xticks(index + bar_width / 2, X_test.index)  # Assuming the index represents sample IDs
plt.legend()
plt.show()  



