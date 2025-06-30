# **House Price Prediction Report**

***Developed By:*** [***M Aqeel Zafar***](https://github.com/maqeelzafar047)

## **1. Importing Libraries**

The following Python libraries were used:

- `pandas`, `numpy`: Data handling
- `matplotlib.pyplot`, `seaborn`: Visualization
- `sklearn`: For machine learning models and preprocessing

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## **2. Uploading and Loading Data**

Files were uploaded using Colab's file upload utility:

```python
from google.colab import files
uploaded = files.upload()
```

Three datasets were loaded:

```python
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")
```

---

## **3. Data Inspection**

Initial inspection was performed using:

```python
train_df.head()
train_df.info()
train_df.describe()
```

- Identified numerical and categorical columns.
- Checked for null values.

---

## **4. Data Visualization**

Several plots were used to understand the relationship between variables:

### **Price Distribution**

```python
sns.histplot(train_df['SalePrice'], kde=True)
```

- Showed skewed price distribution.

### **Correlation Heatmap**

```python
plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
```

- Visualized correlation between numerical features and `SalePrice`.

### **Scatter Plot**

```python
plt.scatter(train_df['GrLivArea'], train_df['SalePrice'])
```

- To observe how `GrLivArea` (above ground living area) impacts price.

---

## **5. Data Preprocessing**

- Null values handled or columns dropped.
- Features selected for training.
- Feature scaling was applied using `StandardScaler`.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## **6. Model Training & Evaluation**

Two models were trained:

### **Linear Regression**

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### **Gradient Boosting Regressor**

```python
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
```

### **Model Evaluation**

Used Mean Absolute Error and Mean Squared Error:

```python
mean_absolute_error(y_test, predictions)
mean_squared_error(y_test, predictions)
```

---

## **7. Predictions on Test Data**

```python
final_predictions = model.predict(test_df_selected_features)
submission = pd.DataFrame({"Id": test_df.Id, "SalePrice": final_predictions})
submission.to_csv("final_submission.csv", index=False)
```

- Final predictions saved in the format required for submission.

---

*Report generated on June, 2025*

