# **Exploring and Visualizing a Simple Dataset**

***Developed By:*** [***M Aqeel Zafar***](https://github.com/maqeelzafar047)



This report summarizes the key steps for loading, inspecting, and visualizing a simple dataset (the Iris dataset) to uncover data trends and distributions. All content is based directly on the Colab notebook and presented in a clear, sequential, and professional manner.

## **1. Loading the Dataset**

There are two main methods for loading a dataset:

**Method 1: Using a CSV File (with Pandas)**

- Load any custom or real-world dataset.
- Full control over file path, column names, and formatting.
- Example:
  ```python
  import pandas as pd
  df = pd.read_csv('iris.csv')
  ```

**Method 2: Using a Built-in Dataset (with Seaborn)**

- Quickly explore a preloaded dataset without downloading files.
- Clean, ready-to-use data ideal for beginners.
- Example:
  ```python
  import seaborn as sns
  df = sns.load_dataset('iris')  # if available
  ```

## **2. Initial Data Inspection**

Once loaded, inspect the dataset to understand its structure:

- **Preview first rows:**
  ```python
  df.head()
  ```
- **Column names:**
  ```python
  df.columns
  ```
- **Shape (rows, columns):**
  ```python
  df.shape
  ```
- **Data types & non-null counts:**
  ```python
  df.info()
  ```
- **Statistical summary:**
  ```python
  df.describe()
  ```

## **3. Visualizing Relationships and Distributions**

### **3.1 Scatter Plot**

**Objective:** Show relationship between Sepal Length and Sepal Width, colored by species.

- Helps distinguish species clusters.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=df,
    x='SepalLengthCm',
    y='SepalWidthCm',
    hue='Species'
)
plt.title("Sepal Length vs Sepal Width by Species")
plt.show()
```

### **3.2 Histograms of All Features**

**Objective:** Visualize the distribution of each numeric feature (excluding ID).

- Understand spread, skewness, and range of measurements.

```python
# Drop non-numeric or ID column before plotting
df.drop(columns=['Id']).hist(edgecolor='black', figsize=(10, 8))
plt.suptitle("Histograms of Iris Features")
plt.tight_layout()
plt.show()
```

### **3.3 Histogram with KDE**

**Objective:** Show distribution of Sepal Length with a smooth KDE curve.

- KDE (Kernel Density Estimation) highlights peaks and overall shape.

```python
sns.histplot(
    data=df,
    x='SepalLengthCm',
    kde=True
)
plt.title("Distribution of Sepal Length")
plt.show()
```

### **3.4 Box Plot**

**Objective:** Compare Sepal Width distributions across species.

- Box plots display median, quartiles, and potential outliers.

```python
sns.boxplot(
    data=df,
    x='Species',
    y='SepalWidthCm'
)
plt.title("Box Plot of Sepal Width by Species")
plt.show()
```

## **4. Key Takeaways**

- **Dataset overview** through `head()`, `info()`, and `describe()` provides a solid foundation.
- **Scatter plots** reveal how features separate different species.
- **Histograms & KDE** illustrate individual feature distributions and potential skewness.
- **Box plots** allow for direct comparison of distributions and highlight outliers.

---

*Report generated on June, 2025*
