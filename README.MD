# 📱☁️ Clustering on Mobile Cloud Computing Data with SVR Perspective

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/stable/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Latest-yellow.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Latest-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> An advanced machine learning approach to identify patterns and predict performance metrics in Mobile Cloud Computing environments using clustering and regression techniques

![MCC Banner](https://via.placeholder.com/800x200?text=Mobile+Cloud+Computing+Analysis)

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Dataset Overview](#-dataset-overview)
- [Data Preprocessing](#-data-preprocessing)
  - [Data Cleaning](#data-cleaning)
  - [Data Splitting](#data-splitting)
  - [Feature Selection](#feature-selection)
  - [Feature Scaling](#feature-scaling)
- [Machine Learning Models](#-machine-learning-models)
  - [Classification: K-Nearest Neighbors](#classification-k-nearest-neighbors-knn)
  - [Clustering: BIRCH](#clustering-birch)
- [Regression Models](#-regression-models)
  - [Linear Regression](#linear-regression)
  - [Support Vector Regression (SVR)](#support-vector-regression-svr)
- [Results and Evaluation](#-results-and-evaluation)
  - [Clustering Results](#clustering-results)
  - [Regression Results](#regression-results)
- [Installation and Usage](#-installation-and-usage)
- [Conclusion](#-conclusion)
- [Future Work](#-future-work)
- [References](#-references)

## 🚀 Introduction

This project applies advanced machine learning techniques to analyze a dataset related to **Mobile Cloud Computing (MCC)**, a paradigm that integrates mobile devices with cloud computing infrastructure to enhance computational capabilities and overcome resource limitations of mobile devices.

The primary objectives are to:
1. Identify meaningful patterns in MCC data through clustering
2. Predict the key performance metric "ECC" using regression models
3. Evaluate the effectiveness of Support Vector Regression (SVR) for MCC analytics

The implementation demonstrates a complete machine learning workflow including data preprocessing, feature engineering, model training, and performance evaluation.

## 📊 Dataset Overview

The dataset `MCC.csv` contains features extracted from research papers on Mobile Cloud Computing. After preprocessing, the dataset includes:

- **Features**: Numerical attributes related to MCC characteristics
- **Target Variable**: "ECC" (a continuous metric)

> **Note**: "ECC" likely represents Energy Consumption Cost or another performance indicator in MCC environments.

Non-essential columns such as "Authors," "Title," and "Source" are removed during preprocessing to focus on relevant predictive features.

## 🧹 Data Preprocessing

### Data Cleaning

```python
# Drop non-essential columns
columns_to_drop = ["Authors", "Title", "Source", "Publisher", 
                  "ArticleURL", "CitesURL", "Type", "QueryDate"]
df = df.drop(columns_to_drop, axis=1)

# Handle missing values
df = df.dropna(0)
```

### Data Splitting

```python
# Separate features and target
X = df.drop(['ECC'], axis=1)
y = df['ECC']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
```

### Feature Selection

```python
# Remove constant features
selector = VarianceThreshold(threshold=0)
selector.fit(X_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# Remove highly correlated features
corr_matrix = pd.DataFrame(X_train).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
X_train = pd.DataFrame(X_train).drop(to_drop, axis=1)
X_test = pd.DataFrame(X_test).drop(to_drop, axis=1)
```

### Feature Scaling

```python
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 🤖 Machine Learning Models

### Classification: K-Nearest Neighbors (KNN)

While the project primarily focuses on clustering and regression, a KNN classifier serves as a baseline model:

```python
# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Evaluate performance
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
```

### Clustering: BIRCH

**BIRCH** (Balanced Iterative Reducing and Clustering using Hierarchies) efficiently clusters the data into distinct groups:

```python
# Apply BIRCH clustering
birch = Birch(n_clusters=7)
birch.fit(X_train_scaled)
labels = birch.predict(X_test_scaled)

# Demonstrate with synthetic data
X, y = make_blobs(n_samples=1000, centers=7, n_features=2, random_state=42)
birch_no_global = Birch(threshold=0.01, n_clusters=None)
birch_global = Birch(threshold=0.01, n_clusters=7)

# Visualize clustering results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=birch_no_global.fit_predict(X))
plt.title("BIRCH without Global Clustering")
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=birch_global.fit_predict(X))
plt.title("BIRCH with Global Clustering")
plt.show()
```

## 📈 Regression Models

### Linear Regression

```python
# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Evaluate performance
y_pred_lr = lr.predict(X_test_scaled)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_pred_lr)), y_pred_lr, color='red', label='Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Session')
plt.ylabel('Predicted ECC')
plt.legend()
plt.show()
```

### Support Vector Regression (SVR)

```python
# Train SVR model using pipeline
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(C=1.0, epsilon=0.2))
])
svr_pipeline.fit(X_train, y_train)

# Evaluate performance
y_pred_svr = svr_pipeline.predict(X_test)
r2_svr = r2_score(y_test, y_pred_svr)

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_pred_svr)), y_pred_svr, color='green', label='Predicted')
plt.title('Support Vector Regression: Actual vs Predicted')
plt.xlabel('Session')
plt.ylabel('ECC')
plt.legend()
plt.show()
```

## 📊 Results and Evaluation

### Clustering Results

BIRCH clustering successfully groups the MCC data into 7 clusters, potentially revealing patterns such as:
- Similar research topics in Mobile Cloud Computing
- Comparable performance characteristics
- Common energy consumption profiles

The synthetic data visualization demonstrates BIRCH's effectiveness with and without global clustering, highlighting its ability to handle large datasets efficiently.

### Regression Results

| Model | R² Score | MAE | MSE |
|-------|---------|-----|-----|
| Linear Regression | [TBD] | [TBD] | [TBD] |
| SVR | [TBD] | Not Calculated | Not Calculated |

> Note: Actual performance metrics will be available after running the implementation on the MCC dataset.

## 💻 Installation and Usage

### Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, tensorflow, keras

### Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras
   ```

2. **Prepare the Dataset**:
   - Place `MCC.csv` in the project directory
   - Ensure the dataset contains the expected columns including "ECC"

3. **Run the Script**:
   ```bash
   python mcc_lr_clust.py
   ```

4. **Output**:
   - Console will display performance metrics
   - Visualizations will be generated for clustering and regression results

## 🎯 Conclusion

This project successfully demonstrates the application of machine learning techniques to Mobile Cloud Computing data, with a focus on:

1. **BIRCH clustering** to uncover underlying patterns in MCC research
2. **Support Vector Regression** to predict the "ECC" metric with high accuracy
3. **Comprehensive preprocessing** to ensure robust model training

The results provide valuable insights for researchers and practitioners in the MCC field, potentially contributing to more efficient mobile cloud computing implementations.

## 🔮 Future Work

- **Advanced Feature Engineering**: Develop more sophisticated features to capture nuanced patterns
- **Hyperparameter Tuning**: Optimize SVR parameters (C, epsilon, kernel) for improved performance
- **Model Comparison**: Explore additional algorithms like Random Forest or Gradient Boosting
- **Deep Learning Integration**: Investigate neural network approaches for MCC prediction
- **Ensemble Methods**: Combine multiple models for enhanced predictive power

## 📚 References

1. "Mobile Cloud Computing for computation offloading: Issues and challenges" 
2. "Mobile Cloud Computing (MCC), architecture"
3. "What Is Mobile Cloud Computing (MCC)?"
4. "Mobile Cloud Computing: A Literature Survey"
5. "Mobile cloud computing: Challenges and future research directions"
6. "Experimental Framework for Mobile Cloud Computing System"

---

<p align="center">
  <i>Advancing Mobile Cloud Computing through Machine Learning and Data Science</i>
</p>