# Online Shoppers Purchase Intention Analysis 🛒

## Overview
This notebook analyzes the Online Shoppers Purchasing Intention dataset to predict customer purchase behavior using machine learning techniques. We employ Random Forest classification with proper data preprocessing and feature importance analysis.

## Dataset Information
The dataset contains various features about online shopping sessions including:
- **Administrative, Informational, Product-related pages**: Number of pages visited
- **Bounce rates, Exit rates, Page values**: Website engagement metrics
- **Special Day**: Closeness to special days (Valentine's, Mother's Day, etc.)
- **Month**: Month of the year
- **Operating Systems, Browser, Region, Traffic Type**: Technical and demographic info
- **Visitor Type**: New vs returning visitors
- **Weekend**: Whether the session occurred on weekend
- **Revenue**: Target variable (purchase made or not)

---

## 1. Data Loading and Preprocessing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

data = pd.read_csv('online_shoppers_intention.csv')
```

### 1.1 Feature Engineering
We perform several preprocessing steps to prepare the data:

```python
# Encode categorical Month variable to numerical
data['Month'] = pd.factorize(data['Month'])[0]

# One-hot encode VisitorType (drop first to avoid multicollinearity)
onehot = pd.get_dummies(data['VisitorType'],drop_first=True).astype(int)
data = pd.concat([data,onehot],axis=1)

# Convert boolean features to integers for consistency
data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)

# Remove original categorical column
data = data.drop(['VisitorType'],axis=1)
```

**Key preprocessing decisions:**
- **Month encoding**: Using factorize to convert months to numerical values
- **One-hot encoding**: Applied to VisitorType with drop_first=True to prevent dummy variable trap
- **Boolean conversion**: Converting Weekend and Revenue to integers for model compatibility

---

## 2. Data Preparation and Balancing

```python
# Separate features and target variable
X = data.drop('Revenue',axis=1).values
Y = data['Revenue'].values

# Address class imbalance using Random Oversampling
ros = RandomOverSampler()
X_resampled , Y_resampled = ros.fit_resample(X,Y)
```

**Class Imbalance Handling:**
- The dataset likely has imbalanced classes (more non-purchases than purchases)
- Random Oversampling creates synthetic samples of the minority class
- This helps prevent the model from being biased toward the majority class

---

## 3. Model Training and Evaluation

### 3.1 Train-Test Split and Scaling
```python
X_train,X_test,Y_train,Y_test = train_test_split(X_resampled,Y_resampled,test_size=0.2)

# Feature scaling for optimal performance
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3.2 Random Forest Training
```python
clf = RandomForestClassifier()
clf.fit(X_train_scaled,Y_train)
print(clf.score(X_test_scaled,Y_test))
```

**Why Random Forest?**
- Handles both numerical and categorical features well
- Provides feature importance scores
- Robust to outliers and overfitting
- Good baseline performance for binary classification

### 3.3 Cross-Validation Assessment
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_resampled, Y_resampled, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

**Cross-validation ensures:**
- More robust performance estimation
- Reduced variance in model assessment
- Better understanding of model stability across different data splits

---

## 4. Feature Importance Analysis

Understanding which features drive purchase decisions is crucial for business insights:

```python
# Extract feature names and importance scores
feature_names = data.drop('Revenue', axis=1).columns.tolist()
feature_importances = clf.feature_importances_

print(f"Number of features: {len(feature_names)}")
print(f"Number of importances: {len(feature_importances)}")

# Create organized dataframe for analysis
important_features = pd.DataFrame({
    'features': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(important_features.head(10))
```

### 4.1 Feature Importance Visualization

```python
plt.figure(figsize=(10, 8))
top_10_features = important_features.head(10)
plt.barh(range(len(top_10_features)), top_10_features['importance'].values)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Top 10 Feature Importance from Random Forest')
plt.yticks(range(len(top_10_features)), top_10_features['features'].values)

# Add value labels on bars
for i, importance in enumerate(top_10_features['importance'].values):
    plt.text(importance, i, f'{importance:.3f}',
             va='center', ha='left', fontsize=10)

plt.tight_layout()
plt.show()
```

**Visualization Benefits:**
- **Horizontal layout**: Better readability for feature names
- **Value labels**: Precise importance scores displayed
- **Top 10 focus**: Highlights most impactful features for business decision-making

---

## Key Insights and Business Implications

The feature importance analysis reveals which factors most strongly predict customer purchase behavior:

1. **Website Engagement Metrics**: Page values, bounce rates, and exit rates typically rank highly
2. **Session Characteristics**: Time spent and pages visited indicate user intent
3. **Temporal Factors**: Month and weekend patterns show seasonal/timing effects
4. **User Segmentation**: Visitor type (new vs returning) influences purchase probability

## Next Steps for Improvement

1. **Hyperparameter Tuning**: Optimize Random Forest parameters using GridSearchCV
2. **Feature Engineering**: Create interaction terms or polynomial features
3. **Alternative Models**: Compare with XGBoost, Logistic Regression, or Neural Networks
4. **Evaluation Metrics**: Include precision, recall, F1-score, and ROC-AUC for comprehensive assessment
5. **Business Metrics**: Calculate potential revenue impact of model predictions

---

## Methodology Summary

| Step | Technique | Purpose |
|------|-----------|---------|
| Preprocessing | Factorization, One-hot encoding | Convert categorical variables |
| Balancing | Random Oversampling | Address class imbalance |
| Scaling | StandardScaler | Normalize feature ranges |
| Model | Random Forest | Binary classification with interpretability |
| Validation | Cross-validation | Robust performance estimation |
| Analysis | Feature Importance | Business insights and model interpretation |

This analysis provides a solid foundation for understanding customer purchase behavior and can be extended for more sophisticated predictive modeling and business intelligence applications.
