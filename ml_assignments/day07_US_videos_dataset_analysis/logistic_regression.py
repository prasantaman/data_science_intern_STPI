import pandas as pd
df = pd.read_csv("USvideos.csv")
df['description'] = df['description'].fillna("")
df['title_length'] = df['title'].apply(len)
df['publish_time'] = pd.to_datetime(df['publish_time'])
from datetime import datetime
import pandas as pd
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['days_since_publish'] = (pd.Timestamp.now(tz='UTC') - df['publish_time']).dt.days
print(df[['publish_time', 'days_since_publish']].head())
threshold = df['views'].quantile(0.75)

df['viral'] = df['views'].apply(lambda x: 1 if x > threshold else 0)
print(df['viral'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. Select features
numeric_features = ['likes', 'dislikes', 'comment_count', 
                    'title_length', 'days_since_publish']

categorical_features = ['channel_title', 'category_id']

# 2. One-Hot Encoding
df_encoded = pd.get_dummies(df[numeric_features + categorical_features], drop_first=True)

# 3. Targets
y_reg = df['views']      # Regression target
y_clf = df['viral']      # Classification target

# 4. Scaling (only numeric)
scaler = StandardScaler()

df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# 5. Train-Test Split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    df_encoded, y_reg, test_size=0.2, random_state=42
)

# For classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    df_encoded, y_clf, test_size=0.2, random_state=42
)

# Check shapes
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Train model
model = LinearRegression()
model.fit(X_train, y_train_reg)

# 2. Predict
y_pred = model.predict(X_test)

# 3. Evaluation
mse = mean_squared_error(y_test_reg, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred)
r2 = r2_score(y_test_reg, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

# 4. Feature Importance (coefficients)
coefficients = pd.Series(model.coef_, index=X_train.columns)

# Top 10 important features
top_features = coefficients.abs().sort_values(ascending=False).head(10)

# Plot
plt.figure()
top_features.plot(kind='bar')
plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt

# 1. Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_clf, y_train_clf)

# 2. Predict
y_pred_clf = clf.predict(X_test_clf)
y_prob = clf.predict_proba(X_test_clf)[:, 1]

# 3. Evaluation
accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)
roc_auc = roc_auc_score(y_test_clf, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

# 4. Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# 5. ROC Curve
fpr, tpr, _ = roc_curve(y_test_clf, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()