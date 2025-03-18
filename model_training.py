from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Model Evaluation
print(classification_report(y_test, y_pred))

# Save Model
with open('../models/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
