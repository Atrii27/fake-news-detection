from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
with open('../models/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
