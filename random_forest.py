import pandas as pd
import numpy as np
import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('heart_v2.csv')



X = df.drop('heart disease', axis=1)
y = df['heart disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

random_param_grid = {
    'n_estimators': np.arange(50, 201, 10),
    'max_depth': [None] + list(np.arange(5, 30, 5)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=random_param_grid,
                                   cv=5, n_iter=20, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print("Best Parameters (Random Search):", random_search.best_params_)
print("Validation Score:", random_search.best_score_)

y_pred_random = random_search.predict(X_test)
print("Test Accuracy (Random Search):", accuracy_score(y_test, y_pred_random))
print(confusion_matrix(y_test, y_pred_random))
print(classification_report(y_test, y_pred_random))

best_model = random_search.best_estimator_
pickle.dump(best_model, open('model.pkl', 'wb'))
print("model.pkl saved.")
