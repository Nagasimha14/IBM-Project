import numpy as np
import optuna
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "GradientBoosting", "SVC"])
    
    if classifier_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    elif classifier_name == "GradientBoosting":
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

    else:
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        model = SVC(C=C, kernel=kernel)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

# Load dataset
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Train best model
best_params = study.best_params
if best_params["classifier"] == "RandomForest":
    model = RandomForestClassifier(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"])
elif best_params["classifier"] == "GradientBoosting":
    model = GradientBoostingClassifier(n_estimators=best_params["n_estimators"], learning_rate=best_params["learning_rate"])
else:
    model = SVC(C=best_params["C"], kernel=best_params["kernel"])

model.fit(X_train, y_train)
joblib.dump(model, "best_model.pkl")
print("Model saved as best_model.pkl")
