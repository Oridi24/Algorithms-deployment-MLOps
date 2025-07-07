# main.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils import load_and_prepare_data, evaluate_model

def main():
    print(" Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler, target_names = load_and_prepare_data(test_size=0.2)

    print(" Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    print(" Evaluating model...")
    evaluate_model(model, X_test, y_test, target_names)

    print(" Saving model and scaler...")
    joblib.dump(model, "wine_model_multiclass.pkl")
    joblib.dump(scaler, "wine_scaler.pkl")
    print(" Model and scaler saved.")

if __name__ == "__main__":
    main()
