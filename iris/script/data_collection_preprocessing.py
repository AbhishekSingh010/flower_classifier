import sklearn as sk
import joblib
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os


def get_project_root():
    """Get the absolute path to the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def load_data():
    try:
        data_path = os.path.join(get_project_root(), "data", "IRIS.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def preprocess(iris_df):
    if iris_df is None:
        return None, None, None, None
    
    try:
        X = iris_df.iloc[:, :-1].values
        y = iris_df['species'].values  # Make sure we're using the species column name

        # Fit label encoder and store the classes
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        # Store the original class names
        class_names = label_encoder.classes_
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save preprocessors and class names
        model_dir = os.path.join(get_project_root(), "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
        joblib.dump(class_names, os.path.join(model_dir, "class_names.pkl"))

        return X_train, X_test, y_train, y_test

    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None, None, None


def train_model(X_train, y_train, model_name):
    if any(x is None for x in [X_train, y_train]):
        st.error("Training data not available")
        return

    try:
        model_dir = os.path.join(get_project_root(), "model")
        os.makedirs(model_dir, exist_ok=True)

        models = {
            0: (RandomForestClassifier(n_estimators=100, random_state=42), "iris_model_randomforestc.pkl"),
            1: (DecisionTreeClassifier(random_state=42), "iris_model_decisiontreec.pkl"),
            2: (LogisticRegression(random_state=42), "iris_model_logisticregression.pkl"),
            3: (SVC(kernel='linear', random_state=42), "iris_model_svc.pkl"),
        }

        model_info = models.get(model_name)
        if model_info is None:
            st.error("Invalid model selection")
            return

        model, filename = model_info
        model.fit(X_train, y_train)
        save_path = os.path.join(model_dir, filename)
        joblib.dump(model, save_path)

    except Exception as e:
        st.error(f"Error in model training: {str(e)}")


def accuracy_check_all():
    try:
        model_dir = os.path.join(get_project_root(), "model")
        if not os.path.exists(model_dir):
            st.error("Model directory not found. Please train models first.")
            return None, None, None, None

        model_files = {
            "Random Forest": "iris_model_randomforestc.pkl",
            "Decision Tree": "iris_model_decisiontreec.pkl",
            "Logistic Regression": "iris_model_logisticregression.pkl",
            "SVM": "iris_model_svc.pkl"
        }

        models = {}
        for name, filename in model_files.items():
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                st.warning(f"Model {filename} not found. Please train all models first.")
                return None, None, None, None
            models[name] = joblib.load(path)

        X_train, X_test, y_train, y_test = preprocess(load_data())
        if X_test is None:
            return None, None, None, None

        accuracies = {}
        for name, model in models.items():
            pred = model.predict(X_test)
            accuracies[name] = accuracy_score(y_test, pred)

        return (accuracies["Random Forest"], 
                accuracies["Decision Tree"], 
                accuracies["Logistic Regression"], 
                accuracies["SVM"])

    except Exception as e:
        st.error(f"Error checking accuracy: {str(e)}")
        return None, None, None, None


def predict(model_name, sepal_length, sepal_width, petal_length, petal_width):
    try:
        model_dir = os.path.join(get_project_root(), "model")
        if not os.path.exists(model_dir):
            return "Model directory not found. Please train models first."

        # Load the saved transformers
        try:
            scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        except FileNotFoundError:
            return "Preprocessor files not found. Please train models first."

        # Load the model
        model_files = {
            0: "iris_model_randomforestc.pkl",
            1: "iris_model_decisiontreec.pkl",
            2: "iris_model_logisticregression.pkl",
            3: "iris_model_svc.pkl"
        }

        model_path = os.path.join(model_dir, model_files.get(model_name, ""))
        if not os.path.exists(model_path):
            return "Model not found. Please train the model first."

        model = joblib.load(model_path)

        # Scale the input features
        sample_input_scaled = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Get numeric prediction
        prediction = model.predict(sample_input_scaled)
        
        # Convert to flower name using label encoder
        predicted_species = label_encoder.inverse_transform(prediction)[0]
        
        # Add proper formatting for species name
        if isinstance(predicted_species, str) and not predicted_species.startswith("Iris-"):
            predicted_species = f"Iris-{predicted_species.lower()}"
            
        return predicted_species

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error making prediction"


