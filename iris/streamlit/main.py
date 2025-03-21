import streamlit as st
import joblib
import numpy as np
import sys
import os

# Add project root to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from script.data_collection_preprocessing import load_data, preprocess, train_model, accuracy_check_all, predict

# Streamlit UI Config
st.set_page_config(page_title="Iris Flower Classifier", layout="centered")
st.title("🌺 Iris Flower Classification")

# Sidebar - Model Selection
st.sidebar.header("Select Model:")
model_options = ["Random Forest", "Decision Tree", "Logistic Regression", "SVM"]
model_name = st.sidebar.radio("Choose Model:", model_options, index=0)

# Map model selection to integer
model_index = model_options.index(model_name)

# Sidebar - Input Fields
st.sidebar.header("Enter Flower Measurements:")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Add an explanation section
st.sidebar.markdown("""
### How to use:
1. Select a model from the options above
2. Train the model using 'Train Model' button
3. Check model accuracy (optional)
4. Adjust flower measurements using sliders
5. Click 'Predict' to see results
""")

# Model Training Button with progress feedback
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test = preprocess(load_data())
        if X_train is not None:
            train_model(X_train, y_train, model_index)
            st.sidebar.success(f"✅ {model_name} Model Trained Successfully!")
        else:
            st.sidebar.error("❌ Error in training model")

# Model Accuracy with better visualization
if st.sidebar.button("Check Model Accuracy"):
    with st.spinner("Calculating accuracies..."):
        rfc_acc, dtc_acc, lr_acc, svc_acc = accuracy_check_all()
        if all(acc is not None for acc in [rfc_acc, dtc_acc, lr_acc, svc_acc]):
            accuracy_dict = {
                "Random Forest": rfc_acc,
                "Decision Tree": dtc_acc,
                "Logistic Regression": lr_acc,
                "SVM": svc_acc
            }
            current_acc = accuracy_dict[model_name]
            st.sidebar.info(f"Model Accuracy: **{current_acc:.2%}**")
            
            # Add visual indicator of accuracy
            if current_acc >= 0.9:
                st.sidebar.success("🎯 High Accuracy!")
            elif current_acc >= 0.7:
                st.sidebar.warning("📊 Moderate Accuracy")
            else:
                st.sidebar.error("⚠️ Low Accuracy")

# Predict Button with enhanced display
if st.button("Predict Flower Type"):
    with st.spinner("🔍 Analyzing flower measurements..."):
        prediction = predict(model_index, sepal_length, sepal_width, petal_length, petal_width)
        
        # Create a summary of input measurements
        st.write("### Input Measurements:")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- Sepal Length: {sepal_length:.1f} cm")
            st.write(f"- Sepal Width: {sepal_width:.1f} cm")
        with col2:
            st.write(f"- Petal Length: {petal_length:.1f} cm")
            st.write(f"- Petal Width: {petal_width:.1f} cm")
        
        # Display prediction result with improved formatting
        if isinstance(prediction, str) and ("Error" in prediction or "not found" in prediction):
            st.error(f"⚠️ {prediction}")
        else:
            species_name = prediction.replace("Iris-", "")  # Remove prefix for cleaner display
            st.success(f"""
            ### 🌸 Prediction Result
            This iris flower appears to be:  
            **{species_name.title()}**
            """)
