please visit inside the iris directory.

# 🌸 Iris Flower Classification

A machine learning web application that classifies Iris flowers based on their measurements using multiple classification algorithms.

## 🚀 Features

- Multiple classification models:
  - Random Forest
  - Decision Tree
  - Logistic Regression
  - Support Vector Machine (SVM)
- Interactive web interface using Streamlit
- Real-time predictions
- Model accuracy visualization
- Easy-to-use measurement input sliders

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/AbhishekSingh010/flower_classifier.git
cd iris
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit/main.py
```

2. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

3. Follow the steps in the application:
   - Select a classification model
   - Train the model
   - Input flower measurements
   - Get predictions

## 📁 Project Structure

```
iris/
├── data/
│   └── IRIS.csv
├── model/
│   └── (trained models will be saved here)
├── script/
│   └── data_collection_preprocessing.py
├── streamlit/
│   └── main.py
├── requirements.txt
└── README.md
```

## 📊 Dataset

The project uses the classic Iris dataset containing measurements for three species of Iris flowers:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

## 🤝 Contributing

Feel free to open issues and pull requests!

## 📜 License

This project is licensed under the MIT License.
