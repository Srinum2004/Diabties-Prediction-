# Diabetes Prediction Web Application 🩺✨

This project implements a machine learning model to predict the likelihood of diabetes based on various health parameters. It includes a Jupyter Notebook for data analysis, model training, and evaluation, and a Flask web application for easy deployment and interaction with the trained model.
<img width="1920" height="1080" alt="Screenshot 2025-07-17 151936" src="https://github.com/user-attachments/assets/f2a1aa83-5a7e-40f3-b891-e7e4ef76fd08" />


## Table of Contents

* [Project Overview](#project-overview)

* [Features](#features)

* [Technologies Used](#technologies-used)

* [Project Structure](#project-structure)

* [Setup and Installation](#setup-and-installation)

* [Usage](#usage)

* [Output Screenshot](#output-screenshot) 📸

* [Model Details](#model-details)

* [Future Improvements](#future-improvements)

* [License](#license)

* [Contact](#contact)

## Project Overview

The goal of this project is to provide a user-friendly tool for predicting diabetes. It leverages a dataset of health indicators, performs comprehensive data preprocessing, trains and evaluates multiple machine learning models, and deploys the best-performing model as a web service.

## Features

* **Data Preprocessing:** Handles missing values (represented as zeros) and scales features. 🧹

* **Exploratory Data Analysis (EDA):** Visualizes data distributions, outliers, and correlations. 📊

* **Multiple Model Evaluation:** Compares the performance of various classification algorithms:

  * Logistic Regression

  * Random Forest

  * Gradient Boosting

  * Support Vector Machine (SVM)

  * Neural Network (MLPClassifier)

* **Class Imbalance Handling:** Utilizes SMOTE to address imbalanced classes in the dataset. ⚖️

* **Hyperparameter Tuning:** Demonstrates GridSearchCV for optimizing model performance (shown for Random Forest). ⚙️

* **Model Persistence:** Saves the best-performing model and scaler for future use. 💾

* **Web Application:** A simple Flask interface to input health parameters and get real-time diabetes predictions. 🌐

## Technologies Used

* **Python** 🐍

* **Flask:** Web framework for deployment. 🚀

* **pandas:** Data manipulation and analysis. 🐼

* **NumPy:** Numerical operations. ✨

* **Matplotlib & Seaborn:** Data visualization. 📈

* **Scikit-learn:** Machine learning algorithms and tools (preprocessing, model selection, metrics). 🧠

* **Imbalanced-learn (SMOTE):** Handling imbalanced datasets. 🤝

* **joblib:** Model serialization. 📦

## Project Structure

```
.
├── app.py                  # Flask web application for prediction
├── model.ipynb             # Jupyter Notebook for ML model training and evaluation
├── requirements.txt        # List of Python dependencies
├── Data/                   # Directory to store your dataset (e.g., diabetes.csv)
│   └── diabetes.csv
└── Notebook/               # Directory to store trained models and scalers
    ├── diabetes_prediction_model.pkl
    └── scaler.pkl

```

**Note:** The `Data/diabetes.csv` and `Notebook/` directories (containing the `.pkl` files) are assumed to exist. You'll need to place your `diabetes.csv` file in the `Data` folder and run `model.ipynb` to generate the `.pkl` files in the `Notebook` folder.

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
   ```

   (Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details)

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Model Training and Evaluation (Jupyter Notebook)

To understand the data, train the models, and evaluate their performance:

1. Ensure you have the `diabetes.csv` file in the `Data/` directory.

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook model.ipynb
   ```

3. Run all cells in the notebook. This will:

   * Load and preprocess the data.

   * Perform EDA and visualize insights.

   * Train and evaluate various ML models.

   * Perform hyperparameter tuning for Random Forest.

   * Select and save the `GradientBoostingClassifier` model and the `StandardScaler` to the `Notebook/` directory.

### 2. Running the Web Application (Flask)

Once the model and scaler are saved:

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. Open your web browser and go to the address displayed in the terminal (e.g., `http://127.0.0.1:5000`).

3. Enter the required health parameters in the form and click "Predict" to see the diabetes prediction.

## Output Screenshot 📸

<img width="1920" height="1080" alt="Screenshot 2025-07-17 151803" src="https://github.com/user-attachments/assets/eed7c902-2d25-40fd-a7c4-a3bd74116bb8" />
<img width="1920" height="1080" alt="Screenshot 2025-07-17 152128" src="https://github.com/user-attachments/assets/24129205-ae2f-46b1-b734-cefdbb52ece1" />



## Model Details

* **Best Performing Model:** Gradient Boosting Classifier

* **Key Predictive Features:** Glucose level, BMI, and Age.

* **Performance (Gradient Boosting):**

  * **Accuracy:** ~75.97%

  * **ROC AUC:** ~0.83

## Future Improvements

* Integrate more advanced deep learning models. 💡

* Explore different feature engineering techniques. 🛠️

* Implement a more sophisticated UI/UX for the web application. ✨

* Add model monitoring and re-training pipelines. 🔄

* Expand the dataset for better generalization. 📈

## License

This project is licensed under the MIT License.
you are always welcome to add any new feature or any type of enhancement to this project.
just email me and do.

## Contact

For any questions or suggestions, feel free to reach out:

* **Your Name:** Srinibash Mishra

* **GitHub:** https://github.com/Srinum2004

* **Email:** srinibashmishra2004@gmail.com
