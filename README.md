

# Customer Churn Prediction Using ANN Model

This project aims to predict customer churn using an Artificial Neural Network (ANN) model. The model is built using TensorFlow and Keras, and the application is deployed using Streamlit for an interactive user interface.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Project Overview
Customer churn prediction is crucial for businesses to retain their customers. This project uses an ANN model to predict whether a customer is likely to churn based on various features such as credit score, geography, gender, age, balance, and more.

## Installation
To run this project, you need to have Python installed on your machine. Follow the steps below to set up the project:

1. Clone the repository:
   ```sh
   git clone https://github.com/sanskaryo/Churn-Prediction-Using_ANN.git
   cd Churn-Prediction-Using_ANN
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```sh
   streamlit run app.py
   ```

This will start a local Streamlit server where you can interact with the application through a web interface.

## Usage
After starting the Streamlit app, enter customer details (such as credit score, age, geography, etc.) in the input fields to predict whether the customer is likely to churn. The model processes the input and provides a real-time prediction of churn likelihood.

## Model Training
The model was trained and evaluated in `experiments.ipynb`, a Jupyter notebook that includes the following steps:

1. **Data Preprocessing**: Scaling numeric features, encoding categorical data (gender and geography), and preparing the dataset for ANN input.
2. **Model Building**: Defining the ANN structure with input, hidden, and output layers tailored for binary classification.
3. **Model Training**: Training the model on historical customer data, allowing it to learn patterns associated with churn.
4. **Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.

The trained model is saved as `model.keras`, which is then loaded in the `app.py` file for predictions.

## Prediction
The `app.py` file serves as the main interface for making predictions. It performs the following tasks:

- **Load the Model and Encoders**: The model, label encoders, and scaler are loaded.
- **Process Input Data**: User inputs are scaled and encoded as needed.
- **Generate Prediction**: The processed input data is fed into the model to obtain a prediction.
- **Display Result**: The result ("Churn" or "No Churn") is shown on the app interface.

## File Descriptions

- **`app.py`**: The main Streamlit application for user interaction and predictions.
- **`experiments.ipynb`**: Jupyter notebook that contains data exploration, preprocessing, model building, training, and evaluation code.
- **`model.keras`**: The trained ANN model saved in Keras format, used for deployment.
- **`label_encoder_gender.pkl`**: Pickle file containing the encoder for the gender feature.
- **`onehot_encoder_geo.pkl`**: Pickle file containing the encoder for the geography feature.
- **`scaler.pkl`**: Pickle file containing the scaler for numerical features.
- **`requirements.txt`**: File listing all dependencies required to run the project.

## Dependencies
This project requires the following Python libraries:

- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorboard`
- `matplotlib`
- `streamlit`
- `scikeras`

Install these dependencies by running:
```sh
pip install -r requirements.txt
```

## Acknowledgements
This project is based on tutorials from Krish Naik's Udemy course. Special thanks to Krish Naik for providing comprehensive guidance on building churn prediction models with machine learning.

---
## Screenshots

1. **Screenshot 1**  
   ![Screenshot 1](https://github.com/user-attachments/assets/31d59286-ccb3-45d8-9e6f-1dd4444cce27)

2. **Screenshot 2**  
   ![Screenshot 2](https://github.com/user-attachments/assets/8b8ec61c-93fd-4ae5-bd9a-dbb6a27e807d)



