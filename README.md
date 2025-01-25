# Fertilizer-Prediction


Fertilizer Prediction Using Random Forest Classifier

This project implements a machine learning model to predict the most suitable fertilizer for crops based on various environmental and soil parameters. It uses a Random Forest Classifier trained on a dataset containing agricultural features such as temperature, humidity, soil type, and crop type.

Features
- Input Parameters: 
  - Temperature
  - Humidity
  - Moisture
  - Soil Type
  - Crop Type
  - Nitrogen Level
  - Potassium Level
  - Phosphorous Level
- Output: Recommended fertilizer name.
- Categorical data (e.g., soil type, crop type) is automatically one-hot encoded.
- Easy-to-use prediction function for real-time fertilizer recommendations.

Prerequisites
To run this project, you need to have the following installed:
- Python (3.6 or above)
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `numpy`

Setup Instructions
1. Clone the repository or download the script.
2. Prepare your dataset with the following columns:
   - `Temperature`, `Humidity`, `Moisture`, `Soil Type`, `Crop Type`, `Nitrogen`, `Potassium`, `Phosphorous`, and `Fertilizer Name`.
3. Update the file path for the dataset in the script:
   ```python
   data = pd.read_csv('/path/to/FertilizerPrediction.csv')
   ```
4. Run the script:
   ```bash
   python fertilizer_prediction.py
   ```

 Usage
1. The script splits the data into training and testing sets and trains a Random Forest Classifier.
2. It evaluates the model's accuracy and provides a `predict_fertilizer()` function for real-time predictions.
3. You can input your environmental and soil parameters during runtime to get fertilizer recommendations:
   ```bash
   Enter temperature: 30
   Enter humidity: 60
   Enter moisture: 25
   Enter soil type: Sandy
   Enter crop type: Rice
   Enter nitrogen: 12
   Enter potassium: 8
   Enter phosphorus: 10
   ```
   Example output:
   ```
   The recommended fertilizer is: Urea
   ```

 How It Works
- Preprocessing: Categorical variables (`Soil Type` and `Crop Type`) are converted to numerical values using one-hot encoding.
- Model: A Random Forest Classifier is trained on 80% of the dataset, and the remaining 20% is used for testing.
- Evaluation: The model's accuracy is calculated on the test set.
- Prediction: User inputs are processed, encoded, and aligned with the model's training features before predicting the fertilizer.

 Model Evaluation
- The script outputs the model's accuracy score after training, e.g.:
  
  Accuracy: 95.00%
  

 Customization
- You can adjust the test-train split ratio in the following line:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```
- Modify the n_estimators parameter of the Random Forest Classifier to tune performance:
  python
  model = RandomForestClassifier(n_estimators=100, random_state=42)

Dataset Information
The dataset used for this project is available: (https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction).


Notes
- Ensure that the input data format matches the expected columns and values for accurate predictions.
- The dataset file path and column names must be updated based on your specific dataset.


