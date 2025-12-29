# Fuel Efficiency Prediction using TensorFlow

This project focuses on predicting vehicle fuel efficiency (miles per gallon – MPG) using machine learning techniques implemented with **TensorFlow and Keras**. It demonstrates the complete ML workflow, including data exploration, preprocessing, visualization, model building, training, and evaluation.

---

## 📌 Project Overview

Fuel efficiency prediction helps understand how vehicle characteristics such as engine size, weight, horsepower, and model year affect fuel consumption. In this project, a **neural network regression model** is trained to predict MPG using historical automobile data.

The project emphasizes:

* Exploratory Data Analysis (EDA)
* Feature correlation analysis
* Neural network modeling using TensorFlow
* Model evaluation using MAE and MSE

---

## 📂 Dataset Description

The dataset contains **398 car records** with the following features:

* **mpg** (target variable)
* cylinders
* displacement
* horsepower
* weight
* acceleration
* model year
* origin
* car name

Missing and inconsistent values (especially in `horsepower`) are handled during preprocessing. Categorical features like `origin` are encoded numerically.

---

## 📊 Exploratory Data Analysis

The following visualizations are generated:

### 1️⃣ Distribution Analysis

* **Cylinder distribution** shows most cars have 4, 6, or 8 cylinders.
* **Origin distribution** highlights vehicles from different regions.

### 2️⃣ Correlation Heatmap

A correlation matrix is used to understand relationships between features:

* MPG is **negatively correlated** with weight, horsepower, displacement, and cylinders.
* MPG is **positively correlated** with model year and origin.

These insights guide feature selection and model design.

---

## 🧠 Model Architecture

The regression model is built using **TensorFlow (Keras Sequential API)**:

* Dense layer (64 neurons, ReLU)
* Batch Normalization
* Dense layer (32 neurons, ReLU)
* Dropout (to reduce overfitting)
* Batch Normalization
* Output Dense layer (1 neuron for MPG prediction)

**Optimizer:** Adam
**Loss Function:** Mean Squared Error (MSE)
**Metrics:** MAE, MSE

---

## 📈 Model Training & Evaluation

The model is trained for multiple epochs with validation data.

### Training Observations:

* **MAE and MSE decrease steadily** over epochs
* Validation curves closely follow training curves
* No severe overfitting observed

Plots included:

* MAE vs Epochs
* MSE vs Epochs

These confirm that the model learns meaningful patterns from the data.

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* TensorFlow / Keras
* VS Code

---

## 🚀 How to Run

1. Clone the repository
2. Install required libraries

   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn
   ```
3. Run the Python script or Jupyter notebook
4. View visualizations and model performance

---

## ✅ Conclusion

This project demonstrates how **deep learning regression models** can effectively predict fuel efficiency. By combining data analysis, visualization, and TensorFlow modeling, the system provides accurate MPG predictions and valuable insights into factors affecting fuel consumption.

---

## 📸 Screenshots

The repository includes screenshots showing:

###* Dataset overview in VS Code
<img width="1920" height="1020" alt="Screenshot 2025-12-29 194120" src="https://github.com/user-attachments/assets/bf62d185-c0cb-49e8-889d-77411a506e9f" />

---
###* Feature distributions
<img width="1920" height="1020" alt="Screenshot 2025-12-29 194152" src="https://github.com/user-attachments/assets/70629058-cbfd-4682-9b43-8ce1ba6e6344" />

---
###* Correlation heatmap
<img width="1920" height="1020" alt="Screenshot 2025-12-29 194204" src="https://github.com/user-attachments/assets/70493fc6-9cfd-4094-81cb-11e98ec220a9" />

---
###* MAE and MSE training curves
<img width="1920" height="1020" alt="Screenshot 2025-12-29 194224" src="https://github.com/user-attachments/assets/f2c00a46-17d9-44b8-add3-cdc5bbf82a15" />

---
###* Model summary and training logs
<img width="1920" height="1020" alt="Screenshot 2025-12-29 194404" src="https://github.com/user-attachments/assets/f499e7b6-8e49-45e1-82a4-b941a16f66fd" />

---

✨ *This project is ideal for beginners learning TensorFlow regression and real-world ML workflows.*
