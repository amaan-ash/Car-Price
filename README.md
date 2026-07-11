# 🚗 AutoWorth AI

### Machine Learning-Based Used Car Price Prediction System

AutoWorth AI is a Machine Learning-based web application developed to estimate the selling price of used cars based on selected vehicle characteristics.

The project demonstrates the complete Machine Learning workflow, including dataset inspection, data preprocessing, feature engineering, categorical encoding, model training, regression evaluation, model serialization, and integration with an interactive Streamlit application.

The final prediction model uses a **Random Forest Regressor** trained on historical used car data.

---

## 📌 Project Overview

Estimating the selling price of a used car can be difficult because vehicle value depends on several factors such as age, brand, kilometers driven, fuel type, transmission, ownership history, and seller type.

Traditional valuation methods may depend on manual comparison, dealer estimates, or individual judgment.

AutoWorth AI applies Machine Learning to historical vehicle data and provides a data-driven estimated selling price based on user-provided vehicle information.

> **Note:** The predicted value is an ML-based estimate derived from historical data and should not be considered a guaranteed real-world market price.

---

## ✨ Key Features

- 🤖 Machine Learning-based car price prediction
- 🌲 Random Forest Regression model
- 🚘 Brand-level vehicle valuation
- 🧠 Feature engineering for vehicle age and brand
- 🔢 Categorical feature encoding
- 💾 Serialized ML model and encoders using Joblib
- 🖥️ Interactive Streamlit web application
- 📊 Interactive Plotly analytics
- 📈 Dataset insights and visualizations
- 🎨 Modern dark-themed user interface
- 🧭 Custom collapsible navigation panel
- ⚡ Fast prediction workflow
- 📱 Responsive application layout

---

## 🧠 Machine Learning Model

The final model used in AutoWorth AI is:

### Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees and averages their predictions.

It was selected because it is suitable for tabular data and can model nonlinear relationships between vehicle characteristics and selling price.

### Why Random Forest?

- Handles nonlinear relationships
- Captures interactions between multiple features
- Suitable for structured tabular datasets
- More robust than a single Decision Tree
- Does not require complex feature scaling
- Effective for regression problems

---

## 📊 Model Performance

The model is evaluated using regression metrics.

| Metric | Result |
|---|---:|
| R² Score | `0.7004` |
| Mean Absolute Error (MAE) | `₹127,126.08` |
| Root Mean Squared Error (RMSE) | `₹302,360.01` |

### Performance Interpretation

The model achieved an **R² Score of approximately 0.70**, indicating that it explains around **70% of the variance in used car selling prices** based on the available input features.

The model is evaluated as a **regression model**, so the R² Score should not be interpreted as classification accuracy.

---

## 📂 Dataset

The project uses a **CarDekho used car dataset** containing:

- **4,340 vehicle records**
- **8 original columns**

### Original Dataset Features

| Feature | Description |
|---|---|
| `name` | Complete vehicle name |
| `year` | Manufacturing year |
| `selling_price` | Vehicle selling price |
| `km_driven` | Total kilometers driven |
| `fuel` | Fuel type |
| `seller_type` | Type of seller |
| `transmission` | Transmission type |
| `owner` | Ownership category |

### Target Variable

```text
selling_price
```

The target variable is continuous numerical data, making the problem a **supervised Machine Learning regression task**.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline prepares the raw dataset before model training.

### 1. Dataset Inspection

The dataset is inspected for:

- Shape
- Column names
- Data types
- Missing values
- Dataset information
- Unique categorical values

The inspected dataset contains **no missing values in the original eight columns**.

### 2. Brand Extraction

The vehicle brand is extracted from the original `name` column.

Example:

```text
Maruti Swift Dzire VDI
        ↓
Brand = Maruti
```

### 3. Car Age Feature Engineering

A new `car_age` feature is derived from the manufacturing year.

This allows the Machine Learning model to directly learn the relationship between vehicle age and selling price.

### 4. Categorical Encoding

Categorical vehicle features are converted into numerical representations using Label Encoding.

Encoded features include:

- Brand
- Fuel Type
- Seller Type
- Transmission
- Owner

The fitted encoders are saved and reused during prediction to maintain consistency between model training and the Streamlit application.

### 5. Train-Test Split

The processed dataset is separated into training and testing data.

The training dataset is used to train the Random Forest Regressor, while the testing dataset is used for model evaluation.

---

## 🔄 Machine Learning Workflow

```text
CarDekho Dataset
        ↓
Dataset Inspection
        ↓
Data Preprocessing
        ↓
Brand Extraction
        ↓
Car Age Feature Engineering
        ↓
Categorical Encoding
        ↓
Feature and Target Separation
        ↓
Train-Test Split
        ↓
Random Forest Regressor
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Serialization
        ↓
Streamlit Integration
        ↓
Used Car Price Prediction
```

---

## 🏗️ System Architecture

AutoWorth AI consists of two primary pipelines.

### Offline Model Training Pipeline

```text
Dataset
   ↓
Pandas Data Processing
   ↓
Feature Engineering
   ↓
Label Encoding
   ↓
Train-Test Split
   ↓
Random Forest Training
   ↓
Model Evaluation
   ↓
Model + Encoders + Metadata
```

### Application Prediction Pipeline

```text
User
   ↓
Streamlit Interface
   ↓
Vehicle Input Form
   ↓
Input Validation
   ↓
Feature Preparation
   ↓
Saved Encoders
   ↓
Random Forest Regressor
   ↓
Predicted Selling Price
   ↓
Interactive Result Display
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning model and preprocessing |
| Plotly | Interactive data visualizations |
| Joblib | Model and encoder serialization |
| Git | Version control |
| GitHub | Source code repository |
| VS Code | Development environment |

---

## 📁 Project Structure

```text
AutoWorthAI/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── car_details.csv
│
└── model/
    ├── car_price_model.pkl
    ├── encoders.pkl
    └── metadata.pkl
```

> The exact generated model files are created or updated through the model training pipeline.

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd AutoWorthAI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Machine Learning Model

```bash
python train_model.py
```

The training pipeline preprocesses the dataset, trains the Random Forest Regressor, evaluates the model, and saves the required model files.

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 🖥️ Application Modules

### 🏠 Home

The Home page provides an overview of AutoWorth AI and the Machine Learning-based valuation workflow.

### 🔮 Predict Price

Users can enter vehicle information and generate an estimated selling price using the trained Random Forest Regressor.

The prediction uses features such as:

- Brand
- Vehicle Age
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission
- Ownership

### 📊 Analytics

The Analytics section provides interactive dataset insights and visualizations using Plotly.

It helps users explore patterns and characteristics present in the used car dataset.

### ℹ️ About

The About page provides project information, Machine Learning workflow details, model information, and project context.

---

## ⚠️ Current Limitations

The current version of AutoWorth AI has several limitations:

- Prediction is performed at the **brand level**
- Exact car model is not directly used
- Vehicle variant is not included
- The dataset contains historical and static data
- Live automotive market prices are not integrated
- Vehicle physical condition is not available
- Accident history is not included
- Service history is not included
- Vehicle location or city is not considered
- Engine power is not included
- Detailed mileage specifications are not available

These limitations affect the level of vehicle-specific price estimation that the model can provide.

---

## 🔮 Future Scope

Future versions of AutoWorth AI may include:

- 🚘 Exact car model prediction
- 🏎️ Variant-level valuation
- ⚙️ Engine and power specifications
- ⛽ Detailed mileage features
- 🔧 Vehicle condition analysis
- 📋 Accident and service history
- 📍 Location-based pricing
- 🌐 Live automotive market data integration
- ☁️ Cloud deployment
- 📱 Mobile application
- 📷 Image-based vehicle recognition
- 🤖 AI-based car valuation assistant
- 🔐 User authentication
- 🕘 Prediction history

A major future improvement is transitioning from:

```text
Current Version
Brand-Level Prediction
```

to:

```text
Future Version
Brand + Model + Variant-Level Prediction
```

Adding richer vehicle specifications may further improve prediction performance.

---

## 🌍 Sustainable Development Goals

### SDG 9 — Industry, Innovation and Infrastructure

AutoWorth AI demonstrates the practical use of Machine Learning and software-based automation for automotive price estimation.

### SDG 12 — Responsible Consumption and Production

Improved information during vehicle resale may support more informed decisions regarding the reuse and resale of existing vehicles.

The project does not claim direct environmental impact but demonstrates how data-driven systems can support informed consumption decisions.

---

## 🎓 Academic Context

AutoWorth AI was developed as a **Machine Learning Micro Project** for the Department of Computer Engineering.

**Institution:** Shah & Anchor Kutchhi Engineering College  
**Department:** Computer Engineering  
**Subject:** Machine Learning  
**Academic Year:** 2025–2026

---

## 👨‍💻 Author

### Amaan Ali Shaikh

**PRN:** 125BTCM2005  
**Department:** Computer Engineering

---

## 📄 License

This project has been developed for academic and educational purposes.

The source code may be studied and referenced for learning purposes. Please provide appropriate credit when using substantial parts of the project.

---

## ⭐ Project Status

```text
Model Training       ✅ Complete
Data Preprocessing   ✅ Complete
Model Evaluation     ✅ Complete
Streamlit UI         ✅ Complete
Analytics Dashboard  ✅ Complete
Documentation        ✅ Complete
```

---

<p align="center">
  <b>AutoWorth AI</b><br>
  Machine Learning-Based Used Car Price Prediction
</p>

<p align="center">
  Developed by Amaan Ali Shaikh
</p>
