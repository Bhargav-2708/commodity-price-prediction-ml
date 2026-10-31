# 🌾 Commodity Price Prediction in Indian Agricultural Markets

A Machine Learning project that estimates the modal market price of agricultural commodities in Indian Mandis using **Random Forest Regression**.

## 📌 Project Overview

Agricultural commodity prices vary across states, districts, markets, varieties and grades. Understanding these price differences can help farmers, traders and market participants analyze agricultural markets.

This project uses Python and Machine Learning to estimate the modal market price of commodities based on their market location and commodity characteristics.

## 🎯 Project Objectives

- Analyze agricultural commodity prices across Indian markets.
- Understand price variations across states and districts.
- Perform Exploratory Data Analysis (EDA).
- Build a Machine Learning model to estimate modal market prices.
- Evaluate model performance using regression metrics.

## 📂 Dataset Information

**Dataset Name:** Current Daily Price of Various Commodities from Various Markets (Mandi)

**Source:** Government of India's Open Government Data Platform.

**Dataset Size:** Approximately 17,825 records.

**Features:**

| Column | Description |
|---|---|
| State | State where the market is located |
| District | District name |
| Market | Agricultural market name |
| Commodity | Agricultural product |
| Variety | Commodity variety |
| Grade | Commodity quality grade |
| Arrival_Date | Reported arrival date |
| Min Price | Minimum market price |
| Max Price | Maximum market price |
| Modal Price | Modal market price |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Random Forest Regression

## ⚙️ Project Workflow

1. Dataset collection
2. Data loading using Pandas
3. Data cleaning and preprocessing
4. Exploratory Data Analysis
5. Data visualization
6. Categorical data encoding using One-Hot Encoding
7. Training and testing split (80:20)
8. Random Forest Regression model training
9. Modal price estimation
10. Model evaluation

## 🤖 Machine Learning Algorithm

**Random Forest Regressor**

Random Forest is a supervised Machine Learning algorithm that combines multiple decision trees to generate numerical predictions.

In this project, it estimates the modal market price using:

- State
- District
- Market
- Commodity
- Variety
- Grade

**Prediction Target:** Modal Market Price.

## 📊 Model Evaluation

The model is evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving more weight to larger errors.
- **R² Score:** Measures how well the model explains variation in modal prices.

Actual evaluation results will depend on the model execution.

## 📈 Visualizations

The project includes:

- Distribution of Modal Market Prices
- Top 10 Commodities by Average Modal Price
- Actual vs Predicted Commodity Prices

## ▶️ How to Run the Project

**Step 1: Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/commodity-price-prediction-ml.git
```

**Step 2: Install required libraries**

```bash
pip install pandas numpy matplotlib scikit-learn
```

**Step 3: Run the Python program**

```bash
python commodity_price_prediction.py
```

Make sure the dataset CSV is available at the file path specified in the Python code.

## 📁 Project Structure

```text
commodity-price-prediction-ml/
│
├── commodity_price_prediction.py
├── mandi_prices_cleaned.csv
├── commodity_price_predictions.csv
├── README.md
│
└── project_screenshots/
```

## ⚠️ Project Limitations

- The dataset contains commodity prices from a single actual reporting date.
- Dates added during dataset preparation were simulated and are not genuine historical observations.
- The model estimates modal prices based on available market and commodity details.
- It does not currently forecast future commodity prices.

## 🔮 Future Scope

- Use genuine historical commodity price data from multiple dates.
- Develop future commodity price forecasting.
- Compare additional Machine Learning algorithms.
- Develop a web application for commodity price estimation.
- Expand the analysis across more agricultural markets.

## 👨‍💻 Author

**Bhargav**

B.Tech – Computer Science and Engineering

Lovely Professional University

## 📜 License

This project is intended for educational and academic purposes.
