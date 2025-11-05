
# 🚗 Electric Vehicle (EV) Price Prediction using Machine Learning

### 🧠 **Overview**

This project predicts the **price of Electric Vehicles (EVs)** based on features such as battery capacity, range, charging time, and brand.
The model uses **Machine Learning (Regression)** to estimate the price, and a **Streamlit web app** provides an easy-to-use interface for users.

---

## 📋 **Project Structure**

```
EV_Price_Prediction/
│
├── data/
│   └── ev_sample.csv              # Dataset (sample data of EVs)
│
├── model/
│   └── ev_price_model.pkl         # Trained ML model (generated after training)
│
├── app.py                         # Streamlit web app for predictions
├── train_model.py                 # Script to train and save the model
├── requirements.txt               # Required Python libraries
└── README.md                      # Project documentation
```

---

## ⚙️ **How It Works**

1. The dataset (`ev_sample.csv`) contains basic EV features:

   * **Brand**
   * **Battery Capacity (kWh)**
   * **Range per Charge (km)**
   * **Charging Time (hours)**
   * **Price (₹ Lakh)**

2. `train_model.py`:

   * Loads and preprocesses the dataset
   * Trains a **RandomForestRegressor** model
   * Saves the model as `ev_price_model.pkl`

3. `app.py`:

   * Provides a **Streamlit interface**
   * Takes user inputs (battery, range, brand, etc.)
   * Loads the trained model and predicts the EV price

---

## 🧑‍💻 **Tech Stack**

* **Python 3.10+**
* **Pandas** → Data handling
* **Scikit-learn** → Model training
* **Streamlit** → Front-end web app
* **Pickle** → Model serialization

---

## 🚀 **How to Run the Project**

### 1️⃣ Clone this repository

```bash
git clone https://github.com/<your-username>/EV_Price_Prediction.git
cd EV_Price_Prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model (if not trained)

```bash
python train_model.py
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser and go to:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧾 **Sample Input**

| Brand | Battery (kWh) | Range (km) | Charging Time (hr) | Output Price (₹ Lakh) |
| ----- | ------------- | ---------- | ------------------ | --------------------- |
| Tata  | 30            | 250        | 6                  | 12                    |
| MG    | 40            | 320        | 5                  | 18                    |
| Tesla | 75            | 500        | 1.5                | 60                    |

---

## 🎨 **Features**

✅ Predict EV price using ML
✅ Clean Streamlit interface
✅ Customizable dataset
✅ Model retraining supported
✅ Beginner-friendly project structure

---

## 📈 **Future Improvements**

* Add more features (e.g., motor power, torque, year)
* Use deep learning models
* Deploy on Streamlit Cloud / Render
* Integrate OpenAI for interactive chat support

---

## 🧑‍🏫 **Author**

**Sai Thumu**
📧 [23U11A0442@gmail.com]
🎓 Developed as part of a Machine Learning practice project

---
