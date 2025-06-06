# Logistics Route Predictor

This project is a machine learning-based solution for predicting the **cost** and **duration** of delivery routes based on a variety of real-world factors.

## 🔍 Project Description

- Predicts delivery route **cost** (in EUR) and **travel time** (in hours)
- Uses a **neural network** built with TensorFlow/Keras
- Includes a **Flask REST API** for easy integration with external systems
- Stores and retrieves user and API key information using **MySQL**
- Trained model is saved as `model.h5`

## 📁 Project Structure

```
logistics-route-predictor/
│
├── app.py                 # Flask API logic
├── model.h5               # Trained neural network model
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── mysql/                 # MySQL scripts (create_tables.sql, seed_data.sql, etc.)
└── utils/                 # (Optional) helper functions
```

## 🚀 How to Run

1. **Clone the repository**  
```bash
git clone https://github.com/YOUR_USERNAME/logistics-route-predictor.git
cd logistics-route-predictor
```

2. **Install requirements**  
```bash
pip install -r requirements.txt
```

3. **Set up MySQL database**  
- Create database and run the SQL scripts in `mysql/`
- Update connection credentials in `app.py`

4. **Run Flask API**  
```bash
python app.py
```

## 🧠 Model Details

- Framework: **TensorFlow/Keras**
- Loss Function: **MSE**
- Optimizer: **Adam**
- Regularization: **L2**, **Dropout**
- Early Stopping to prevent overfitting

## 🔐 API Endpoints

- `POST /predict` – Accepts route factors and returns predicted cost/time
- `POST /register` – Register a new user (email, password, company)
- `POST /login` – Login and get access token
- `GET /validate` – API key validation

## 🛠 Technologies Used

- Python, TensorFlow, Keras, Flask, NumPy, scikit-learn, MySQL

## 👩‍💻 Author

Ivana Kostić  
[LinkedIn](https://www.linkedin.com/in/ivana-kostic-55a34a1b2/)  

