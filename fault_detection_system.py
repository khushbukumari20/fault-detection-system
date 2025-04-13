import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import random
import time
from pymongo import MongoClient
from flask import Flask, jsonify
import threading

# ---------------------------
# MongoDB Setup
# ---------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client['smart_home']
collection = db['sensor_data']

def insert_sensor_data(data):
    collection.insert_one(data)

def get_recent_data():
    return list(collection.find().sort('_id', -1).limit(1))[0]

# ---------------------------
# Simulate Sensor Data
# ---------------------------
def generate_sensor_data():
    return {
        "temperature": random.uniform(15, 45),
        "humidity": random.uniform(20, 80),
        "motion": random.choice([0, 1])
    }

def simulate_sensors():
    while True:
        data = generate_sensor_data()
        insert_sensor_data(data)
        print(f"Inserted data: {data}")
        time.sleep(5)

# ---------------------------
# Train and Save Model
# ---------------------------
def train_and_save_model():
    # Simulate or load dataset
    data = pd.DataFrame({
        "temperature": np.random.uniform(15, 45, 500),
        "humidity": np.random.uniform(20, 80, 500),
        "motion": np.random.choice([0, 1], 500),
    })
    # Randomly label ~20% as anomalies
    data['label'] = np.random.choice([0, 1], size=500, p=[0.8, 0.2])

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
    print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
    print(f"SVM Accuracy: {svm_acc * 100:.2f}%")

    best_model = rf_model if rf_acc > svm_acc else svm_model
    joblib.dump(best_model, 'anomaly_model.pkl')

# ---------------------------
# Flask API
# ---------------------------
app = Flask(__name__)
model = None

@app.route('/predict', methods=['GET'])
def predict():
    data = get_recent_data()
    df = pd.DataFrame([data])
    df = df[['temperature', 'humidity', 'motion']]
    prediction = model.predict(df)[0]
    status = 'Anomaly' if prediction == 1 else 'Normal'
    return jsonify({
        "data": data,
        "status": status
    })

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '_main_':
    train_and_save_model()
    model = joblib.load('anomaly_model.pkl')
    
    # Start sensor simulation in background
    sensor_thread = threading.Thread(target=simulate_sensors)
    sensor_thread.daemon = True
    sensor_thread.start()

    # Run Flask app
    app.run(debug=True)