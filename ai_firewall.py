import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scapy.all import sniff, IP
import numpy as np
import pickle

# -----------------------
# 1. معالجة البيانات
# -----------------------
def preprocess_data(input_file="network_traffic_labeled.csv"):
    data = pd.read_csv(input_file)
    data['Source'] = data['Source'].apply(lambda x: hash(x) % 10000)
    data['Destination'] = data['Destination'].apply(lambda x: hash(x) % 10000)
    encoder = LabelEncoder()
    data['Protocol'] = encoder.fit_transform(data['Protocol'])
    data['Label'] = encoder.fit_transform(data['Label'])
    X = data[['Source', 'Destination', 'Protocol', 'PacketSize']]
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, X_test, y_train_balanced, y_test

# -----------------------
# 2. بناء النموذج وتحسينه
# -----------------------
def build_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    with open("traffic_model.pkl", "wb") as file:
        pickle.dump(model, file)
    return model

# -----------------------
# 3. تقييم النموذج
# -----------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

# -----------------------
# 4. تحليل حركة المرور الحية
# -----------------------
def live_traffic_analysis(model):
    def classify_packet(packet):
        if IP in packet:
            ip_src = hash(packet[IP].src) % 10000
            ip_dst = hash(packet[IP].dst) % 10000
            protocol = packet[IP].proto
            packet_size = len(packet)
            input_data = pd.DataFrame([[ip_src, ip_dst, protocol, packet_size]], 
                                      columns=["Source", "Destination", "Protocol", "PacketSize"])
            prediction = model.predict(input_data)
            label = "Normal" if prediction[0] == 0 else "Suspicious"
            print(f"Packet classified as: {label}")
    sniff(prn=classify_packet, count=50)

# -----------------------
# التشغيل الرئيسي
# -----------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    model = build_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    live_traffic_analysis(model)
