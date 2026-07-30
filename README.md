# 🛡️ Real-World Network Intrusion Detection System (NIDS)

A production-ready Machine Learning-based Network Intrusion Detection System (NIDS) that detects cyberattacks in real time using ensemble learning techniques. The system analyzes network traffic, identifies malicious activities, generates automated alerts, and provides detailed monitoring reports.

---

## 📌 Project Overview

This project leverages Machine Learning algorithms to classify network traffic as either normal or malicious. It supports real-time monitoring, automated threat detection, attack simulation, and comprehensive logging, making it suitable for cybersecurity research, learning, and demonstration purposes.

---

## ✨ Features

- 🔍 Real-time Network Traffic Monitoring
- 🤖 Machine Learning-based Threat Detection
- 🌲 Random Forest + Decision Tree Ensemble
- 🚨 Automated Severity-based Alerts
- 📊 Performance Monitoring & Reports
- 📝 JSON-based Logging
- ⚡ Attack Traffic Simulation
- 📈 Prediction Result Export

---

## 🚨 Supported Attack Types

- Port Scanning
- DoS (Denial of Service)
- DDoS (Distributed Denial of Service)
- Brute Force Attacks
- Buffer Overflow
- Backdoor Access
- Normal Network Traffic

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Scikit-learn | Machine Learning |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Joblib | Model Serialization |
| JSON | Logging |
| NSL-KDD Dataset | Training Dataset |

---

## 📂 Project Structure

```
Real-World-NIDS/
│
├── code/
│   └── nids.py
│
├── logs/
│
├── models/
│
├── results/
│
├── network_traffic_generator.py
├── nids_monitor.py
├── deploy.py
├── test_system.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/real-world-network-intrusion-detection-system.git
cd real-world-network-intrusion-detection-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train Models & Start Monitoring

```bash
python deploy.py
```

### Monitor Custom Traffic

```bash
python deploy.py --custom-file your_traffic.csv
```

### Run System Tests

```bash
python test_system.py
```

---

## 📊 Performance

| Metric | Score |
|---------|-------|
| Accuracy | 77–78% |
| Precision | 96–97% |
| Recall | 63–64% |

---

## 📁 Output Files

```
logs/
    alerts_*.json
    report_*.json

results/
    real_world_predictions.csv
```

---

## 📚 Dataset

The project is trained using the **NSL-KDD Dataset**, a benchmark dataset for intrusion detection research.

### Dataset Statistics

- 43 Features
- 125,000+ Training Records
- 22 Attack Categories

Dataset:
https://www.unb.ca/cic/datasets/nsl-kdd.html

---

## 🔒 Security Features

- Real-time Threat Detection
- Automated Alert Generation
- Attack Classification
- Network Traffic Analysis
- Event Logging
- Performance Reporting

---

## 🚀 Future Enhancements

- Deep Learning Models (LSTM/CNN)
- Live Packet Capture using Scapy
- Web Dashboard
- Email & SMS Alerts
- SIEM Integration
- Docker Deployment
- Cloud Monitoring Support
- REST API

---

## 📖 Learning Outcomes

- Machine Learning for Cybersecurity
- Network Traffic Analysis
- Intrusion Detection Systems
- Ensemble Learning
- Feature Engineering
- Threat Detection
- Real-time Monitoring
- Security Event Logging

---

## 👨‍💻 Author

**Ameenul Lumin**

Software Developer

### Areas of Interest

- Cybersecurity
- Machine Learning
- Java Backend Development
- Computer Vision
- Web Development

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub!

---

## 📄 License

This project is licensed under the **MIT License**.
