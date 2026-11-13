"""
Real-World Network Intrusion Detection System (NIDS)
Core Machine Learning Module: Ensemble Learning on NSL-KDD
Author: Ameenul Lumin Fathima M.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NIDS-Core")

# Standard 41 feature names for NSL-KDD
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERICAL_FEATURES = [col for col in FEATURE_NAMES if col not in CATEGORICAL_FEATURES]

# Attack mappings to broad categories
ATTACK_CATEGORIES: Dict[str, str] = {
    "normal": "Normal",
    "neptune": "DoS",
    "smurf": "DoS",
    "pod": "DoS",
    "teardrop": "DoS",
    "land": "DoS",
    "back": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    "ddos_syn": "DDoS",
    "ddos_udp": "DDoS",
    "ddos_http": "DDoS",
    "ddos": "DDoS",
    "ipsweep": "Port Scanning",
    "portsweep": "Port Scanning",
    "nmap": "Port Scanning",
    "satan": "Port Scanning",
    "mscan": "Port Scanning",
    "saint": "Port Scanning",
    "guess_passwd": "Brute Force",
    "ftp_write": "Brute Force",
    "imap": "Brute Force",
    "phf": "Brute Force",
    "multihop": "Brute Force",
    "warezmaster": "Brute Force",
    "warezclient": "Brute Force",
    "spy": "Brute Force",
    "xlock": "Brute Force",
    "xsnoop": "Brute Force",
    "snmpguess": "Brute Force",
    "snmpgetattack": "Brute Force",
    "sendmail": "Brute Force",
    "named": "Brute Force",
    "buffer_overflow": "Buffer Overflow",
    "loadmodule": "Buffer Overflow",
    "rootkit": "Buffer Overflow",
    "perl": "Buffer Overflow",
    "sqlattack": "Buffer Overflow",
    "xterm": "Buffer Overflow",
    "ps": "Buffer Overflow",
    "backdoor": "Backdoor",
    "worm": "Backdoor",
}

SEVERITY_LEVELS: Dict[str, str] = {
    "Normal": "INFO",
    "Port Scanning": "MEDIUM",
    "DoS": "HIGH",
    "DDoS": "CRITICAL",
    "Brute Force": "HIGH",
    "Buffer Overflow": "CRITICAL",
    "Backdoor": "CRITICAL",
}


class NIDSModel:
    """
    Ensemble Machine Learning Model for Real-World Network Intrusion Detection.
    Combines Random Forest and Decision Tree classifiers with soft voting.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "ensemble_nids.joblib")
        self.scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        self.meta_path = os.path.join(self.model_dir, "metadata.joblib")

        self.scaler = StandardScaler()
        self.model: Optional[VotingClassifier] = None
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.target_classes: List[str] = [
            "Normal", "Port Scanning", "DoS", "DDoS",
            "Brute Force", "Buffer Overflow", "Backdoor"
        ]
        self.is_trained = False

    def _encode_categories(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features into deterministic integer representations."""
        df_encoded = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col not in df_encoded.columns:
                df_encoded[col] = "other"
            df_encoded[col] = df_encoded[col].astype(str).str.lower().str.strip()

            if fit:
                unique_vals = sorted(list(df_encoded[col].unique()))
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                mapping["unknown"] = len(mapping)
                self.cat_mappings[col] = mapping

            mapping = self.cat_mappings.get(col, {})
            unknown_code = mapping.get("unknown", len(mapping))
            df_encoded[col] = df_encoded[col].map(lambda x: mapping.get(x, unknown_code))

        return df_encoded

    def preprocess(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocesses incoming dataframe matching NSL-KDD schema into model-ready array."""
        df_clean = df.copy()

        # Fill missing features with default 0 / unknown
        for col in FEATURE_NAMES:
            if col not in df_clean.columns:
                if col in CATEGORICAL_FEATURES:
                    df_clean[col] = "other"
                else:
                    df_clean[col] = 0.0

        # Encode categorical variables
        df_clean = self._encode_categories(df_clean[FEATURE_NAMES], fit=fit)

        # Ensure correct numeric types
        for col in FEATURE_NAMES:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)

        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(df_clean[FEATURE_NAMES].values)
        else:
            X_scaled = self.scaler.transform(df_clean[FEATURE_NAMES].values)

        return X_scaled

    def build_ensemble(self) -> VotingClassifier:
        """Constructs the Random Forest + Decision Tree ensemble with soft voting."""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        dt = DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=4,
            random_state=42
        )
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("dt", dt)],
            voting="soft"
        )
        return ensemble

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Trains the ensemble model and computes performance evaluation metrics."""
        logger.info("Initializing Ensemble NIDS Training (Random Forest + Decision Tree)...")
        self.model = self.build_ensemble()
        self.model.fit(X, y)
        self.is_trained = True

        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4)
        }
        logger.info(f"Training completed successfully. Metrics: {metrics}")
        return metrics

    def save(self) -> None:
        """Serializes model, scaler, and categorical mappings to disk."""
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save untrained model.")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump({
            "cat_mappings": self.cat_mappings,
            "target_classes": self.target_classes,
            "feature_names": FEATURE_NAMES
        }, self.meta_path)
        logger.info(f"Model and artifacts persisted to {self.model_dir}")

    def load(self) -> bool:
        """Loads serialized model artifacts if available."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.meta_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            meta = joblib.load(self.meta_path)
            self.cat_mappings = meta.get("cat_mappings", {})
            self.target_classes = meta.get("target_classes", self.target_classes)
            self.is_trained = True
            logger.info("Model artifacts loaded successfully.")
            return True
        return False

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on provided traffic dataframe.
        Returns predicted attack category labels and confidence probabilities.
        """
        if not self.is_trained:
            if not self.load():
                raise RuntimeError("NIDS model is not trained and no saved weights found.")

        X = self.preprocess(df, fit=False)
        preds = self.model.predict(X)
        probs = np.max(self.model.predict_proba(X), axis=1)
        return preds, probs

    def predict_detailed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches incoming traffic dataframe with detection results, severity level,
        and alert status.
        """
        preds, probs = self.predict(df)
        results = df.copy()
        results["predicted_label"] = preds
        results["confidence"] = np.round(probs, 4)
        results["is_intrusion"] = results["predicted_label"] != "Normal"
        results["severity"] = results["predicted_label"].map(lambda x: SEVERITY_LEVELS.get(x, "INFO"))
        return results


def generate_benchmark_dataset(num_samples: int = 6000) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a realistic benchmark dataset modeled after the statistical distributions
    of the NSL-KDD dataset for offline training and evaluation.
    """
    logger.info(f"Generating realistic NSL-KDD benchmark dataset with {num_samples} samples...")
    np.random.seed(42)

    categories = ["Normal", "Port Scanning", "DoS", "DDoS", "Brute Force", "Buffer Overflow", "Backdoor"]
    weights = [0.52, 0.12, 0.16, 0.08, 0.06, 0.03, 0.03]
    labels = np.random.choice(categories, size=num_samples, p=weights)

    data = {col: np.zeros(num_samples, dtype=float) for col in FEATURE_NAMES}
    data["protocol_type"] = ["tcp"] * num_samples
    data["service"] = ["http"] * num_samples
    data["flag"] = ["SF"] * num_samples

    for i, label in enumerate(labels):
        if label == "Normal":
            data["duration"][i] = np.random.exponential(1.5)
            data["protocol_type"][i] = np.random.choice(["tcp", "udp", "icmp"], p=[0.75, 0.20, 0.05])
            data["service"][i] = np.random.choice(["http", "smtp", "domain_u", "ftp_data", "private"], p=[0.5, 0.2, 0.15, 0.1, 0.05])
            data["flag"][i] = "SF"
            data["src_bytes"][i] = np.random.randint(100, 3500)
            data["dst_bytes"][i] = np.random.randint(200, 15000)
            data["logged_in"][i] = 1.0 if data["protocol_type"][i] == "tcp" else 0.0
            data["count"][i] = np.random.randint(1, 20)
            data["srv_count"][i] = np.random.randint(1, 20)
            data["same_srv_rate"][i] = np.random.uniform(0.85, 1.0)
            data["diff_srv_rate"][i] = np.random.uniform(0.0, 0.15)
            data["dst_host_count"][i] = np.random.randint(1, 255)
            data["dst_host_srv_count"][i] = np.random.randint(1, 255)
            data["dst_host_same_srv_rate"][i] = np.random.uniform(0.8, 1.0)
            data["dst_host_diff_srv_rate"][i] = np.random.uniform(0.0, 0.1)

        elif label == "DoS":
            data["duration"][i] = 0.0
            data["protocol_type"][i] = np.random.choice(["tcp", "icmp"], p=[0.7, 0.3])
            data["service"][i] = np.random.choice(["private", "eco_i", "http"], p=[0.6, 0.3, 0.1])
            data["flag"][i] = np.random.choice(["S0", "REJ", "SF"], p=[0.7, 0.2, 0.1])
            data["src_bytes"][i] = np.random.choice([0, 1032, 42])
            data["dst_bytes"][i] = 0
            data["logged_in"][i] = 0.0
            data["count"][i] = np.random.randint(150, 511)
            data["srv_count"][i] = np.random.randint(100, 511)
            data["serror_rate"][i] = np.random.uniform(0.8, 1.0)
            data["srv_serror_rate"][i] = np.random.uniform(0.8, 1.0)
            data["same_srv_rate"][i] = np.random.uniform(0.7, 1.0)
            data["dst_host_count"][i] = 255
            data["dst_host_srv_count"][i] = np.random.randint(1, 50)
            data["dst_host_serror_rate"][i] = np.random.uniform(0.8, 1.0)

        elif label == "DDoS":
            data["duration"][i] = 0.0
            data["protocol_type"][i] = "udp"
            data["service"][i] = np.random.choice(["private", "other", "domain_u"])
            data["flag"][i] = "SF"
            data["src_bytes"][i] = np.random.randint(1200, 4000)
            data["dst_bytes"][i] = 0
            data["count"][i] = np.random.randint(350, 512)
            data["srv_count"][i] = np.random.randint(350, 512)
            data["same_srv_rate"][i] = 1.0
            data["diff_srv_rate"][i] = 0.0
            data["dst_host_count"][i] = 255
            data["dst_host_srv_count"][i] = 255
            data["dst_host_same_src_port_rate"][i] = np.random.uniform(0.7, 1.0)

        elif label == "Port Scanning":
            data["duration"][i] = np.random.exponential(0.2)
            data["protocol_type"][i] = np.random.choice(["tcp", "icmp"], p=[0.8, 0.2])
            data["service"][i] = np.random.choice(["private", "eco_i", "other", "ftp"])
            data["flag"][i] = np.random.choice(["REJ", "RSTO", "S0"], p=[0.5, 0.3, 0.2])
            data["src_bytes"][i] = np.random.randint(0, 100)
            data["dst_bytes"][i] = 0
            data["logged_in"][i] = 0.0
            data["count"][i] = np.random.randint(50, 300)
            data["srv_count"][i] = np.random.randint(1, 10)
            data["diff_srv_rate"][i] = np.random.uniform(0.6, 1.0)
            data["dst_host_count"][i] = np.random.randint(150, 255)
            data["dst_host_diff_srv_rate"][i] = np.random.uniform(0.5, 1.0)

        elif label == "Brute Force":
            data["duration"][i] = np.random.randint(1, 10)
            data["protocol_type"][i] = "tcp"
            data["service"][i] = np.random.choice(["ftp", "telnet", "smtp", "pop_3"])
            data["flag"][i] = "SF"
            data["src_bytes"][i] = np.random.randint(100, 800)
            data["dst_bytes"][i] = np.random.randint(200, 1200)
            data["num_failed_logins"][i] = np.random.randint(3, 10)
            data["logged_in"][i] = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
            data["count"][i] = np.random.randint(5, 30)
            data["same_srv_rate"][i] = np.random.uniform(0.8, 1.0)

        elif label == "Buffer Overflow":
            data["duration"][i] = np.random.randint(5, 30)
            data["protocol_type"][i] = "tcp"
            data["service"][i] = np.random.choice(["telnet", "ftp", "other"])
            data["flag"][i] = "SF"
            data["src_bytes"][i] = np.random.randint(2000, 10000)
            data["dst_bytes"][i] = np.random.randint(1000, 5000)
            data["root_shell"][i] = 1.0
            data["num_root"][i] = np.random.randint(1, 5)
            data["num_file_creations"][i] = np.random.randint(1, 4)
            data["logged_in"][i] = 1.0

        elif label == "Backdoor":
            data["duration"][i] = np.random.randint(10, 120)
            data["protocol_type"][i] = "tcp"
            data["service"][i] = "private"
            data["flag"][i] = "SF"
            data["src_bytes"][i] = np.random.randint(50, 500)
            data["dst_bytes"][i] = np.random.randint(50, 500)
            data["num_shells"][i] = 1.0
            data["logged_in"][i] = 1.0
            data["count"][i] = np.random.randint(1, 5)

    df = pd.DataFrame(data)
    return df, labels
