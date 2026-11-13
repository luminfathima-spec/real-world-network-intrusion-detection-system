"""
Real-World Network Intrusion Detection System (NIDS)
Comprehensive System & Unit Verification Suite
Author: Ameenul Lumin Fathima M.
"""

import os
import unittest
import shutil
import tempfile
import pandas as pd
from code.nids import NIDSModel, generate_benchmark_dataset, FEATURE_NAMES, SEVERITY_LEVELS
from network_traffic_generator import NetworkTrafficGenerator
from nids_monitor import NIDSMonitor


class TestNIDSSystem(unittest.TestCase):
    """Automated test cases for NIDS components."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="nids_test_")
        cls.models_dir = os.path.join(cls.test_dir, "models")
        cls.logs_dir = os.path.join(cls.test_dir, "logs")
        cls.results_dir = os.path.join(cls.test_dir, "results")
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.logs_dir, exist_ok=True)
        os.makedirs(cls.results_dir, exist_ok=True)

        # Train a fast test model
        cls.model = NIDSModel(model_dir=cls.models_dir)
        cls.df_train, cls.labels = generate_benchmark_dataset(num_samples=1000)
        X_train = cls.model.preprocess(cls.df_train, fit=True)
        cls.metrics = cls.model.train(X_train, cls.labels)
        cls.model.save()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_benchmark_dataset_generation(self):
        """Verify the synthetic benchmark dataset structure and columns."""
        df, labels = generate_benchmark_dataset(num_samples=100)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(labels), 100)
        for feature in FEATURE_NAMES:
            self.assertIn(feature, df.columns, f"Missing required NSL-KDD feature: {feature}")

    def test_02_model_training_and_serialization(self):
        """Verify that the model successfully trains and serializes artifacts."""
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "ensemble_nids.joblib")))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "scaler.joblib")))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "metadata.joblib")))
        self.assertGreater(self.metrics["accuracy"], 0.70)

    def test_03_model_reloading_and_inference(self):
        """Verify model reloading from disk and test predictions."""
        fresh_model = NIDSModel(model_dir=self.models_dir)
        loaded = fresh_model.load()
        self.assertTrue(loaded)

        test_df = self.df_train.head(10)
        preds, probs = fresh_model.predict(test_df)
        self.assertEqual(len(preds), 10)
        self.assertEqual(len(probs), 10)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_04_traffic_generator(self):
        """Verify that the traffic generator outputs valid packet records."""
        generator = NetworkTrafficGenerator(seed=123)
        stream = list(generator.stream_traffic(count=20, delay=0.0))
        self.assertEqual(len(stream), 20)

        first = stream[0]
        self.assertIn("src_ip", first)
        self.assertIn("dst_ip", first)
        self.assertIn("protocol_type", first)
        self.assertIn("flag", first)
        self.assertIn("service", first)

    def test_05_monitor_alert_generation(self):
        """Verify that the monitor detects intrusions and records structured alerts."""
        monitor = NIDSMonitor(model=self.model, logs_dir=self.logs_dir)
        generator = NetworkTrafficGenerator(seed=999)

        # Inspect known attack scenario
        dos_packet = generator._generate_record(attack_type="DoS")
        res = monitor.inspect_packet(dos_packet)
        self.assertTrue(res["is_intrusion"])
        self.assertEqual(res["predicted_attack"], "DoS")
        self.assertEqual(res["severity"], "HIGH")

        alerts_file, report_file = monitor.finalize_session()
        self.assertTrue(os.path.exists(alerts_file))
        self.assertTrue(os.path.exists(report_file))

    def test_06_detailed_prediction_dataframe(self):
        """Verify predict_detailed generates correct columns."""
        generator = NetworkTrafficGenerator(seed=42)
        sample_df = generator.generate_csv(os.path.join(self.test_dir, "test_traffic.csv"), num_records=25)
        detailed = self.model.predict_detailed(sample_df)

        self.assertIn("predicted_label", detailed.columns)
        self.assertIn("confidence", detailed.columns)
        self.assertIn("severity", detailed.columns)
        self.assertIn("is_intrusion", detailed.columns)
        self.assertEqual(len(detailed), 25)


if __name__ == "__main__":
    unittest.main()
