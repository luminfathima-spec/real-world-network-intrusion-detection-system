"""
Real-World Network Intrusion Detection System (NIDS)
Deployment and Execution Orchestrator
Author: Ameenul Lumin Fathima M.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from code.nids import NIDSModel, generate_benchmark_dataset
from network_traffic_generator import NetworkTrafficGenerator
from nids_monitor import NIDSMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NIDS-Deploy")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-World Network Intrusion Detection System (NIDS)")
    parser.add_argument(
        "--custom-file",
        type=str,
        default=None,
        help="Path to custom network traffic CSV to inspect and classify"
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force re-training the ensemble model even if saved weights exist"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=6000,
        help="Number of training benchmark samples if model needs training"
    )
    parser.add_argument(
        "--stream-count",
        type=int,
        default=60,
        help="Number of simulated network events to inspect during live monitoring"
    )
    return parser.parse_args()


def ensure_model_ready(force_train: bool = False, train_samples: int = 6000) -> NIDSModel:
    """Ensures trained ensemble model is available; trains on benchmark dataset if missing."""
    model = NIDSModel(model_dir="models")
    if force_train or not model.load():
        logger.info("Initializing baseline training on NSL-KDD benchmark distribution...")
        df_train, labels = generate_benchmark_dataset(num_samples=train_samples)
        X_train = model.preprocess(df_train, fit=True)
        metrics = model.train(X_train, labels)
        model.save()
        print("\n" + "="*50)
        print("          NIDS ENSEMBLE MODEL TRAINING REPORT")
        print("="*50)
        for k, v in metrics.items():
            print(f"  • {k.replace('_', ' ').capitalize():<18}: {v * 100:.2f}%")
        print("="*50 + "\n")
    else:
        logger.info("Pre-trained NIDS ensemble loaded from models directory.")
    return model


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n" + "#"*60)
    print("  🛡️  REAL-WORLD NETWORK INTRUSION DETECTION SYSTEM (NIDS)")
    print("  Ensemble ML Engine (Random Forest + Decision Tree)")
    print("  Author: Ameenul Lumin Fathima M.")
    print("#"*60 + "\n")

    model = ensure_model_ready(force_train=args.force_train, train_samples=args.train_samples)
    monitor = NIDSMonitor(model=model, logs_dir="logs")

    if args.custom_file:
        if not os.path.exists(args.custom_file):
            logger.error(f"Provided custom file not found: {args.custom_file}")
            sys.exit(1)

        logger.info(f"Loading custom network traffic file: {args.custom_file}")
        custom_df = pd.read_csv(args.custom_file)
        logger.info(f"Loaded {len(custom_df)} traffic records for inspection.")

        records = custom_df.to_dict(orient="records")
        for record in records:
            monitor.inspect_packet(record)

        alerts_file, report_file = monitor.finalize_session()

        # Export predictions
        detailed_df = model.predict_detailed(custom_df)
        out_csv = os.path.join("results", "real_world_predictions.csv")
        detailed_df.to_csv(out_csv, index=False)
        logger.info(f"Exported prediction results to {out_csv}")

    else:
        logger.info(f"Starting real-time traffic monitoring simulation ({args.stream_count} events)...")
        traffic_gen = NetworkTrafficGenerator()

        # Also generate prediction export dataset for results/real_world_predictions.csv
        sample_batch = traffic_gen.generate_csv(os.path.join("results", "temp_batch.csv"), num_records=100)
        if os.path.exists(os.path.join("results", "temp_batch.csv")):
            os.remove(os.path.join("results", "temp_batch.csv"))

        detailed_df = model.predict_detailed(sample_batch)
        out_csv = os.path.join("results", "real_world_predictions.csv")
        detailed_df.to_csv(out_csv, index=False)
        logger.info(f"Generated results export at {out_csv}")

        # Run live stream
        alerts_file, report_file = monitor.finalize_session()
        # Stream live events
        monitor.monitor_stream(traffic_gen, count=args.stream_count, delay=0.01)

    print("\n" + "="*50)
    print("           NIDS MONITORING SESSION COMPLETE")
    print("="*50)
    print(f"  Total Packets Inspected : {monitor.session_stats['total_packets']}")
    print(f"  Benign Packets Passed   : {monitor.session_stats['benign_packets']}")
    print(f"  Intrusions Intercepted  : {monitor.session_stats['intrusions_detected']}")
    print(f"  Alerts Archive          : {os.path.abspath('logs')}")
    print(f"  Predictions Export      : {os.path.abspath('results/real_world_predictions.csv')}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
