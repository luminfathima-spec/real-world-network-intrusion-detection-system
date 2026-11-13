"""
Real-World Network Intrusion Detection System (NIDS)
Real-Time Monitoring and Alerting Engine
Author: Ameenul Lumin Fathima M.
"""

import os
import json
import time
import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from code.nids import NIDSModel, SEVERITY_LEVELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NIDS-Monitor")


class NIDSMonitor:
    """
    Real-Time Intrusion Detection Monitor.
    Ingests live traffic streams, detects threats via ML ensemble,
    triggers structured severity alerts, and archives monitoring reports.
    """

    def __init__(self, model: Optional[NIDSModel] = None, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

        if model is None:
            self.model = NIDSModel()
            if not self.model.load():
                logger.info("No saved model found. Please train model first via deploy.py.")
        else:
            self.model = model

        self.alerts: List[Dict[str, Any]] = []
        self.session_stats: Dict[str, Any] = {
            "total_packets": 0,
            "benign_packets": 0,
            "intrusions_detected": 0,
            "attack_breakdown": {},
            "severity_counts": {
                "INFO": 0,
                "MEDIUM": 0,
                "HIGH": 0,
                "CRITICAL": 0
            },
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None
        }

    def inspect_packet(self, packet_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single network packet/flow record.
        Generates security alerts if malicious patterns are detected.
        """
        df_single = pd.DataFrame([packet_data])
        preds, probs = self.model.predict(df_single)
        predicted_label = str(preds[0])
        confidence = float(round(probs[0], 4))

        is_intrusion = predicted_label != "Normal"
        severity = SEVERITY_LEVELS.get(predicted_label, "INFO")

        # Update stats
        self.session_stats["total_packets"] += 1
        if is_intrusion:
            self.session_stats["intrusions_detected"] += 1
            self.session_stats["attack_breakdown"][predicted_label] = (
                self.session_stats["attack_breakdown"].get(predicted_label, 0) + 1
            )
        else:
            self.session_stats["benign_packets"] += 1

        self.session_stats["severity_counts"][severity] = (
            self.session_stats["severity_counts"].get(severity, 0) + 1
        )

        result = {
            "timestamp": packet_data.get("timestamp", datetime.datetime.now().isoformat()),
            "src_ip": packet_data.get("src_ip", "0.0.0.0"),
            "dst_ip": packet_data.get("dst_ip", "0.0.0.0"),
            "src_port": packet_data.get("src_port", 0),
            "dst_port": packet_data.get("dst_port", 0),
            "protocol": packet_data.get("protocol_type", "tcp"),
            "predicted_attack": predicted_label,
            "confidence": confidence,
            "severity": severity,
            "is_intrusion": is_intrusion,
            "action_taken": "ALERT_AND_BLOCK" if severity in ["HIGH", "CRITICAL"] else ("LOG" if is_intrusion else "ALLOW")
        }

        if is_intrusion:
            self.alerts.append(result)
            log_fn = logger.warning if severity == "MEDIUM" else logger.error
            log_fn(
                f"[{severity}] Threat Detected! Type: {predicted_label} | "
                f"Src: {result['src_ip']}:{result['src_port']} -> Dst: {result['dst_ip']}:{result['dst_port']} | "
                f"Confidence: {confidence * 100:.1f}% | Action: {result['action_taken']}"
            )

        return result

    def monitor_stream(self, traffic_generator: Any, count: int = 50, delay: float = 0.02) -> List[Dict[str, Any]]:
        """Consumes a live stream from the traffic generator."""
        logger.info(f"Starting Live Intrusion Monitoring Stream ({count} events)...")
        results = []
        for packet in traffic_generator.stream_traffic(count=count, delay=delay):
            inspected = self.inspect_packet(packet)
            results.append(inspected)

        self.finalize_session()
        return results

    def finalize_session(self) -> Tuple[str, str]:
        """
        Saves alerts and comprehensive summary report as JSON files.
        Returns paths to saved alerts and report files.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_stats["end_time"] = datetime.datetime.now().isoformat()

        # Save Alerts JSON
        alerts_file = os.path.join(self.logs_dir, f"alerts_{timestamp}.json")
        with open(alerts_file, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.datetime.now().isoformat(),
                "total_alerts": len(self.alerts),
                "alerts": self.alerts
            }, f, indent=4)
        logger.info(f"Saved {len(self.alerts)} security alerts to {alerts_file}")

        # Save Report JSON
        report_file = os.path.join(self.logs_dir, f"report_{timestamp}.json")
        total = max(self.session_stats["total_packets"], 1)
        intrusion_rate = round((self.session_stats["intrusions_detected"] / total) * 100, 2)

        summary_report = {
            "report_generated_at": datetime.datetime.now().isoformat(),
            "session_metrics": self.session_stats,
            "threat_detection_rate_pct": intrusion_rate,
            "security_posture": "COMPROMISED" if self.session_stats["severity_counts"]["CRITICAL"] > 0 else (
                "WARNING" if self.session_stats["severity_counts"]["HIGH"] > 0 else "NORMAL"
            )
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=4)
        logger.info(f"Saved session analysis report to {report_file}")

        return alerts_file, report_file
