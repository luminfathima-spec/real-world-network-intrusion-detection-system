"""
Real-World Network Intrusion Detection System (NIDS)
Network Traffic Generator Module
Simulates realistic benign and adversarial network traffic flows.
Author: Ameenul Lumin Fathima M.
"""

import time
import random
import datetime
from typing import Generator, Dict, Any, List, Optional
import pandas as pd
from code.nids import FEATURE_NAMES


class NetworkTrafficGenerator:
    """
    Simulates high-fidelity network traffic streams incorporating both benign operations
    and advanced persistent cyber threats.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.internal_ips = [
            f"192.168.1.{i}" for i in range(10, 100)
        ] + [
            f"10.0.0.{i}" for i in range(2, 50)
        ]
        self.server_ips = [
            "192.168.1.5",   # Web Server
            "192.168.1.6",   # Database Server
            "192.168.1.7",   # Mail Server
            "192.168.1.8",   # Internal Gateway
            "10.0.0.1"       # Domain Controller
        ]
        self.external_ips = [
            f"198.51.100.{i}" for i in range(1, 250)
        ] + [
            f"203.0.113.{i}" for i in range(1, 250)
        ]

    def _generate_record(self, attack_type: str = "Normal") -> Dict[str, Any]:
        """Synthesize a single packet/session record corresponding to the specified scenario."""
        record: Dict[str, Any] = {col: 0.0 for col in FEATURE_NAMES}
        record["timestamp"] = datetime.datetime.now().isoformat()

        if attack_type == "Normal":
            src_ip = random.choice(self.internal_ips)
            dst_ip = random.choice(self.server_ips + self.external_ips)
            src_port = random.randint(30000, 65000)
            dst_port = random.choice([80, 443, 53, 25, 22])

            record["protocol_type"] = random.choices(["tcp", "udp", "icmp"], weights=[0.8, 0.18, 0.02])[0]
            record["service"] = "http" if dst_port in [80, 443] else ("domain_u" if dst_port == 53 else "smtp")
            record["flag"] = "SF"
            record["duration"] = round(random.uniform(0.1, 5.0), 2)
            record["src_bytes"] = random.randint(120, 2500)
            record["dst_bytes"] = random.randint(500, 30000)
            record["logged_in"] = 1.0 if record["protocol_type"] == "tcp" else 0.0
            record["count"] = random.randint(1, 15)
            record["srv_count"] = random.randint(1, 15)
            record["same_srv_rate"] = round(random.uniform(0.9, 1.0), 2)
            record["diff_srv_rate"] = round(random.uniform(0.0, 0.1), 2)
            record["dst_host_count"] = random.randint(10, 255)
            record["dst_host_srv_count"] = random.randint(10, 255)
            record["dst_host_same_srv_rate"] = round(random.uniform(0.85, 1.0), 2)
            record["dst_host_diff_srv_rate"] = round(random.uniform(0.0, 0.08), 2)

        elif attack_type == "DoS":
            src_ip = random.choice(self.external_ips)
            dst_ip = "192.168.1.5"  # Victim Web Server
            src_port = random.randint(1024, 65535)
            dst_port = 80

            record["protocol_type"] = "tcp"
            record["service"] = "http"
            record["flag"] = random.choice(["S0", "REJ"])
            record["duration"] = 0.0
            record["src_bytes"] = 0
            record["dst_bytes"] = 0
            record["count"] = random.randint(250, 511)
            record["srv_count"] = random.randint(200, 511)
            record["serror_rate"] = round(random.uniform(0.85, 1.0), 2)
            record["srv_serror_rate"] = round(random.uniform(0.85, 1.0), 2)
            record["same_srv_rate"] = 1.0
            record["dst_host_count"] = 255
            record["dst_host_srv_count"] = random.randint(1, 20)
            record["dst_host_serror_rate"] = round(random.uniform(0.9, 1.0), 2)

        elif attack_type == "DDoS":
            src_ip = random.choice(self.external_ips)
            dst_ip = "192.168.1.5"
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 53, 443])

            record["protocol_type"] = "udp"
            record["service"] = "private"
            record["flag"] = "SF"
            record["src_bytes"] = random.randint(1500, 4000)
            record["count"] = random.randint(400, 512)
            record["srv_count"] = random.randint(400, 512)
            record["same_srv_rate"] = 1.0
            record["dst_host_count"] = 255
            record["dst_host_srv_count"] = 255
            record["dst_host_same_src_port_rate"] = round(random.uniform(0.8, 1.0), 2)

        elif attack_type == "Port Scanning":
            src_ip = random.choice(self.external_ips)
            dst_ip = random.choice(self.server_ips)
            src_port = random.randint(40000, 65000)
            dst_port = random.randint(1, 1024)

            record["protocol_type"] = "tcp"
            record["service"] = "private"
            record["flag"] = random.choice(["REJ", "RSTO", "S0"])
            record["src_bytes"] = 0
            record["dst_bytes"] = 0
            record["count"] = random.randint(100, 300)
            record["srv_count"] = random.randint(1, 5)
            record["diff_srv_rate"] = round(random.uniform(0.7, 1.0), 2)
            record["dst_host_count"] = 255
            record["dst_host_diff_srv_rate"] = round(random.uniform(0.6, 1.0), 2)

        elif attack_type == "Brute Force":
            src_ip = random.choice(self.external_ips)
            dst_ip = "192.168.1.7"  # Mail or SSH server
            src_port = random.randint(30000, 60000)
            dst_port = 22

            record["protocol_type"] = "tcp"
            record["service"] = "telnet"
            record["flag"] = "SF"
            record["duration"] = round(random.uniform(1.0, 8.0), 2)
            record["src_bytes"] = random.randint(200, 900)
            record["dst_bytes"] = random.randint(200, 1500)
            record["num_failed_logins"] = random.randint(4, 12)
            record["logged_in"] = random.choice([0.0, 1.0])
            record["count"] = random.randint(10, 40)
            record["same_srv_rate"] = round(random.uniform(0.8, 1.0), 2)

        elif attack_type == "Buffer Overflow":
            src_ip = random.choice(self.external_ips)
            dst_ip = "192.168.1.6"  # Database
            src_port = random.randint(30000, 60000)
            dst_port = 1521

            record["protocol_type"] = "tcp"
            record["service"] = "other"
            record["flag"] = "SF"
            record["duration"] = round(random.uniform(5.0, 25.0), 2)
            record["src_bytes"] = random.randint(3500, 12000)
            record["dst_bytes"] = random.randint(1200, 6000)
            record["root_shell"] = 1.0
            record["num_root"] = random.randint(1, 4)
            record["num_file_creations"] = random.randint(1, 3)
            record["logged_in"] = 1.0

        elif attack_type == "Backdoor":
            src_ip = random.choice(self.internal_ips)
            dst_ip = random.choice(self.external_ips)
            src_port = random.randint(40000, 60000)
            dst_port = 4444  # Typical C2 port

            record["protocol_type"] = "tcp"
            record["service"] = "private"
            record["flag"] = "SF"
            record["duration"] = round(random.uniform(30.0, 180.0), 2)
            record["src_bytes"] = random.randint(80, 450)
            record["dst_bytes"] = random.randint(80, 450)
            record["num_shells"] = 1.0
            record["logged_in"] = 1.0
            record["count"] = random.randint(1, 4)

        record["src_ip"] = src_ip
        record["dst_ip"] = dst_ip
        record["src_port"] = src_port
        record["dst_port"] = dst_port
        record["ground_truth"] = attack_type

        return record

    def stream_traffic(self, count: int = 100, delay: float = 0.05) -> Generator[Dict[str, Any], None, None]:
        """Yields traffic events with inter-packet delay simulating live network telemetry."""
        attack_choices = [
            "Normal", "Port Scanning", "DoS", "DDoS",
            "Brute Force", "Buffer Overflow", "Backdoor"
        ]
        weights = [0.55, 0.12, 0.15, 0.07, 0.05, 0.03, 0.03]

        for _ in range(count):
            scenario = random.choices(attack_choices, weights=weights)[0]
            record = self._generate_record(scenario)
            yield record
            if delay > 0:
                time.sleep(delay)

    def generate_csv(self, filename: str, num_records: int = 500) -> pd.DataFrame:
        """Saves a batch of simulated traffic to a CSV file."""
        records = [record for record in self.stream_traffic(count=num_records, delay=0.0)]
        df = pd.DataFrame(records)
        df.to_csv(filename, index=False)
        return df


if __name__ == "__main__":
    generator = NetworkTrafficGenerator(seed=42)
    sample_df = generator.generate_csv("sample_traffic.csv", num_records=20)
    print(f"Generated {len(sample_df)} sample network traffic records.")
