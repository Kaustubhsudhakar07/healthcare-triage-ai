"""
Real-Time Ambulance Field Telemetry Streaming Demo
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Simulates an active emergency dispatch center receiving live field telemetry
packets from multiple ambulance units with colorized triage outputs and safety alerts.
"""

import os
import sys
import time
import random
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from src.predict import ClinicalInferenceService


# ANSI Terminal Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


TIER_COLORS = {
    "Low": Colors.GREEN,
    "Moderate": Colors.YELLOW,
    "Elevated": "\033[38;5;208m",  # Orange
    "High": Colors.RED,
    "Critical": Colors.HEADER
}


def stream_ambulance_telemetry(num_encounters: int = 10, delay_seconds: float = 1.0):
    """
    Streams simulated ambulance arrivals with live AI triage scoring.
    """
    pipeline_path = "models/pipeline.joblib"
    if not os.path.exists(pipeline_path):
        print(f"Error: Model pipeline not found at {pipeline_path}. Run 'python src/train.py' first.")
        return

    service = ClinicalInferenceService(pipeline_path=pipeline_path, train_path="data/processed/train.csv")
    
    test_path = "data/processed/test.csv"
    if not os.path.exists(test_path):
        print(f"Error: Test data not found at {test_path}.")
        return
        
    test_df = pd.read_csv(test_path).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.CYAN}[EMS STREAM] LIVE AMBULANCE TELEMETRY & FIELD TRIAGE STREAM{Colors.END}")
    print(f"Listening to active emergency transit channels (Simulating {num_encounters} encounters)...")
    print("=" * 80 + "\n")
    
    for i in range(min(num_encounters, len(test_df))):
        row = test_df.iloc[i].to_dict()
        patient_id = row.get("patient_id", f"AMB-UNIT-{101 + i}")
        unit_id = f"MEDIC-UNIT-{random.randint(10, 99)}"
        
        t0 = time.perf_counter()
        result = service.predict(row)
        latency = (time.perf_counter() - t0) * 1000
        
        score = result["criticality_score"]
        tier = result["urgency_tier"]
        color = TIER_COLORS.get(tier, Colors.BLUE)
        
        print(f"[{time.strftime('%H:%M:%S')}] {Colors.BOLD}{unit_id}{Colors.END} --> Inbound Transit for Patient: {Colors.BOLD}{patient_id}{Colors.END}")
        print(f"   Demographics: Age {row['age']}, Sex {row['sex']} | Vitals: HR {row['heart_rate']:.0f} | BP {row['systolic_bp']:.0f}/{row['diastolic_bp']:.0f} | SpO2 {row['spo2']:.0f}% | GCS {row['gcs']}")
        print(f"   Acuity Score: {color}{Colors.BOLD}{score:.1f} / 10.0{Colors.END} | Priority Tier: {color}{Colors.BOLD}[{tier.upper()}]{Colors.END} (Latency: {latency:.1f}ms)")
        
        if result["red_flags"]:
            for rf in result["red_flags"]:
                print(f"   {Colors.RED}[RED FLAG] {rf}{Colors.END}")
                
        if result.get("explanation") and "narrative" in result["explanation"]:
            top_narr = result["explanation"]["narrative"][:2]
            print(f"   [DRIVERS] Primary Physiological Drivers: {'; '.join(top_narr)}")
            
        print(f"   [ROUTING] {result['clinical_routing_guidance']}")
        print("-" * 80)
        
        time.sleep(delay_seconds)
        
    print(f"\n{Colors.GREEN}[COMPLETE] Telemetry stream simulation finished.{Colors.END}\n")


if __name__ == "__main__":
    stream_ambulance_telemetry(num_encounters=5, delay_seconds=0.5)
