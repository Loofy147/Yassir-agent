import pandas as pd
import os
import datetime
from src.ml.dqn_agent import YassirPricingAgent

# ============================================================================
# PHASE 1: DATA LOADING AND CLEANING
# ============================================================================
print("--- Loading and Cleaning Synthetic Training Data ---")
try:
    df = pd.read_csv("yassir_synthetic_pricing_data.csv")
    print("Successfully loaded yassir_synthetic_pricing_data.csv")
except FileNotFoundError:
    print("Error: yassir_synthetic_pricing_data.csv not found.")
    print("Please run data_generator.py first to generate the dataset.")
    exit()

# Data Quality Enhancement: Filter out samples with high cancellation rates
initial_rows = len(df)
df_clean = df[df['reward_cancel_rate'] <= 0.05].copy()
final_rows = len(df_clean)
print(f"Filtered out {initial_rows - final_rows} samples with cancellation rates > 5%.")
print(f"Clean dataset contains {final_rows} samples.")

# ============================================================================
# PHASE 2: DATA PREPARATION FOR PRE-TRAINING
# ============================================================================
print("\n--- Preparing Data for Offline Pre-training ---")
historical_logs = []
for _, row in df_clean.iterrows():
    state_dict = {
        "hour": int(row["hour"]),
        "day": int(row["day"]),
        "drivers": int(row["drivers"]),
        "requests": int(row["requests"]),
        "traffic": float(row["traffic"]),
        "weather": float(row["weather"])
    }
    historical_logs.append((
        state_dict,
        float(row["action_multiplier"]),
        float(row["reward_gmv"]),
        float(row["reward_cancel_rate"])
    ))
print(f"Converted {len(historical_logs)} rows into the required training format.")

# ============================================================================
# PHASE 3: AGENT TRAINING
# ============================================================================
print("\n--- Initializing and Pre-training the Agent ---")
zone_config = {"max_drivers": 150, "max_requests": 300}
agent = YassirPricingAgent(zone_config=zone_config)

# Train the agent on the cleaned historical data
agent.offline_pretrain(historical_logs, epochs=100)

# ============================================================================
# PHASE 4: SAVING THE MODEL (with Versioning)
# ============================================================================
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Generate a timestamp for the model version
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Save the model with a zone-and-version-specific name
ZONE_NAME = "BAB_EZZOUAR"
model_filename = f"{ZONE_NAME}-v{timestamp}.pth"
model_path = os.path.join(MODELS_DIR, model_filename)
agent.save_model(model_path)
print(f"\n✅ Training complete. Model for zone '{ZONE_NAME}' saved to {model_path}")

# ============================================================================
# PHASE 5: VALIDATION
# ============================================================================
print("\n--- Running Validation Checks on Trained Agent ---")
test_scenarios = [
    {"hour": 18, "day": 4, "drivers": 40, "requests": 200, "traffic": 0.85, "weather": 0.2, "name": "Rush Hour + Rain (Expect Surge)"},
    {"hour": 11, "day": 2, "drivers": 120, "requests": 50, "traffic": 0.3, "weather": 0.9, "name": "Midday Oversupply (Expect Discount)"},
]

for scenario in test_scenarios:
    multiplier, metadata = agent.predict_price(scenario)
    print(f"\nScenario: {scenario['name']}")
    print(f"  Input -> Drivers: {scenario['drivers']}, Requests: {scenario['requests']}")
    print(f"  Predicted Multiplier: {multiplier}x")
    print(f"  Metadata: {metadata}")
