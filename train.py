import pandas as pd
from data_generator import generate_yassir_pricing_data
from agent import YassirPricingAgent

# ============================================================================
# PHASE 1: DATA GENERATION
# ============================================================================
print("--- Generating Synthetic Training Data ---")
# Generate 50,000 samples for a more robust training session
df = generate_yassir_pricing_data(num_samples=50000)
df.to_csv("yassir_synthetic_pricing_data.csv", index=False)
print("Synthetic data saved to yassir_synthetic_pricing_data.csv")
print("Data Preview:")
print(df.head())

# ============================================================================
# PHASE 2: DATA PREPARATION FOR PRE-TRAINING
# ============================================================================
print("\n--- Preparing Data for Offline Pre-training ---")
# Convert the DataFrame to the list format expected by the agent's offline_pretrain method
historical_logs = []
for _, row in df.iterrows():
    # Construct the state dictionary from the dataframe row
    state_dict = {
        "hour": int(row["hour"]),
        "day": int(row["day"]),
        "drivers": int(row["drivers"]),
        "requests": int(row["requests"]),
        "traffic": float(row["traffic"]),
        "weather": float(row["weather"])
    }

    # Extract the other required components for the training tuple
    multiplier = float(row["action_multiplier"])
    gmv = float(row["reward_gmv"])
    cancel_rate = float(row["reward_cancel_rate"])

    # Append the formatted tuple to our list
    historical_logs.append((state_dict, multiplier, gmv, cancel_rate))

print(f"Converted {len(historical_logs)} rows into the required training format.")

# ============================================================================
# PHASE 3: AGENT TRAINING
# ============================================================================
print("\n--- Initializing and Pre-training the Agent ---")
# Configure the agent for the "Bab Ezzouar" zone as per the instructions
zone_config = {"max_drivers": 150, "max_requests": 300}
agent = YassirPricingAgent(zone_config=zone_config)

# Train the agent on the historical data for 100 epochs
agent.offline_pretrain(historical_logs, epochs=100)

# Save the trained model
model_path = "yassir_pretrained_algiers.pth"
agent.save_model(model_path)
print(f"\nTraining complete. Model saved to {model_path}")


# ============================================================================
# PHASE 4: VALIDATION
# ============================================================================
print("\n--- Running Validation Checks on Trained Agent ---")
# Test on a couple of edge-case scenarios to see if the agent learned correctly

test_scenarios = [
    # Scenario 1: High demand, low supply, bad traffic & weather (should surge)
    {"hour": 18, "day": 4, "drivers": 40, "requests": 200, "traffic": 0.85, "weather": 0.2, "name": "Rush Hour + Rain"},
    # Scenario 2: High supply, low demand, good conditions (should discount)
    {"hour": 11, "day": 2, "drivers": 120, "requests": 50, "traffic": 0.3, "weather": 0.9, "name": "Midday Oversupply"},
]

for scenario in test_scenarios:
    # The agent's predict_price method handles state engineering internally
    multiplier, metadata = agent.predict_price(scenario)
    print(f"\nScenario: {scenario['name']}")
    print(f"  Input -> Drivers: {scenario['drivers']}, Requests: {scenario['requests']}")
    print(f"  Predicted Multiplier: {multiplier}x")
    print(f"  Metadata: {metadata}")