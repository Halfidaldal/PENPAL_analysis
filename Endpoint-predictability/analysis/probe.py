import matplotlib.pyplot as plt
import pandas as pd
import ast
from pathlib import Path

# Resolve directory paths relative to this script
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "surprisal_scores.csv"

def plot_trajectory(raw_surprisals, title="Endpoint Predictability Curve"):
    # Cumulative drop: S(0) - S(t)
    s0 = raw_surprisals[0]
    cumulative_drops = [s0 - st for st in raw_surprisals[1:]]
    positions = list(range(1, len(cumulative_drops) + 1))
    
    plt.figure(figsize=(8, 5))
    plt.plot(positions, cumulative_drops, marker='o', color='#3b82f6', label="Real Order")
    
    # Draw a diagonal reference line to show perfect linearity
    if len(positions) >= 2:
        plt.plot([1, positions[-1]], [0, cumulative_drops[-1]], 
                 linestyle='--', color='#94a3b8', label="Perfect Linear Baseline")
    
    plt.title(title)
    plt.xlabel("Sentence Position (t)")
    plt.ylabel("Cumulative Surprisal Drop (bits)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_output_path = DATA_DIR / "predictability_curve.png"
    plt.savefig(plot_output_path)
    print(f"Plot saved to: {plot_output_path}")

def main():
    if not CSV_PATH.exists():
        print(f"Error: Could not find scores at {CSV_PATH}")
        return
        
    print(f"Loading scores from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Grab the first row (the pilot paper)
    row = df.iloc[0]
    
    print("\n=== Computed Information Flow Metrics ===")
    print(f"Mean Surprisal Drop (mean_delta_s): {row['mean_delta_s']:.5f} bits")
    print(f"Variance of Drops (var_delta_s):   {row['var_delta_s']:.5f}")
    print(f"Linearity (linearity_r2):          {row['linearity_r2']:.5f}")
    print(f"Cumulative Slope (cum_slope):      {row['cum_slope']:.5f}")
    print(f"Tail-Drop Ratio:                   {row['tail_drop_ratio']:.5f}")
    print("=========================================\n")
    
    # Parse the raw surprisal list from string format in CSV
    raw_surprisals_str = row['raw_surprisals']
    try:
        raw_surprisals = ast.literal_eval(raw_surprisals_str)
    except Exception as e:
        print(f"Error parsing raw surprisals list: {e}")
        return
        
    # Generate the curve plot
    plot_trajectory(raw_surprisals, title="Pilot Paper Predictability Curve (Gemma-4)")

if __name__ == "__main__":
    main()
