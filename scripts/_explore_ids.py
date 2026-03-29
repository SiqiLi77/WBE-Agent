import pandas as pd
df = pd.read_csv("data/raw/NWSS_Public_SARS-CoV-2_Concentration_in_Wastewater_Data.csv", nrows=1000)
samples = df["key_plot_id"].unique()[:30]
for s in samples:
    parts = s.split("_")
    state_guess = parts[2] if len(parts) > 2 else "N/A"
    print(f"  state={state_guess:4s}  |  {s[:90]}")
