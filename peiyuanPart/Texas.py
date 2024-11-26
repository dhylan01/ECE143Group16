import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = './ProcessedGunData.csv'
gun_data = pd.read_csv(file_path)

# Preprocess data
gun_data['Total_Victims'] = gun_data['Victims Killed'] + gun_data['Victims Injured']
gun_data['Severity_Ratio'] = gun_data['Victims Killed'] / gun_data['Victims Injured']
gun_data['Severity_Ratio'] = gun_data['Severity_Ratio'].replace([float('inf'), -float('inf')], 0).fillna(0)
gun_data['Law_Enforcement_Response'] = gun_data['Suspects Arrested'] / gun_data['Total_Victims']
gun_data['Law_Enforcement_Response'] = gun_data['Law_Enforcement_Response'].replace([float('inf'), -float('inf')], 0).fillna(0)

# Texas data
state_population_data = {
    "Texas": 29145505
}
state_population_df = pd.DataFrame(list(state_population_data.items()), columns=["State", "Population"])

# Calculate incidents per million for Texas by year
state_trends = gun_data.groupby(['Year', 'State']).agg(
    total_incidents=('Incident ID', 'count')
).reset_index()

state_trends = state_trends.merge(state_population_df, on="State", how="inner")
state_trends['Incidents_per_Million'] = (state_trends['total_incidents'] / state_trends['Population']) * 1e6

# Filter for Texas
tx_data = state_trends[state_trends['State'] == 'Texas']

# Plot data for Texas
plt.figure(figsize=(10, 6))
plt.bar(tx_data['Year'].astype(str), tx_data['Incidents_per_Million'], color='green', alpha=0.7)
plt.title('Incidents per Million (Texas)')
plt.xlabel('Year')
plt.ylabel('Incidents per Million')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
