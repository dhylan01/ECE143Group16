import pandas as pd
import numpy as np
import folium
import json

# Step 1: Load the dataset
gun_data = pd.read_csv('./All_Year_Sorted_Data.csv')

# Step 2: Preprocess data
gun_data['Total_Victims'] = gun_data['Victims Killed'] + gun_data['Victims Injured']
gun_data['Severity_Ratio'] = gun_data['Victims Killed'] / gun_data['Victims Injured']
gun_data['Severity_Ratio'] = gun_data['Severity_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
gun_data['Law_Enforcement_Response'] = gun_data['Suspects Arrested'] / gun_data['Total_Victims']
gun_data['Law_Enforcement_Response'] = gun_data['Law_Enforcement_Response'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Step 3: Load real population data
state_population_data = {
    "California": 39538223, "Texas": 29145505, "Florida": 21538187, "New York": 20201249,
    "Pennsylvania": 13002700, "Illinois": 12812508, "Ohio": 11799448, "Georgia": 10711908,
    "North Carolina": 10439388, "Michigan": 10077331, "New Jersey": 9288994, "Virginia": 8631393,
    "Washington": 7705281, "Arizona": 7151502, "Massachusetts": 7029917, "Tennessee": 6910840,
    "Indiana": 6785528, "Missouri": 6154913, "Maryland": 6177224, "Wisconsin": 5893718,
    "Colorado": 5773714, "Minnesota": 5706494, "South Carolina": 5118425, "Alabama": 5024279,
    "Louisiana": 4657757, "Kentucky": 4505836, "Oregon": 4237256, "Oklahoma": 3963516,
    "Connecticut": 3605944, "Utah": 3271616, "Iowa": 3190369, "Nevada": 3104614,
    "Arkansas": 3011524, "Mississippi": 2961279, "Kansas": 2937880, "New Mexico": 2117522,
    "Nebraska": 1961504, "Idaho": 1839106, "West Virginia": 1793716, "Hawaii": 1455271,
    "New Hampshire": 1377529, "Maine": 1362359, "Montana": 1084225, "Rhode Island": 1097379,
    "Delaware": 989948, "South Dakota": 886667, "North Dakota": 779094, "Alaska": 733391,
    "Vermont": 643077, "Wyoming": 576851, "District of Columbia": 689545
}
state_population_df = pd.DataFrame(list(state_population_data.items()), columns=["State", "Population"])

# Step 4: Load GeoJSON for US states
with open('us-states.geojson', 'r') as geojson_file:
    us_states = json.load(geojson_file)

# Step 5: Initialize a combined map
combined_map = folium.Map(location=[37.8, -96], zoom_start=4)

# Step 6: Process overall state-level data (no year differentiation)
# Aggregate state-level data
state_trends = gun_data.groupby('State').agg(
    total_incidents=('Incident ID', 'count')
).reset_index()

state_trends = state_trends.merge(state_population_df, on="State", how="left")
state_trends['Incidents_per_Million'] = (state_trends['total_incidents'] / state_trends['Population']) * 1e6

state_ratios = gun_data.groupby('State').agg(
    Avg_Severity_Ratio=('Severity_Ratio', 'mean'),
    Avg_Response_Ratio=('Law_Enforcement_Response', 'mean')
).reset_index()

state_ratios = state_ratios.merge(
    state_trends[['State', 'Incidents_per_Million']],
    on="State",
    how="left"
)

# Add Incidents per Million Layer with yellow to red to black gradient
bins = list(state_trends['Incidents_per_Million'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
folium.Choropleth(
    geo_data=us_states,
    name=f"Incidents per Million",
    data=state_trends,
    columns=["State", "Incidents_per_Million"],
    key_on="feature.properties.name",
    fill_color="YlOrRd",  # Changed to 'YlOrRd' for yellow to red to black gradient
    nan_fill_color="white",
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Incidents per Million",
    bins=bins
).add_to(combined_map)

# Add Severity Ratio Layer with yellow to red to black gradient
bins = list(state_ratios['Avg_Severity_Ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
folium.Choropleth(
    geo_data=us_states,
    name=f"Severity Ratio",
    data=state_ratios,
    columns=["State", "Avg_Severity_Ratio"],
    key_on="feature.properties.name",
    fill_color="YlOrRd",  # Changed to 'YlOrRd' for yellow to red to black gradient
    nan_fill_color="white",
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Severity Ratio",
    bins=bins
).add_to(combined_map)

# Add Response Ratio Layer with red to yellow to green gradient (Red = Low response)
# First, calculate the bins
bins = list(state_ratios['Avg_Response_Ratio'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))

# Ensure bins are in increasing order
bins = sorted(bins)

# Now add the choropleth layer with the correct bins
folium.Choropleth(
    geo_data=us_states,
    name=f"Response Ratio",
    data=state_ratios,
    columns=["State", "Avg_Response_Ratio"],
    key_on="feature.properties.name",
    fill_color="RdYlGn",  # Red to Yellow to Green gradient (Red = Low response)
    nan_fill_color="white",
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Response Ratio (Low = Red)",
    bins=bins  # Pass bins in increasing order
).add_to(combined_map)


# Add JavaScript for "Select All" and "Deselect All"
select_all_script = """
<script>
    function toggleLayers(action) {
        const checkboxes = document.querySelectorAll('.leaflet-control-layers-selector');
        checkboxes.forEach(checkbox => {
            if (checkbox.checked !== (action === 'select')) {
                checkbox.click();
            }
        });
    }
</script>
<div style="position: fixed; top: 10px; left: 10px; z-index: 9999;">
    <button onclick="toggleLayers('select')">Select All</button>
    <button onclick="toggleLayers('deselect')">Deselect All</button>
</div>
"""

# Step 7: Add Layer Control
folium.LayerControl().add_to(combined_map)

# Inject the script and save the map
combined_map.get_root().html.add_child(folium.Element(select_all_script))
combined_map.save('2All_year_data_Combined_Gun_Violence_Map_With_YellowToRed_Gradient.html')
print("Map created with improved colors (Yellow to Red to Black gradient) and Select All/Deselect All buttons.")
