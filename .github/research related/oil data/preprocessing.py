##################################################
import pandas as pd
import json
# Load ExxonMobil Assay (XLS format)
exxon_data = pd.read_excel("exxonmobil_assays.xlsx")

# Load BP Assay (XLS format)
bp_data = pd.read_excel("bp_assays.xlsx")

# Load COA OpenEI Data (CSV format from archive)
coa_data = pd.read_csv("COA_openEI.csv")

# Load ADIOS Oil Database (JSON format)
with open('adios_data.json', 'r') as file:
    adios_data = json.load(file)

# Convert ADIOS JSON data into a DataFrame (structure may need adjusting depending on JSON schema)
adios_df = pd.json_normalize(adios_data)

# Step 1: Standardize Units across datasets
# Convert specific gravity to API gravity for consistency (if applicable)
def specific_gravity_to_api(gravity):
    return (141.5 / gravity) - 131.5 if gravity is not None else None

coa_data['API'] = coa_data['Specific_Gravity'].apply(specific_gravity_to_api)

# Step 2: Harmonize Column Names (make consistent)
def rename_columns(df):
    df.rename(columns={
        'Sulfur Content': 'Sulfur_wt%',
        'API Gravity': 'API',
        'Viscosity': 'Viscosity_cP',
        'Distillation Cut': 'Distillation_Temperature_C',
        # Add other necessary column harmonizations
    }, inplace=True)
    return df

# Apply to all dataframes
exxon_data = rename_columns(exxon_data)
bp_data = rename_columns(bp_data)
coa_data = rename_columns(coa_data)
adios_df = rename_columns(adios_df)

# Step 3: Merge All Dataframes
# Merge COA, Exxon, BP, and ADIOS dataframes on a common column like "Crude Name" (adjust based on actual data structure)
merged_df = pd.merge(coa_data, exxon_data, on="Crude Name", how="outer")
merged_df = pd.merge(merged_df, bp_data, on="Crude Name", how="outer")
merged_df = pd.merge(merged_df, adios_df, on="Crude Name", how="outer")

# Step 4: Handle Missing Data (Simple example: filling NaNs with a placeholder or interpolating)
merged_df.fillna({'API': merged_df['API'].mean(), 'Sulfur_wt%': merged_df['Sulfur_wt%'].mean()}, inplace=True)

# Step 5: Save the merged dataframe
merged_df.to_csv("merged_crude_assay_data.csv", index=False)

print("Merged and cleaned data saved as 'merged_crude_assay_data.csv'.")
