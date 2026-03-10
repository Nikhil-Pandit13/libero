##########################################################
import pandas as pd

# Read Excel file
df_vortexa = pd.read_excel("C:/Users/nikhilp/Downloads/ucube.xlsx")
df_cleaned = df_vortexa.dropna(subset=['API_UCube', 'Sulfur_UCube'], how='all')

# Show first 5 rows
print(df_vortexa.head())


df_indres = pd.read_excel("C:/Users/nikhilp/Downloads/indres.xlsx")
df_indres_cleaned = df_indres.dropna(subset=['API', 'Sulfur'], how='all')

column_rename_dict = {
    'API_UCube': 'API',
    'Sulfur_UCube': 'Sulfur',
    # Add more columns as needed
}

# Rename the columns using the dictionary
df_vortexa.rename(columns=column_rename_dict, inplace=True)

df_merged = df_vortexa.merge(df_indres_cleaned, on='RowID', how='left', suffixes=('_vortexa', '_indres_cleaned'))
df_merged = df_merged.drop_duplicates(subset='RowID', keep='first')

hello = df_merged.copy()
hello['API_vortexa'] = hello['API_indres_cleaned'].combine_first(hello['API_vortexa'])
hello['Sulfur_vortexa'] = hello['Sulfur_indres_cleaned'].combine_first(hello['Sulfur_vortexa'])
hello = hello.drop(columns=['API_indres_cleaned', 'Sulfur_indres_cleaned'])
hello.to_excel('C:/Users/nikhilp/Downloads/hello.xlsx', index=False)