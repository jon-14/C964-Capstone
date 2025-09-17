import data.transform_data as td

df = td.import_from_csv()

print(df['currency'].unique())