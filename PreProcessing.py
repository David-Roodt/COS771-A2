import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

file_path = 'almonds/Almond.csv'
almond_data = pd.read_csv(file_path)

almond_data.info(), print(almond_data.head())

almond_data_cleaned = almond_data.drop(columns=['Unnamed: 0'])

features = almond_data_cleaned.drop(columns=['Type'])
target = almond_data_cleaned['Type']

scaler = MinMaxScaler()
features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

features_normalized_filled = features_normalized.fillna(-1)

encoder = OneHotEncoder(sparse=False)
target_encoded_filled = pd.DataFrame(encoder.fit_transform(target.values.reshape(-1, 1)), columns=encoder.categories_[0])

preprocessed_data_filled = pd.concat([features_normalized_filled, target_encoded_filled], axis=1)

for index, row in preprocessed_data_filled.iterrows():
        for col in preprocessed_data_filled.columns:
            value = row[col]
            if not (0 <= value <= 1 or value == -1):
                print(f"Invalid value found: {value} at row {index}, column '{col}'")

output_file_path = 'almonds/Almond_Prepped.csv'
preprocessed_data_filled.to_csv(output_file_path, index=False)