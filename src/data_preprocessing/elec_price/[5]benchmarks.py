import pandas as pd

data_raw = pd.read_csv('./data/elec_price/priceMT_full.txt', header=None)

# Transpose the table
df_transposed = data_raw.transpose()

# Set the first row as column names
df_transposed.columns = df_transposed.iloc[0]

# Drop the first row (original column names)
df_transposed = df_transposed[1:]

# Set the first column as 'date' and format the date
df_transposed.set_index(pd.date_range(start='1990-01-01', end='1999-12-01', freq='MS').strftime('%Y-%m-%d'))

# Convert type of columns
columns_to_convert = df_transposed.columns[0:]
df_transposed[columns_to_convert] = df_transposed[columns_to_convert].astype(float)

df_transposed.to_csv('./data/elec_price/priceMT_full_table.csv', index=False)