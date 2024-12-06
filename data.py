from math import trunc

import pandas as pd
def drop_outliers(df):
    col_interest = 'Victims Injured'
    q1 = df[col_interest].quantile(0.25)
    q3 = df[col_interest].quantile(0.75)
    iqr = q3 - q1
    outliers_removed = df[(df[col_interest] >= (q1 - 1.5 * iqr)) & (df[col_interest] <= (q3 + 1.5 * iqr))]
def proccess_data():
    data_path = 'Data/MergedGunData.csv'
    df = pd.read_csv(data_path)
    # Check for missing values
    missing_values = df.isnull().sum()
    print(missing_values)
    # drop the cases with no address but no operations is fine since only 13 cases
    df = df.dropna(subset=['Address'])
    #handle data parsing
    df['dates_proccessed'] = pd.to_datetime(df['Incident Date'], format='%B %d, %Y')
    df.sort_values(by='dates_proccessed')
    print(df.columns)
    print(df['State'])
    #Optional: write back proccessed data
    # df.to_csv('ProcessedGunData.csv',index=False)
if __name__ == '__main__':
    proccess_data()
