import pandas as pd

FEATURES = ['PM2.5', 'PM10', 'NO2', 'CO']
TARGET   = 'AQI'


def clean_city_day(path):
    df = pd.read_csv(path)

    cols = ['City'] + FEATURES + [TARGET]
    df = df[cols]

    # Median imputation — keeps more samples than dropna
    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    return df.reset_index(drop=True)


def clean_city_hour(path):
    df = pd.read_csv(path)
    df = df[['City', 'AQI']]
    df = df.dropna()
    return df.reset_index(drop=True)