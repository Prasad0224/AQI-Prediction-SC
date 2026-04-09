import pandas as pd

FEATURES = ['PM2.5', 'PM10', 'NO2', 'CO']
TARGET   = 'AQI'


def clean_city_day(path):
    df = pd.read_csv(path)

    cols = ['City'] + FEATURES + [TARGET]
    df = df[cols]

    # Use forward fill then backward fill to preserve time-series continuity,
    # fallback to median only if all are NaN for a specific city block.
    df[FEATURES] = df.groupby('City')[FEATURES].ffill().bfill()
    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[TARGET])
    
    # Drop impossible negative AQI and extreme outliers (sensor errors > 600)
    df = df[(df[TARGET] > 0) & (df[TARGET] <= 600)]
    
    # Stricter physical bounds for the incoming 4 features to prevent models 
    # from skewing due to broken telemetry.
    df = df[
        (df['PM2.5'] <= 500) &
        (df['PM10']  <= 800) &
        (df['NO2']   <= 200) &
        (df['CO']    <= 10)
    ]

    return df.reset_index(drop=True)


def clean_city_hour(path):
    df = pd.read_csv(path)
    
    # Ensure Datetime object to extract hour
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Hour'] = df['Datetime'].dt.hour
    else:
        df['Hour'] = 0  # Fallback

    cols = ['City', 'AQI', 'Hour']
    available_cols = [c for c in cols if c in df.columns]
    
    df = df[available_cols].dropna(subset=['AQI'])
    # Filter extreme outliers
    df = df[(df['AQI'] > 0) & (df['AQI'] <= 600)]
    return df.reset_index(drop=True)