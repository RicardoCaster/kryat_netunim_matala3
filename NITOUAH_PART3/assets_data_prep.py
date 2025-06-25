def prepare_data(df, mode="train"):
    import pandas as pd
    import numpy as np
    
### Clean and preprocess both TRAIN and TEST datasets automatically The function detects whether 'price' exists to decide how to treat the data (point 4 below).
    
    # --- 1. Keep only regular apartments ---
    df = df[df['property_type'] == 'דירה']
    df = df.drop(columns=['property_type'])

    # --- 2. Drop irrelevant columns ---
    columns_to_drop = ['address', 'description', 'num_of_payments', 'num_of_images']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # --- 3. Convert important columns to numeric ---
    numeric_cols = ['room_num', 'area', 'price', 'floor', 'total_floors',
                    'monthly_arnona', 'building_tax', 'days_to_enter', 'distance_from_center']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 4. Drop rows with unrealistic price only if we are in training set
    if 'price' in df.columns:
        df = df[df['price'].notna()]
        df = df[(df['price'] >= 1000) & (df['price'] <= 50000)]

    # --- 5. Remove unrealistic area values ---
    df = df[df['area'] >= 10]

    # --- 6. Remove rows without neighborhood ---
    df = df[df['neighborhood'].notna()]

    # --- 7. Impute room_num when equal to 0 using similar apartments by area ---
    def impute_room_num(row, reference_df):
        if row['room_num'] == 0:
            similar = reference_df[
                (reference_df['area'] >= row['area'] - 5) &
                (reference_df['area'] <= row['area'] + 5) &
                (reference_df['room_num'] > 0)
            ]
            if not similar.empty:
                return round(similar['room_num'].median(), 1)
            else:
                return round(reference_df['room_num'].median(), 1)
        return row['room_num']

    if 'room_num' in df.columns and 'area' in df.columns:
        reference_df = df[df['room_num'] > 0]
        df['room_num'] = df.apply(lambda row: impute_room_num(row, reference_df), axis=1)

    # --- 8. Fix total_floors if it's less than floor ---
    if 'floor' in df.columns and 'total_floors' in df.columns:
        df.loc[(df['floor'] > df['total_floors']) & df['total_floors'].notna(), 'total_floors'] = df['floor']

    # --- 9. Fill missing total_floors using floor or median ---
    if 'total_floors' in df.columns:
        median_total = df[df['total_floors'] > 0]['total_floors'].median()

        def fix_missing_total_floors(row):
            if pd.isna(row['total_floors']):
                if not pd.isna(row['floor']):
                    return max(row['floor'], median_total)
                else:
                    return median_total
            return row['total_floors']

        df['total_floors'] = df.apply(fix_missing_total_floors, axis=1)

    # --- 10. Fill missing values in each column with reasonable defaults ---
    if 'floor' in df.columns:
        df['floor'] = df['floor'].fillna(df['floor'].median())
    if 'days_to_enter' in df.columns:
        df['days_to_enter'] = df['days_to_enter'].fillna(df['days_to_enter'].median())
    if 'distance_from_center' in df.columns:
        df['distance_from_center'] = df['distance_from_center'].fillna(df['distance_from_center'].median())
    if 'monthly_arnona' in df.columns:
        df['monthly_arnona'] = df['monthly_arnona'].fillna(df['monthly_arnona'].mean())
    if 'building_tax' in df.columns:
        df['building_tax'] = df['building_tax'].fillna(df['building_tax'].mean())
    if 'garden_area' in df.columns:
        df['garden_area'] = df['garden_area'].fillna(0)

    # --- 11. Encode 'neighborhood' with one-hot encoding ---
    if 'neighborhood' in df.columns:
        df = pd.get_dummies(df, columns=['neighborhood'], drop_first=True)

    return df
