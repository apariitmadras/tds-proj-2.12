# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "s3fs",
#   "scikit-learn",
# ]
# ///
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import s3fs
from sklearn.linear_model import LinearRegression
import numpy as np


def fetch_data(s3_bucket_path, metadata_parquet_path, courts_of_interest=None, start_year=None, end_year=None, court_code=None, columns_of_interest=None):
    """
    Fetches and filters data from the S3 bucket using DuckDB.
    """
    s3 = s3fs.S3FileSystem(anon=False) # or set anon=True for public access.
    con = duckdb.connect()

    if isinstance(courts_of_interest, list):
         courts_of_interest_str = ", ".join([f"'{court}'" for court in courts_of_interest])
    else:
        courts_of_interest_str = courts_of_interest

    parquet_path = metadata_parquet_path.replace("*","{year}")
    
    if courts_of_interest is not None and start_year is not None and end_year is not None:
        all_files = []
        for year in range(start_year, end_year + 1):
            for court in courts_of_interest:
                file_path = f"{s3_bucket_path}{parquet_path.format(year=year, court=court, bench='')}"
                try:
                    files = s3.glob(file_path)
                    all_files.extend([f"s3://{file}" for file in files])
                except Exception as e:
                    print(f"Error globbing {file_path}: {e}")
        if len(all_files) == 0:
             return None

        query = f"""
            SELECT * FROM read_parquet([{', '.join([f"'{file}'" for file in all_files])}])
            WHERE year BETWEEN {start_year} AND {end_year}
            """
        
    elif court_code:
        file_path = f"{s3_bucket_path}{parquet_path.format(year='*', court=court_code, bench='')}"
        files = s3.glob(file_path)
        all_files = [f"s3://{file}" for file in files]
        if len(all_files) == 0:
             return None

        query = f"""
            SELECT * FROM read_parquet([{', '.join([f"'{file}'" for file in all_files])}])
            """

    else:
        return None
    
    try:
        df = con.execute(query).df()
    except Exception as e:
        print(f"DuckDB Execution Error: {e}")
        return None
    
    if columns_of_interest:
        df = df[columns_of_interest]
    return df


def question1(s3_bucket_path, metadata_parquet_path, start_year, end_year):
    """
    Answers: Which high court disposed the most cases from 2019 - 2022?
    """
    courts_of_interest = "*"
    columns_of_interest = ['court']
    df = fetch_data(s3_bucket_path, metadata_parquet_path, courts_of_interest=courts_of_interest, start_year=start_year, end_year=end_year, columns_of_interest=columns_of_interest)
    if df is None:
        return "Data not available."
    
    court_counts = df.groupby('court').size().reset_index(name='count')
    if court_counts.empty:
        return "No data found for the specified years."
    
    most_cases_court = court_counts.sort_values(by='count', ascending=False).iloc[0]['court']
    return most_cases_court


def question2(s3_bucket_path, metadata_parquet_path, court_code):
    """
    Answers: What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?
    """
    df = fetch_data(s3_bucket_path, metadata_parquet_path, court_code=court_code, columns_of_interest=['decision_date', 'date_of_registration', 'year'])

    if df is None or df.empty:
        return "Data not available."

    df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
    df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')

    df = df.dropna(subset=['decision_date', 'date_of_registration', 'year'])

    df['delay_days'] = (df['date_of_registration'] - df['decision_date']).dt.days

    df = df.dropna(subset=['delay_days'])

    X = df[['year']]
    y = df['delay_days']
    if len(X) < 2:
         return "Not enough data to perform regression."

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    return slope


def question3(s3_bucket_path, metadata_parquet_path, court_code, image_format='webp'):
    """
    Answers: Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters
    """
    df = fetch_data(s3_bucket_path, metadata_parquet_path, court_code=court_code, columns_of_interest=['decision_date', 'date_of_registration', 'year'])

    if df is None or df.empty:
        return "Data not available."

    df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
    df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')

    df = df.dropna(subset=['decision_date', 'date_of_registration', 'year'])

    df['delay_days'] = (df['date_of_registration'] - df['decision_date']).dt.days

    df = df.dropna(subset=['delay_days'])
    if df.empty:
        return "Not enough data to generate plot."

    if len(df) < 2:
        return "Not enough data to generate plot."


    X = df[['year']]
    y = df['delay_days']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='year', y='delay_days', data=df)
    sns.lineplot(x=df['year'], y=y_pred, color='red')
    plt.xlabel('Year')
    plt.ylabel('Delay (Days)')
    plt.title('Delay vs. Year with Regression Line')

    img = io.BytesIO()
    plt.savefig(img, format=image_format)
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')

    if len(img_base64) > 100000:
        return "Plot size exceeds 100,000 characters."

    return f"data:image/{image_format};base64,{img_base64}"


s3_bucket_path = 's3://indian-high-court-judgments/'
metadata_parquet_path = 'metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1'
courts_of_interest = '33_10'
start_year = 2019
end_year = 2022
court_code = '33_10'
image_format = 'webp'

output = {
    "Which high court disposed the most cases from 2019 - 2022?": question1(s3_bucket_path, metadata_parquet_path, start_year, end_year),
    "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": question2(s3_bucket_path, metadata_parquet_path, court_code),
    "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": question3(s3_bucket_path, metadata_parquet_path, court_code, image_format)
}

print(output)
