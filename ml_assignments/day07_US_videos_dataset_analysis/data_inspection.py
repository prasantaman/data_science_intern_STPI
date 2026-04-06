import pandas as pd
df = pd.read_csv("USvideos.csv")
print(df.head())
print(df.columns)
print(df.isnull().sum())
df['description'] = df['description'].fillna("")
print(df.isnull().sum())
df['title_length'] = df['title'].apply(len)
df.head()
print(df.info())
df['publish_time'] = pd.to_datetime(df['publish_time'])
from datetime import datetime
import pandas as pd

df['publish_time'] = pd.to_datetime(df['publish_time'])

df['days_since_publish'] = (pd.Timestamp.now(tz='UTC') - df['publish_time']).dt.days
print(df[['publish_time', 'days_since_publish']].head())
threshold = df['views'].quantile(0.75)

df['viral'] = df['views'].apply(lambda x: 1 if x > threshold else 0)
print(df['viral'].value_counts())

print(df.head())
print(df.describe())

print(df.info())