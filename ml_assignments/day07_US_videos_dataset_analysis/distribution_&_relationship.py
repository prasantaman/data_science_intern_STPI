import pandas as pd
df = pd.read_csv("USvideos.csv")
df['description'] = df['description'].fillna("")
df['title_length'] = df['title'].apply(len)
df['publish_time'] = pd.to_datetime(df['publish_time'])
from datetime import datetime
import pandas as pd
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['days_since_publish'] = (pd.Timestamp.now(tz='UTC') - df['publish_time']).dt.days
print(df[['publish_time', 'days_since_publish']].head())


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Histogram of views
plt.figure()
plt.hist(df['views'], bins=50)
plt.title("Distribution of Views")
plt.xlabel("Views")
plt.ylabel("Frequency")
plt.show()

# 2. Scatter plot (likes vs views)
plt.figure()
plt.scatter(df['likes'], df['views'])
plt.title("Likes vs Views")
plt.xlabel("Likes")
plt.ylabel("Views")
plt.show()

# 3. Correlation matrix
corr = df[['views', 'likes', 'dislikes', 'comment_count', 
           'title_length', 'days_since_publish']].corr()

print("\nCorrelation Matrix:\n", corr)

# Heatmap
plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix Heatmap")
plt.show()