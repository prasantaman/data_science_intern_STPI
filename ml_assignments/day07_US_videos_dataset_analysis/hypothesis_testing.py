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


from scipy.stats import ttest_ind

# Function to perform t-test
def perform_ttest(feature):
    threshold = df[feature].median()
    
    high = df[df[feature] > threshold]['views']
    low = df[df[feature] <= threshold]['views']
    
    t_stat, p_value = ttest_ind(high, low, equal_var=False)
    
    print(f"\nFeature: {feature}")
    print("p-value:", p_value)

# Run for all features
perform_ttest('likes')
perform_ttest('dislikes')
perform_ttest('comment_count')