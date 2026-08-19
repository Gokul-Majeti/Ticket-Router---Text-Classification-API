import pandas as pd

df = pd.read_csv('data/bbc-text.csv')

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))

print("\n--- Class balance ---")
print(df['category'].value_counts())

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Sample row ---")
print("Category:", df.iloc[0]['category'])
print("Text preview:", df.iloc[0]['text'][:300])

print("\n--- Text length stats (word count) ---")
df['word_count'] = df['text'].str.split().str.len()
print(df['word_count'].describe())
