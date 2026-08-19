import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

df = pd.read_csv('data/bbc-text.csv')

X = df['text']
y = df['category']

# stratify=y ensures the train/test split keeps the same class proportions
# (important since our classes aren't perfectly balanced: 386 to 511)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# TF-IDF vectorizer settings explained:
# - stop_words='english': drop common words like "the", "is", "and"
# - max_features=5000: cap vocabulary size to the 5000 most informative words
#   (keeps the model small and fast, avoids overfitting on rare words)
# - ngram_range=(1,2): use both single words AND two-word phrases
#   (e.g. "prime minister" carries more signal as a phrase than "prime" + "minister" separately)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)  # learn vocabulary + transform train data
X_test_vec = vectorizer.transform(X_test)         # reuse SAME vocabulary on test data (no leakage)

print(f"Feature matrix shape: {X_train_vec.shape}")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n--- Classification report ---")
print(classification_report(y_test, y_pred))

print("--- Confusion matrix ---")
print("(rows = true label, columns = predicted label)")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# Save the trained model + vectorizer so we can load them later in the API
joblib.dump(model, 'models/model.joblib')
joblib.dump(vectorizer, 'models/vectorizer.joblib')
print("\nSaved model and vectorizer to models/")
