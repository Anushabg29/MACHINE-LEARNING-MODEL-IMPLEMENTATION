
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("spam.csv")

# Show dataset size (DEBUG)
print("Rows loaded:", len(data))

# Rename columns
data.columns = ["label", "message"]

# Remove missing values
data.dropna(inplace=True)

# Encode labels
data["label"] = data["label"].str.lower().str.strip()
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# Final safety check
if data.empty:
    raise ValueError("Dataset is EMPTY. Check spam.csv file.")

# Split dataset
X = data["message"]
y = data["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, zero_division=0))



# Test custom messages
test_messages = [
    "Congratulations you won free cash",
    "Can we meet tomorrow?"
]

test_vec = vectorizer.transform(test_messages)
predictions = model.predict(test_vec)

print("\nCustom Predictions:")
for msg, pred in zip(test_messages, predictions):
    print(f"\nMessage: {msg}")
    print("Prediction:", "SPAM" if pred == 1 else "HAM")
