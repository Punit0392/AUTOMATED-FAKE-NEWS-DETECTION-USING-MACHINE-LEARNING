import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
fake = pd.read_csv("Dataset/Fake.csv")
real = pd.read_csv("Dataset/True.csv")

# Add labels
fake['label'] = 'FAKE'
real['label'] = 'REAL'

# Combine datasets
data = pd.concat([fake, real])
data = data[['text', 'label']]  # Use only the article text

# Train/test split
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Train classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Predict and evaluate
y_pred = model.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score * 100, 2)}%")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))