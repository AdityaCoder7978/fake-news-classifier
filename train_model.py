import pandas as pd
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from sklearn.utils import resample

# Download NLTK resources
download('stopwords')
download('wordnet')

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels: 0 = fake, 1 = real
df_fake['label'] = 0
df_true['label'] = 1

# Combine both datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Optional: Drop unnecessary columns (if present)
df = df[['text', 'label']]  # Keep only text and label columns
df = df.dropna()

# Balance dataset (if needed)
fake = df[df['label'] == 0]
real = df[df['label'] == 1]
real_upsampled = resample(real, replace=True, n_samples=len(fake), random_state=42)
df_balanced = pd.concat([fake, real_upsampled])

# Preprocess text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

df_balanced['cleaned'] = df_balanced['text'].apply(clean_text)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_balanced['cleaned']).toarray()
y = df_balanced['label']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("\n✅ Model and vectorizer saved successfully!")
