import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# Load dataset
data = pd.read_csv('C:\\Users\\Inbaraj\\OneDrive\\Desktop\\spam Email project\\spam_ham_dataset.csv', encoding='latin-1')
data = data[['label', 'text']]  
data.columns = ['label', 'message']

# Preprocessing: Convert labels to binary
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Create a model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'spam_model.pkl')

# Evaluate the model
predictions = model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))