import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('xxxxxx.csv')

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply preprocessing
data['text'] = data['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
