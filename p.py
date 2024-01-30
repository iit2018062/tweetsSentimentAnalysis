import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# da taset
df = pd.read_csv('Tweets.csv')

# Dropping the 'textID' column
df.drop(columns=['textID'], inplace=True)

# Function to clean and preprocess text
def clean_and_preprocess_text(text):
    # Convert non-string to string
    text = str(text)
    # Lowercasing the text
    text = text.lower()
    # Removing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Removing special characters and numbers
    text = re.sub(r'\W+|\d+', ' ', text)
    # Tokenization
    tokens = word_tokenize(text)
    #     print("token")
    #     print(tokens)
    # Removing stop words and stemming
    stop_words = set(stopwords.words('english'))
    #     print("after removing stop words")
    #     print(stop_words)
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Rejoining the tokens into a string
    return " ".join(filtered_tokens)

# Applying the cleaning function to 'text' and 'selected_text' columns
df['text'] = df['text'].apply(clean_and_preprocess_text)
df['selected_text'] = df['selected_text'].apply(clean_and_preprocess_text)

# Display
print(df.head(2))
# Set plot style
sns.set(style="whitegrid")

# Analyzing Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', data=df)
plt.title('Distribution of Sentiments_in_Tweets')
plt.show()
# Analyzing the Relationship Between Tweet Length and Sentiment
df['text_length'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment', y='text_length', data=df)
plt.title('Tweet Length vs Sentiment')
plt.show()
# Visualizing Common Words in Each Sentiment Category
for sentiment in df['sentiment'].unique():
    subset = df[df['sentiment'] == sentiment]
    text = " ".join(subset['text'].tolist())

    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Most Common Words in {sentiment.capitalize()} Tweets')
    plt.axis("off")
    plt.show()
df.columns
# begin the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# Apply TF-IDF to the 'text' column to have the feature matrix
X_tfidf = tfidf_vectorizer.fit_transform(df['selected_text'])
print(X_tfidf)
print("*****")

# Encoding the 'sentiment' column
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['sentiment'])
print(label_encoder)
print("*****")

# create additional features
df['tweet_length'] = df['text'].apply(lambda x: len(x.split()))


#  include 'tweet_length' as a feature:
from scipy.sparse import hstack
X_final = hstack((X_tfidf, df[['tweet_length']].values.astype(float)))
# Actually the model performance fall behind around 10% when use tweet length as a feature so not use it.
print(X_final)
df.head(2)
# checking encoding
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(label_mapping)
# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the models
logistic_model = LogisticRegression(max_iter=1000)
# Train the Logistic Regression model and evaluate on validation set
logistic_model.fit(X_train, y_train)
logistic_train_preds = logistic_model.predict(X_train)
logistic_val_preds = logistic_model.predict(X_val)
logistic_train_accuracy = accuracy_score(y_train, logistic_train_preds)
logistic_val_accuracy = accuracy_score(y_val, logistic_val_preds)
print("Logistic Regression Training Accuracy:", logistic_train_accuracy)
print("Logistic Regression Validation Accuracy:", logistic_val_accuracy)
logistic_test_preds = logistic_model.predict(X_test)
logistic_test_report = classification_report(y_test, logistic_test_preds)
print("Logistic Regression Test Classification Report")
print(logistic_test_report)