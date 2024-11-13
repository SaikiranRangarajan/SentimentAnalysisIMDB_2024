import nltk
import sklearn
import textstat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


lemmatizer = nltk.stem.WordNetLemmatizer()
# First, we get the stopwords list from nltk
stopwords=set(nltk.corpus.stopwords.words('english'))
# We can add more words to the stopword list, like punctuation marks
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
filename = r'C:\Users\c23097138\Downloads\bbc-text.csv'
df = pd.read_csv(filename, encoding='unicode_escape')
dataset_file = []
set_categories = set() 
for index, row in df.iterrows():
    dataset_file.append(row.text)
    set_categories.add(row.category)
# Step 1: Create a dictionary to hold string keys and integer values
categories = df['category'].unique()
my_map = {category: idx for idx, category in enumerate(categories)}

# Step 2 : Create Features
#Feature 1 : Perform Text Preprecessing using TfidfVectorizer instance. 
# This function uses a weighted scheme called [tf-idf](term frequency-inverse document frequency) which basically penalizes words that are repeated across many documents (e.g. frequent words such as "the" or "a").
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(dataset_file)
tf_idf_matrix_array = tfidf_matrix.toarray()

#Feature 2: Apply sentiment analysis and store the results in a new DataFrame column
text_data = df['text'].fillna('')  # Handle missing values by replacing them with an empty string
# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['Sentiment_Score'] = text_data.apply(lambda text: sia.polarity_scores(text)['compound'])

#Feature 3 : Calculate Flesch Reading Ease (FRES) score. The Flesch Reading Ease Score is the score of readability which indicates how difficult a passage in English is to understand.
df['Flesch_Reading_Ease'] = text_data.apply(textstat.flesch_reading_ease)
#Feature 4 : Calculate ARI (Automated Readability Index)  score. The ARI (Automated Readability Index) score is the grade level needed to comprehend the text.
df['Automated_Readability_Index'] = text_data.apply(textstat.automated_readability_index)
#Feature 5 : Calculate Coleman-Liau Index score. The Coleman-Liau Index score is the grade level needed to comprehend the text. Uses character count instead of syllables.
df['Coleman_Liau_Index'] = text_data.apply(textstat.coleman_liau_index)
#Feature 6 : Calculate Dale–Chall readability score. The Dale–Chall readability score is the grade level needed to comprehend the text, based on familiar vs. unfamiliar words.
df['Dale_Chall_Score'] = text_data.apply(textstat.dale_chall_readability_score)
#Feature 7 : Calculate Difficult Words. The Difficult Words Counts the number of complex words.
df['Difficult_Words'] = text_data.apply(textstat.difficult_words)
#Feature 8 : Calculate Linsear Write Formula score. The Linsear Write Formula score is the grade level needed to comprehend the text.
df['Linsear_Write_Score'] = text_data.apply(textstat.linsear_write_formula)

# Step 3 : Combine features
additional_features = df[['Sentiment_Score', 'Flesch_Reading_Ease', 'Automated_Readability_Index',
                          'Coleman_Liau_Index', 'Dale_Chall_Score', 'Difficult_Words', 'Linsear_Write_Score']].values
labels = df['category'].map(my_map).values
print(labels)
# Step 4 :Combine TF-IDF and additional features
full_features = np.hstack((tfidf_matrix.toarray(), additional_features))

# Step 5: Assuming full_features and labels are defined earlier perform training of data with 80% Train, 20% Temporary (Test + Dev)
X_train, X_temp, y_train, y_temp = train_test_split(full_features, labels, test_size=0.2, random_state=42)
# Split 20% Temporary into 10% Test, 10% Dev
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Step 6: Initialize and train the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Evaluate on Dev set
y_dev_pred = rf_model.predict(X_dev)
# Evaluate on Test set
y_test_pred = rf_model.predict(X_test)

# Step 7: Method that Returns a list of keys from the dictionary that have the specified target value.
def get_keys_by_value(dictionary, target_value):
    """
    Returns a list of keys from the dictionary that have the specified target value.

    :param dictionary: dict, the input dictionary
    :param target_value: the value to search for
    :return: list of keys that have the target value
    """
    keys = [key for key, value in dictionary.items() if value == target_value]
    return keys

# Step 8: Store the final result as appropriate text values instead of binaries.
final_predicted_result = []
for pred_value in y_test_pred:
  result = get_keys_by_value(my_map, pred_value)
  final_predicted_result.append(result)
# Step 9: Display the predicted data
print("Predicted Data is", final_predicted_result)

# Step 10: Evaluate the model for Random Forest Classifier
print("Accuracy for Random Forest Classifier is:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report 2:\n", classification_report(y_test, y_test_pred))

# Step 10: Plot and Display the graph to show the difference
plt.plot(range(len(y_test)), y_test, color='blue')
plt.plot(range(len(y_test_pred)), y_test_pred, color='red')
plt.xlabel('Actual Positive Score')
plt.ylabel('Predicted Positive Score')
plt.title('Actual vs Predicted Score')
plt.show()