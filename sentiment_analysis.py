import pandas as pd  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score

# Load the dataset  
dataset_path = 'exampledataset2.csv'  # Adjust the path if necessary  
df = pd.read_csv(dataset_path)

# Specify the columns  
text_column = 'text'    
sentiment_column = 'sentiment'  

# Split the dataset into features and target  
X = df[text_column]  
y = df[sentiment_column]

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data  
vectorizer = CountVectorizer()  
X_train_vectorized = vectorizer.fit_transform(X_train)  
X_test_vectorized = vectorizer.transform(X_test)

# Initialize the SVM classifier  
svm_classifier = SVC()

# Train the model  
svm_classifier.fit(X_train_vectorized, y_train)

# Make predictions  
y_pred = svm_classifier.predict(X_test_vectorized)

# Evaluate the model's performance  
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy  
print(f'Accuracy of the SVM model: {accuracy:.2f}')