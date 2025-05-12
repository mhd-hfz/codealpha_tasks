import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("music_recommendation_dataset.csv")

# # Quick preview
# print(df.head())
# print(df.info())

le_user = LabelEncoder()
le_song = LabelEncoder()
le_genre = LabelEncoder()

df['user_id'] = le_user.fit_transform(df['user_id'])
df['song_id'] = le_song.fit_transform(df['song_id'])
df['genre'] = le_genre.fit_transform(df['genre'])

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract features
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday

# Add new derived features
df['is_weekend'] = (df['weekday'] >= 5).astype(int)
df['short_song'] = (df['song_duration'] < 180).astype(int)

# Drop original timestamp
df.drop(columns=['timestamp'], inplace=True)

# Define features and label
X = df.drop(columns=['replayed_within_30_days'])
y = df['replayed_within_30_days']

# #See how imbalanced the target variable is:
# print(df['replayed_within_30_days'].value_counts())


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
model = RandomForestClassifier(class_weight='balanced',random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
