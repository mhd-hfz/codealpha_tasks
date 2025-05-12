import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Sample data pools
users = [f"user_{i}" for i in range(1, 11)]
songs = [f"song_{j}" for j in range(1, 21)]
genres = ['Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Classical']

data = []

for _ in range(500):  # 500 rows
    user = random.choice(users)
    song = random.choice(songs)
    genre = random.choice(genres)
    timestamp = datetime(2024, random.randint(1, 12), random.randint(1, 28))
    duration = random.randint(120, 300)  # seconds
    replayed = random.choices([0, 1], weights=[0.7, 0.3])[0]  # 30% chance of replay
    
    data.append([user, song, genre, timestamp, duration, replayed])

df = pd.DataFrame(data, columns=[
    "user_id", "song_id", "genre", "timestamp", "song_duration", "replayed_within_30_days"
])

df.to_csv("music_recommendation_dataset.csv", index=False)
