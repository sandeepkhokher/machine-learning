# machine Learning Program with Persistence Model

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

music_data = pd.read_csv("music.csv")
x = music_data.drop(columns=['genre'])
y = music_data['genre']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 

# Create  and dump model to a file and the  use the same for prediction
#model = DecisionTreeClassifier()
#model.fit(x_train, y_train)

#joblib.dump(model, 'music-recommender.joblib')
#model = joblib.load('music-recommender.joblib')
predictions = model.predict(x_test)

print(x_test)
print(predictions)
score = accuracy_score(y_test, predictions)
print(score)
#print(df.describe())