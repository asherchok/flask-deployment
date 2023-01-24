import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('dataset.csv')
df.rename(columns = {'Nacionality':'Nationality'}, inplace = True)

le = LabelEncoder()
df['Target'] = le.fit_transform(df['Target'])

x = df[['Tuition fees up to date','Scholarship holder', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']]
y = df['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))
pickle_model = pickle.load(open('model.pkl', 'rb'))

print(pickle_model.predict([[1,0,0,0,0,0]]))