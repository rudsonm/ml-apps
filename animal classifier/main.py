from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

classes = {
    1: 'Mammal',
    2: 'Bird',
    3: 'Reptile',
    4: 'Fish',
    5: 'Amphibian',
    6: 'Bug',
    7: 'Invertebrate'
}

data = pd.read_csv('zoo.csv')

y = data['class_type']
x = data.ix[:, 1:17]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(max_depth = 5)

# from sklearn.neighbors import KNeighborsClassifier 
# model = KNeighborsClassifier(n_neighbors = 7)

from sklearn.naive_bayes import GaussianNB 
model = GaussianNB()

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
predictions = model.predict(x_test)

cm = confusion_matrix(y_test, predictions)

# hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail,domestic,catsize
ornitorrinco = [ 0,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1 ]
prediction = model.predict([
    ornitorrinco
])

p_class = classes[prediction[0]]

print(accuracy)
print(p_class)