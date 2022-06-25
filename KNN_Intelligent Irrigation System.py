import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Input Dataset
dataset = pd.read_csv('Data Intelligent Irrigation System.csv')
dataset.head()
x = dataset.iloc[:, [1, 2]].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

#Membagi Dataset ke Data Training dan Data Testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(x_train)
len(x_train)
len(x)
len(x_test)
print(y_train)
len(y_train)
len(y_test)

#Scaling Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print(x_test)

#Memanggil Function KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

#Menentukan Prediksi
y_pred = classifier.predict(x_test)

#Evaluasi dan Validasi
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report
akurasi = classification_report(y_test,y_pred)
print(akurasi)

from sklearn.metrics import accuracy_score
akurasi = accuracy_score(y_test,y_pred) 
print("Tingkat Akurasi :%d persen"%(akurasi*100))

#Visualisasi Data (Grafik)
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01),
                     np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 0].max()+1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Klasifikasi Data dengan K-Nearest Neighbor')
plt.xlabel('Moisture')
plt.ylabel('Temp')
plt.legend()
plt.show()