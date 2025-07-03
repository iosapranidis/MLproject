import numpy as np
import pandas as pd
import glob
from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import csv
import os.path
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#READ PART
#Σε αυτό το κομμάτι διαβάζονται τα csv αρχεία και ενώνονται σε μια λίστα

print('please wait...')

csv_files = glob.glob('D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/ProFiles/*.csv')
list1 = []
for csv_file in csv_files:
    df = []
    df = pd.read_csv(csv_file)
    proi = df.values.tolist()
    list1.append(proi)


open_file = list1

#Κατασκευάζουμε τα παράθυρα 20 δευτερολέπτων που αλλάζουν με βήμα 1 δευτερόλεπτο
#και επειδή η συχνότητα δειγματοληψίας είναι 50 δείγματα το δευτερόλεπτο άρα
#το μέγεθος του παραθύρου θα είναι 1000 δείγματα με βήμα αλλαγής 50 δειγμάτων.
#Βάσει αυτού υπολογίζουμε μέσα στο παράθυρο Μέσο όρο, Τυπική απόκλιση, Ασυμμετρία
#κατανομής, Μέγιστη τιμή, Ελάχιστη τιμή, Διαφορά μέγιστης ελάχιστης τιμής.


fields = ['Mean','Std','Skewness','Max','Min','Max-Min','Activity']
finallist = []

for i in range(0,10):
    ltime = []
    for j in range(0,1241):
        w = 0
        si = 0
        st = 0
        jo = 0
        bi = 0
        up = 0
        down = 0
        l22 = []
        for k in range(50*j,50*j+1000):
            l22.append(open_file[i][k][0])
            if open_file[i][k][1] == 'walking':
                w += 1
            elif open_file[i][k][1] == 'sitting':
                si += 1
            elif open_file[i][k][1] == 'standing':
                st += 1
            elif open_file[i][k][1] == 'jogging':
                jo += 1
            elif open_file[i][k][1] == 'biking':
                bi += 1
            elif open_file[i][k][1] == 'upstairs':
                up += 1
            elif open_file[i][k][1] == 'downstairs':
                down += 1
        if w > st:
            act = 'walking'
        elif st > jo:
            act = 'standing'
        elif jo > si:
            act = 'jogging'
        elif si > bi:
            act = 'sitting'
        elif bi > up:
            act = 'biking'
        elif up > down:
            act = 'upstairs'
        else:
            act = 'downstairs'
        a = np.array(l22)
        d = pd.DataFrame(l22)
        l = [np.mean(a),np.std(a),float(d.skew().values),max(a),min(a),max(a)-min(a),act]
        ltime.append(l)
    with open(f"D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/Fakelos/finalcsv{i}.csv",'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(ltime)
    finallist.append(ltime)    

#Τέλος κατασκευάζουμε ένα ξεχωριστό csv αρχείο για το κάθε άτομο με τις νέες τιμές.


print('data loading complete!')
input('press enter to continue')
print()


#Ενώνουμε τα csv αρχεία σε ένα


csv_files11 = glob.glob('D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/Fakelos/*.csv')
Finals = pd.DataFrame()
for csv_file in csv_files11:
    df = pd.read_csv(csv_file)
    Finals = pd.concat([Finals, df])

print('Start')
print()
print('style of data')
print()
print(Finals.head())
print()
print('(number of objects, columns) = ',Finals.shape)
print()
print('---------------------------------------------')


#Διαβάζουμε τις τιμές για κάθε δραστηριότητα και την αντίστοιχη δραστηριότητα
#Χωρίζουμε το δείγμα σε 10 ίσα κομμάτια χωρίς ανακάτεμα.


data = Finals.values
X = data[:,:-1]
y = data[:,6]
kf = KFold(n_splits=10)
kf.get_n_splits(X)

print()
print('method used for training\testing : ',kf)
print()
print('---------------------------------------------')
print()


#Κατασκευάζουμε την επαναληπτική μέθοδο για τους knn γείτονες.
#Χωρίζουμε σε test και train set το σύνολό μας που αντιστοιχεί σε 1 άτομο να
#είναι το test set και τα υπόλοιπα 9 να είναι το train set. (L.O.S.O.)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=13)
acc = []

fcm = [[0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0]]
for i, (train_index, test_index) in enumerate(kf.split(X)):
     input('press enter to continue')
     print()
     X_train = []
     X_test = []
     y_train = []
     y_test = []
     print(f"Fold {i+1}:")
     print()
     print(f"  Train: index={train_index}")
     print(f"  Test:  index={test_index}")
     print()
     for i in train_index:
         X_train.append(X[i])
         y_train.append(y[i])
     for i in test_index:
         X_test.append(X[i])
         y_test.append(y[i])
     print('len(X_train)=',len(X_train))
     print('len(X_test)=',len(X_test))
     print('len(y_train)=',len(y_train))
     print('len(y_test)=',len(y_test))
     print()
     scaler.fit(X_train)
     X_train = scaler.transform(X_train) 
     X_test = scaler.transform(X_test)
     classifier.fit(X_train, y_train)
     y_pred = classifier.predict(X_test)
     cm = confusion_matrix(y_test, y_pred)
     print()
     print('confusion matrix:')
     print()
     print(cm)
     print()
     print('classification report:')
     print()
     print(classification_report(y_test, y_pred))
     print('---------------------------------------------')
     acc.append(sklearn.metrics.accuracy_score(y_test, y_pred))
     

     for m in range(len(cm)):
         for n in range(len(cm[0])):
             fcm[m][n] += cm[m][n]

#Παίρνουμε τους πίνακες σύγχισης για κάθε περίπτωση.


#Ελέγχουμε για άλλες τιμές των k κοντινότερων γειτόνων ποια θα μας δώσει καλύτερα
#αποτελέσματα.
     
print('learning and testing for best value of nn is complete.')
print()
print('final confusion matrix:')
print()
fcm = np.array(fcm)
print(fcm)
print()
print('final accuracy:',np.mean(acc))
print()
'''
print('checking for other values...')
print()
input('press enter to continue')

mainerror = []
error = []
for j in range(1, 40, 2):
    for i, (train_index, test_index) in enumerate(kf.split(X)): 
        knn = KNeighborsClassifier(n_neighbors = j)
        pred_j = []
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for w in train_index:
            X_train.append(X[w])
            y_train.append(y[w])
        for w in test_index:
            X_test.append(X[w])
            y_test.append(y[w])
        scaler.fit(X_train)
        X_train = scaler.transform(X_train) 
        X_test = scaler.transform(X_test)
        knn.fit(X_train, y_train)
        pred_j = knn.predict(X_test)
        error.append(np.mean(pred_j != y_test))
    mainerror.append(np.mean(error))
    error = []
    
print()
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40, 2), mainerror, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
 '''   

print('End')
print('Thanks for watching')


                
