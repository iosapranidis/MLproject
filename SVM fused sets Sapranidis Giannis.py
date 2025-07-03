import numpy as np
import pandas as pd
import glob
from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import csv
import sklearn
from sklearn import svm
import os.path
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


#READ PART
#Σε αυτό το κομμάτι διαβάζονται τα csv αρχεία και ενώνονται σε μια λίστα

print('please wait...')

csv_files1 = glob.glob('D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/ProFiles1/*.csv')
list1 = []
for csv_file in csv_files1:
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
        #up = 0
        #down = 0
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
            #elif open_file[i][k][1] == 'upstairs':
            #    up += 1
            #elif open_file[i][k][1] == 'downstairs':
            #    down += 1
        if w > si:
            act = 'walking'
        elif si > st:
            act = 'sitting'
        elif st > jo:
            act = 'standing'
        elif jo > bi:
            act = 'jogging'
        else:
            act = 'biking'
        #elif up > down:
        #    act = 'upstairs'
        #else:
        #    act = 'downstairs'
        a = np.array(l22)
        d = pd.DataFrame(l22)
        l = [np.mean(a),np.std(a),float(d.skew().values),max(a),min(a),max(a)-min(a),act]
        ltime.append(l)
    with open(f"D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/Fakelos1/finalcsv{i}.csv",'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(ltime)
    finallist.append(ltime)    

#Τέλος κατασκευάζουμε ένα ξεχωριστό csv αρχείο για το κάθε άτομο με τις νέες τιμές.


print('data loading complete!')
input('press any button to continue')
print()


#Ενώνουμε τα csv αρχεία σε ένα


csv_files12 = glob.glob("D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/Fakelos1/*.csv")
Finals1 = pd.DataFrame()
for csv_file in csv_files12:
    df = pd.read_csv(csv_file)
    Finals1 = pd.concat([Finals1, df])

print('Start')
print()
print('style of data')
print()
print(Finals1.head())
print()
print('(number of objects, columns) = ',Finals1.shape)
print()
print('---------------------------------------------')


#Διαβάζουμε τις τιμές για κάθε δραστηριότητα και την αντίστοιχη δραστηριότητα
#Χωρίζουμε το δείγμα σε 10 ίσα κομμάτια χωρίς ανακάτεμα.

data = Finals1.values
X = data[:,:-1]
y = data[:,6]
for i in range(10):
    y[i*1241:i*1241+530] = [0] #walking = 0
    y[i*1241+530:i*1241+710] = [1] #sitting = 1
    y[i*1241+710:i*1241+890] = [2] #standing = 2
    y[i*1241+890:i*1241+1070] = [3] #jogging = 3
    y[i*1241+1070:i*1241+1241] = [4] #biking = 4
    #y[i*1241+890:i*1241+1070] = [5] #upstairs = 5
    #y[i*1241+1070:i*1241+1241] = [6] #downstairs = 6
kf = KFold(n_splits=10)
kf.get_n_splits(X)

#Αλλάξαμε τις τιμές του y αντιστοιχίζοντας την κάθε δραστηριότητα σε έναν αριθμό
#για να δημιουργήσουμε τα διανύσματα στόχους


print()
print('method used for training/testing : svm with rbf kernel for γ=0.1 and C=19')
print()
print('---------------------------------------------')
print()


#Κατασκευάζουμε την επαναληπτική μέθοδο για το svm.
#Χωρίζουμε σε test και train set το σύνολό μας που αντιστοιχεί σε 1 άτομο να
#είναι το test set και τα υπόλοιπα 9 να είναι το train set. (L.O.S.O.)

acc = []
fcm = [[0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0]]
for i, (train_index, test_index) in enumerate(kf.split(X)):
     input('press any button to continue')
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
     rbf = svm.SVC(kernel='rbf', gamma=0.1, C=19).fit(X_train, y_train)
     rbf_pred = rbf.predict(X_test)
     cm = confusion_matrix(y_test, rbf_pred)
     print()
     print('confusion matrix:')
     print()
     print(cm)
     print()
     print('classification report:')
     print()
     print(classification_report(y_test, rbf_pred))
     print('---------------------------------------------')
     acc.append(sklearn.metrics.accuracy_score(y_test, rbf_pred))

     for m in range(len(cm)):
         for n in range(len(cm[0])):
             fcm[m][n] += cm[m][n]
     
#Παίρνουμε τους πίνακες σύγχισης για κάθε περίπτωση.


#Ελέγχουμε για άλλες τιμές των C και γ από 1 έως 20 και από 0.1 έως 1 αντίστοιχα
#για να βρούμε το βέλτιστο ζεύγος τιμών

print('learning and testing for best values of C and Gamma is complete.')
print()
print('final confusion matrix:')
print()
fcm = np.array(fcm)
print(fcm)
print()
print('final accuracy:',np.mean(acc))
print()
'''
print('checking for other values of C...')
print()
input('press any button to continue')


mainerror1 = []
error1 = []
for j in range(1, 21):
    for i, (train_index, test_index) in enumerate(kf.split(X)): 
        pred1_j = []
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
        rbf1 = svm.SVC(kernel='rbf', gamma=0.1, C=j).fit(X_train, y_train)
        pred1_j = rbf1.predict(X_test)
        error1.append(np.mean(pred1_j != y_test))
    mainerror1.append(np.mean(error1))
    error1 = []
    
print()
plt.figure(figsize=(12, 6))
plt.plot(range(1, 21), mainerror1, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate C Value')
plt.xlabel('C Value')
plt.ylabel('Mean Error')
plt.show()


print('checking for other values of Gamma...')
print()
input('press any button to continue')

mainerror2 = []
error2 = []
for j in range(1, 21):
    for i, (train_index, test_index) in enumerate(kf.split(X)): 
        pred2_j = []
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
        rbf2 = svm.SVC(kernel='rbf', gamma=j*0.05, C=19).fit(X_train, y_train)
        pred2_j = rbf2.predict(X_test)
        error2.append(np.mean(pred2_j != y_test))
    mainerror2.append(np.mean(error2))
    error2 = []
    
print()
plt.figure(figsize=(12, 6))
plt.plot(range(1, 21), mainerror2, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate Gamma Value')
plt.xlabel('Gamma Value')
plt.ylabel('Mean Error')
plt.show()
'''
print('End')
print('Thanks for watching')


