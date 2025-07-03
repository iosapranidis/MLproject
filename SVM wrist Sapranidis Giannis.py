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
#Σε αυτό το κομμάτι διαβάζονται τα csv αρχεία για τον καρπό και ενώνονται σε μια λίστα

print('please wait...')

csv_files = glob.glob('D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/ProFiles3/*.csv')
list1 = []
for csv_file in csv_files:
    df = []
    df = pd.read_csv(csv_file)
    proi = []
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
    with open(f"D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/Fakelos3/finalcsv{i}(1).csv",'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(ltime)
    finallist.append(ltime)    

#Τέλος κατασκευάζουμε ένα ξεχωριστό csv αρχείο για το κάθε άτομο με τις νέες τιμές.


print('data loading for wrist complete!')
input('press enter to continue')
print()


#Ενώνουμε τα csv αρχεία σε ένα


csv_files11 = glob.glob('D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/Fakelos3/*.csv')
Finals2 = pd.DataFrame()
for csv_file in csv_files11:
    df = pd.read_csv(csv_file)
    Finals2 = pd.concat([Finals2, df])

print('Start')
print()
print('style of data for wrist')
print()
print(Finals2.head())
print()
print('(number of objects, columns) = ',Finals2.shape)
print()

#Διαβάζουμε τις τιμές για κάθε δραστηριότητα και την αντίστοιχη δραστηριότητα

data = Finals2.values
X = data[:,:-1]
y = data[:,6]

for i in range(10):
    y[i*1241:i*1241+170] = [0] #walking = 0
    y[i*1241+170:i*1241+350] = [1] #standing = 1
    y[i*1241+350:i*1241+530] = [2] #jogting = 2
    y[i*1241+530:i*1241+710] = [3] #sitting = 3
    y[i*1241+710:i*1241+890] = [4] #biking = 4
    y[i*1241+890:i*1241+1070] = [5] #upstairs = 5
    y[i*1241+1070:i*1241+1241] = [6] #downstairs = 6
X_test = []
y_test = []
for j in X:
    X_test.append(j)
for k in y:
    y_test.append(k)


#Αλλάξαμε τις τιμές του y αντιστοιχίζοντας την κάθε δραστηριότητα σε έναν αριθμό
#για να δημιουργήσουμε τα διανύσματα στόχους

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Σε αυτό το κομμάτι διαβάζονται τα csv αρχεία για την αριστερή τσέπη και ενώνονται σε μια λίστα
print('please wait')

csv_files2 = glob.glob('D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/ProFiles/*.csv')
list2 = []
for csv_file in csv_files2:
    df = []
    df = pd.read_csv(csv_file)
    proi = []
    proi = df.values.tolist()
    list2.append(proi)


open_file2 = list2

#Κατασκευάζουμε τα παράθυρα 20 δευτερολέπτων που αλλάζουν με βήμα 1 δευτερόλεπτο
#και επειδή η συχνότητα δειγματοληψίας είναι 50 δείγματα το δευτερόλεπτο άρα
#το μέγεθος του παραθύρου θα είναι 1000 δείγματα με βήμα αλλαγής 50 δειγμάτων.
#Βάσει αυτού υπολογίζουμε μέσα στο παράθυρο Μέσο όρο, Τυπική απόκλιση, Ασυμμετρία
#κατανομής, Μέγιστη τιμή, Ελάχιστη τιμή, Διαφορά μέγιστης ελάχιστης τιμής.


fields = ['Mean','Std','Skewness','Max','Min','Max-Min','Activity']
finallist2 = []

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
            l22.append(open_file2[i][k][0])
            if open_file2[i][k][1] == 'walking':
                w += 1
            elif open_file2[i][k][1] == 'sitting':
                si += 1
            elif open_file2[i][k][1] == 'standing':
                st += 1
            elif open_file2[i][k][1] == 'jogging':
                jo += 1
            elif open_file2[i][k][1] == 'biking':
                bi += 1
            elif open_file2[i][k][1] == 'upstairs':
                up += 1
            elif open_file2[i][k][1] == 'downstairs':
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
    finallist2.append(ltime)    

#Τέλος κατασκευάζουμε ένα ξεχωριστό csv αρχείο για το κάθε άτομο με τις νέες τιμές.

print('data loading for left pocket complete!')
input('press enter to continue')
print()


#Ενώνουμε τα csv αρχεία σε ένα


csv_files12 = glob.glob('D:/Desktop/Postgrad/Χαρίσης/Εργασία Μηχανική Μάθηση/Fakelos/*.csv')
Finals = pd.DataFrame()
for csv_file in csv_files12:
    df = pd.read_csv(csv_file)
    Finals = pd.concat([Finals, df])

#print('Start')
#print()
print('style of data for left pocket')
print()
print(Finals.head())
print()
print('(number of objects, columns) = ',Finals.shape)
print()
print('---------------------------------------------')


#Διαβάζουμε τις τιμές για κάθε δραστηριότητα και την αντίστοιχη δραστηριότητα

data2 = Finals.values
X = data2[:,:-1]
y = data2[:,6]

for i in range(10):
    y[i*1241:i*1241+170] = [0] #walking = 0
    y[i*1241+170:i*1241+350] = [1] #standing = 1
    y[i*1241+350:i*1241+530] = [2] #jogting = 2
    y[i*1241+530:i*1241+710] = [3] #sitting = 3
    y[i*1241+710:i*1241+890] = [4] #biking = 4
    y[i*1241+890:i*1241+1070] = [5] #upstairs = 5
    y[i*1241+1070:i*1241+1241] = [6] #downstairs = 6
X_train = []
y_train = []
for j in X:
    X_train.append(j)
for k in y:
    y_train.append(k)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Αλλάξαμε τις τιμές του y αντιστοιχίζοντας την κάθε δραστηριότητα σε έναν αριθμό
#για να δημιουργήσουμε τα διανύσματα στόχους
input('press enter to continue')

print()
print('method used for training/testing : svm with rbf kernel for γ=0.1 and C=2')
print()
print('---------------------------------------------')
print()


#Κατασκευάζουμε τη μέθοδο για το svm.

acc = []
fcm = [[0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0]]

print('len(X_train)=',len(X_train),' left pocket data')
print('len(X_test)=',len(X_test),' wrist data')
print('len(y_train)=',len(y_train))
print('len(y_test)=',len(y_test))
print()
scaler.fit(X_train)
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)
rbf = svm.SVC(kernel='rbf', gamma=0.1, C=2).fit(X_train, y_train)
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


#Ελέγχουμε για άλλες τιμές των C και γ από 1 έως 10 και από 0.1 έως 1 αντίστοιχα
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
print('End')
print('Thanks for watching')

