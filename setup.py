from helper import datasets
import numpy as np
import pickle
from models import svm, rf, knn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

contagio_ben, contagio_ben_labels, contagio_ben_names = datasets.csv2numpy('data/contagio-ben.csv')
contagio_mal, contagio_mal_labels, contagio_mal_names = datasets.csv2numpy('data/contagio-mal.csv')
google_ben, google_ben_labels, google_ben_names = datasets.csv2numpy('data/google-ben.csv')
virustotal_mal, virustotal_mal_labels, virustotal_mal_names = datasets.csv2numpy('data/virustotal-mal.csv')

def extract(array2d):
    i = 0; 
    j= 1000;
    result = []
    while i != 5000:
        result.append(array2d[i:j,:])
        i = j; 
        j = j + 1000;
    return result

def extractarray(array):
    i = 0;
    j = 1000;
    result = []
    while i != 5000:
        result.append(array[i:j])
        i = j; 
        j = j + 1000;
    return result

contagio_ben_extract = extract(contagio_ben)
contagio_mal_extract = extract(contagio_mal)
google_ben_extract = extract(google_ben)
virustotal_mal_extract = extract(virustotal_mal)
X = np.concatenate((contagio_ben_extract[0],contagio_mal_extract[0], google_ben_extract[0], virustotal_mal_extract[0]), axis=0)
for i in range(1,5):
    x1 = np.concatenate((contagio_ben_extract[i],contagio_mal_extract[i], google_ben_extract[i], virustotal_mal_extract[i]), axis=0)
    X = np.concatenate((X, x1), axis=0)

contagio_ben_labels_extract = extractarray(contagio_ben_labels)
contagio_mal_labels_extract = extractarray(contagio_mal_labels)
google_ben_labels_extract = extractarray(google_ben_labels)
virustotal_mal_labels_extract = extractarray(virustotal_mal_labels)
y = np.concatenate((contagio_ben_labels_extract[0],contagio_mal_labels_extract[0], google_ben_labels_extract[0], virustotal_mal_labels_extract[0]))

for i in range(1,5):
    y1 = np.concatenate((contagio_ben_labels_extract[i],contagio_mal_labels_extract[i], google_ben_labels_extract[i], virustotal_mal_labels_extract[i]))
    y = np.concatenate((y, y1))

contagio_ben_names_extract = extractarray(contagio_ben_names)
contagio_mal_names_extract = extractarray(contagio_mal_names)
google_ben_names_extract = extractarray(google_ben_names)
virustotal_mal_names_extract = extractarray(virustotal_mal_names)
z = np.concatenate((contagio_ben_names_extract[0],contagio_mal_names_extract[0], google_ben_names_extract[0], virustotal_mal_names_extract[0]))

for i in range(1,5):
    z1 = np.concatenate((contagio_ben_names_extract[i],contagio_mal_names_extract[i], google_ben_names_extract[i], virustotal_mal_names_extract[i]))
    z = np.concatenate((z, z1))

datasets.numpy2csv('data/train/train.csv', X, y, z.tolist())
scaler = StandardScaler()
scaler.fit(X)


with open('data/scaler/scaler.pkl', 'wb+') as f:
    pickle.dump(scaler, f)

with open('data/scaler/scaler.pkl', 'rb') as f:
        scaler_loaded = pickle.load(f)

def CreateOriginalModel():
    original = svm.sklearn_SVC()

    # Train model
    X,y,_ = datasets.csv2numpy('data/train/train.csv')
    X_standard = scaler_loaded.transform(X)

    kf = KFold(n_splits=10)
    accuracy = []
    recall = []
    auc = []
    print("----------Model is being trained----------")
    for train_index, test_index in kf.split(X_standard):
        print(train_index, test_index)
        X_train, X_test = X_standard[train_index], X_standard[test_index]
        y_train, y_test = y[train_index], y[test_index]
        original.fit(X_train, y_train)
        y_pred = original.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, y_pred))
    print("----------Successful model training-------")

    accuracy_avg = sum(accuracy)/len(accuracy)
    recall_avg = sum(recall)/len(recall)
    auc_avg = sum(auc)/len(auc)

    print("Accuracy: ", accuracy_avg)
    print("Recall: ", recall_avg)
    print("Auc: ", auc_avg)

    original.save_model('data/models/origin.pkl')

CreateOriginalModel()
    
