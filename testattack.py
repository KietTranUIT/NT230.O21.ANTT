from models import svm,rf,lr,dt, knn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from helper import datasets
import pickle

with open('data/scaler/scaler.pkl', 'rb') as f:
        scaler_loaded = pickle.load(f)

def TestAttack():
    original = svm.sklearn_SVC()
    original.load_model('data/models/origin.pkl')
    # Test attack
    print(type(original))
    X_test,y_test,_ = datasets.csv2numpy('data/attack/evademl_best.csv')
    X_test_standard = scaler_loaded.transform(X_test)
    print("----------Model is being tested-----------")
    y_pred = original.predict(X_test_standard)
    print("----------Successful model testing--------")

    print("----------Result--------------------------")
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy = ', accuracy)
    
    recall = recall_score(y_test, y_pred)
    print('Recall = ', recall)

    print(y_pred)

    auc = roc_auc_score(y_test, y_pred)
    print('AUC = ', auc)
    print("------------------------------------------")

TestAttack()
