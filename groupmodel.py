from models import svm, dt, lr, rf, knn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from helper import datasets
import pickle
import random
import time

# Init a seed random
current_time = time.time()
seed = int(current_time)
random.seed(seed)

# Load StandardScaler
with open('data/scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# load original model
original = svm.sklearn_SVC()
original.load_model('data/models/origin.pkl')

# load train dataset
X,y,_ = datasets.csv2numpy('data/train/train.csv')
X_standard = scaler.transform(X)

# mutate operator SVM model
svc_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
def mutateSVM():
    c = random.randint(1, 100)
    kernel = svc_kernel[random.randint(0,3)]
    print("SVM model with C= " + str(c) + ", Kernel= " + kernel)
    model = svm.sklearn_SVC(C=c, kernel=kernel)
    return model

# mutate operator DT model
def mutateDT():
    max_depth = random.randint(2,80)
    min_samples_leaf = random.randint(1,3)
    max_leaf_nodes = random.randint(2,100)
    print("DT model with max_depth=" +str(max_depth)+", min_samples_leaf= "+ str(min_samples_leaf)+", max_leaf_nodes=" + str(max_leaf_nodes))
    model = dt.sklearn_DT(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
    return model

# mutate operator RF model
def mutateRF():
    max_depth = random.randint(2,80)
    min_samples_leaf = random.randint(1,3)
    max_leaf_nodes = random.randint(2,100)
    print("RF model with max_depth=" +str(max_depth)+", min_samples_leaf= "+ str(min_samples_leaf)+", max_leaf_nodes=" + str(max_leaf_nodes))
    model = rf.sklearn_RF(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
    return model

# mutate operator LR model
lr_solver = ["lbfgs", "newton-cg", "liblinear"]
def mutateLR():
    c = random.randint(1, 100)
    solver = lr_solver[random.randint(0,2)]
    print("LR model with C=" + str(c) + ", Solver=" + solver)
    model = lr.sklearn_LR(C=c, solver=solver)
    return model

# mutate operator RF model
def mutateKNN():
    n_neighbors = random.randint(2,5)
    print("KNN model with n_neighbors=" + str(n_neighbors))
    model = knn.sklearn_KNN(n_neighbors=n_neighbors)
    return model

# training model
def trainModel(model):
    kf = KFold(n_splits=2)
    accuracy = []
    recall = []
    auc = []
    for train_index, test_index in kf.split(X_standard):
        X_train, X_test = X_standard[train_index], X_standard[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, y_pred))
    accuracy_avg = sum(accuracy)/len(accuracy)
    recall_avg = sum(recall)/len(recall)
    auc_avg = sum(auc)/len(auc)

    print("accuracy: "+str(accuracy_avg)+", recall: "+str(recall_avg)+", auc: "+str(auc_avg))
    if accuracy_avg < 0.987:
        return model, False
    if recall_avg < 0.980:
        return model, False
    if auc_avg < 0.985:
        return model, False
    return model, True

# calculate score of a model
# def calculateScore(model):
#     y_predict = model.predict(X_test_standard)
#     accuracy = accuracy_score(y_test, y_predict)
#     recall = recall_score(y_test, y_predict)
#     auc = roc_auc_score(y_test, y_predict)
#     print("Score: accuracy="+str(accuracy)+", recall="+str(recall)+",auc="+str(auc))
#     if accuracy < 0.996:
#         return False
#     if recall < 0.996:
#         return False
#     if auc < 0.996:
#         return False
#     return True


# algorithm create a model group
n = 50 # the number of models in group

# 0 -> svm 
# 1- > dt
# 2 -> rf
# 3 -> lr
# 4 -> knn
group_model = dict() # list contains all model of group

def createModelGroup():
    index = 0
    while index < n:
        option = random.randint(0, 4)
        model = None
        name = ''
        if option == 0:
            model = mutateSVM()
            name = 'svm-' + str(index) +'.pkl'
        elif option == 1:
            model = mutateDT()
            name = 'dt-' + str(index) +'.pkl'
        elif option == 2:
            model = mutateRF()
            name = 'rf-' + str(index) +'.pkl'
        elif option == 3:
            model = mutateLR()
            name = 'lr-' + str(index) +'.pkl'
        else:
            model = mutateKNN()
            name = 'knn-' + str(index) + '.pkl'

        if model is None:
            print("Error--------------")
            return
        
        model, flag = trainModel(model)
        if flag:
            group_model[name] = model
            index = index + 1
            print("Added a model to list! Size of list = "+str(index))
    print("Generate Model Group Successfully!")

# Save list model
models_path = "data/models/group/"
def saveListModel():
    for key in group_model:
        group_model[key].save_model(models_path + key)

createModelGroup()
saveListModel()
