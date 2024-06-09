import pickle

from sklearn.neighbors import KNeighborsClassifier

class sklearn_KNN(KNeighborsClassifier):
    def __init__(self, 
                 n_neighbors=5, # Found using grid search
                 weights='uniform', #'rbf', 
                 algorithm='auto', 
                 leaf_size=30, # Found using grid search
                 p=2,
                 metric='minkowski', 
                 metric_params=None, 
                 n_jobs=None):
        super(sklearn_KNN, self).__init__(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, 
                       leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params,
                       n_jobs=n_jobs)
    
    def save_model(self, modelfile):
        pickle.dump(self.__dict__, open(modelfile, 'wb+'))
    
    def load_model(self, modelfile):
        self.__dict__.update(pickle.load(open(modelfile, 'rb')))

