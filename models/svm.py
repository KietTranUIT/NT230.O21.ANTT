import pickle

kernal_function = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

from sklearn.svm import SVC

class sklearn_SVC(SVC):
    def __init__(self, 
                 C=20, # Found using grid search
                 kernel='rbf', 
                 degree=3, 
                 gamma=0.01, # Found using grid search
                 coef0=0.0, 
                 shrinking=True, 
                 probability=False, 
                 tol=0.001, 
                 cache_size=200, 
                 class_weight=None, 
                 verbose=False, 
                 max_iter=-1):
        super(sklearn_SVC, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, 
                       coef0=coef0, shrinking=shrinking, probability=probability, 
                       tol=tol, cache_size=cache_size, class_weight=class_weight, 
                       verbose=verbose, max_iter=max_iter)
    
    def save_model(self, modelfile):
        pickle.dump(self.__dict__, open(modelfile, 'wb+'))
    
    def load_model(self, modelfile):
        self.__dict__.update(pickle.load(open(modelfile, 'rb')))


