import pickle

from sklearn.linear_model import LogisticRegression

class sklearn_LR(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, 
                 tol=0.0001, C=1.0, fit_intercept=True, 
                 intercept_scaling=1, class_weight=None, 
                 random_state=None, solver='lbfgs', max_iter=100, 
                 multi_class='auto', verbose=0, warm_start=False, 
                 n_jobs=None):
        super(sklearn_LR, self).__init__(penalty=penalty, dual=dual, 
                                          tol=tol, C=C, fit_intercept=fit_intercept, 
                                          intercept_scaling=intercept_scaling, class_weight=class_weight, 
                                          random_state=random_state, solver=solver, max_iter=max_iter, 
                                          multi_class=multi_class, verbose=verbose, warm_start=warm_start, 
                                          n_jobs=n_jobs)
    
    def save_model(self, modelfile):
        pickle.dump(self.__dict__, open(modelfile, 'wb+'))
    
    def load_model(self, modelfile):
        self.__dict__.update(pickle.load(open(modelfile, 'rb')))

