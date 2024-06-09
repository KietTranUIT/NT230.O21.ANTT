from models import svm, dt, lr, rf, knn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from helper import datasets, utility, featureedit
from argparse import ArgumentParser
import pickle
import os
import sys
import numpy as np
import multiprocessing

# Load StandardScaler
with open('data/scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# load original model
original = svm.sklearn_SVC()
original.load_model('data/models/origin.pkl')

# path to folder contain group model
group_path = "data/models/group"

# load all file models
pkl_files = [f for f in os.listdir(group_path) if f.endswith('.pkl')]
# list contains all model in group model
group_models = []

# benign dataset
X_benign,y_benign,_ = datasets.csv2numpy('data/contagio-ben.csv')
X_benign_standard = scaler.transform(X_benign)
pir_list = []

# load a folder into model
def loadModel(file):
    model = None
    if file.find('svm') != -1:
        model = svm.sklearn_SVC()
    elif file.find('dt') != -1:
        model = dt.sklearn_DT()
    elif file.find('lr') != -1:
        model = lr.sklearn_LR()
    elif file.find('rf') != -1:
        model = rf.sklearn_RF()
    else:
        model = knn.sklearn_KNN

    if model is None:
        print("Error load model")
        sys.exit()
    model.load_model(group_path + '/' + file)
    return model

# calculate PIR for benign examples
def calculatePIR(example):
    p = 0
    # convert numpy 1D to 2D
    data = example.reshape(1,-1)
    # original model predict
    orig_pred = original.predict(data)
    if len(orig_pred) > 1:
        print("Error predict of original model!")
        sys.exit()
    
    # group model predict
    for model in group_models:
        result = model.predict(data)
        if len(result) > 1:
            print("Error predict of group model!")
            sys.exit()
        if orig_pred[0] != result[0]:
            p = p + 1
    return p / float(len(group_models))

# get features of a pdf file
def get_features(pdf_name):
    try:
        pdf = featureedit.FeatureEdit(pdf_name)
    except:
        return None
    feats = pdf.retrieve_feature_dictionary()
    if feats['size'] == 0 or isinstance(feats['size'], Exception):
        return None
    return pdf_name, pdf.retrieve_feature_vector_numpy()

# extract pdf to vector features
def extract_features(pdfs):
    feat_vecs = []
    file_names = []
    # Extract malicious and benign features
    pool = multiprocessing.Pool()
    for pdf, feats in pool.imap(get_features, pdfs):
        if feats is not None:
            feat_vecs.append(feats)
            file_names.append(pdf)
    
    # Convert the data points into numpy.array
    X = np.array(np.zeros((len(feat_vecs), 
                                 featureedit.FeatureDescriptor.get_feature_count())), 
                                 dtype=np.float64, order='C')
    for i, v in enumerate(feat_vecs):
        X[i, :] = v
    return X, file_names

# predict for a PDF file
def main():
    # Setup argument parser
    parser = ArgumentParser()
    parser.add_argument('--file', help='file PDFs (directory or file with list of paths)')
    parser.add_argument('--test', help='test attack .csv')


    # Process arguments
    args = parser.parse_args()
    pdfs = [], []
    if args.file:
        pdfs = sorted(utility.get_pdfs(args.file))
    else:
        print("File not found!")
        #return 0
    #X, file_names = extract_features(pdfs)

    # load all models from group folder
    for file in pkl_files:
        model = loadModel(file)
        group_models.append(model)

    # calculate PIR benign
    data = X_benign_standard[:100,:]
    for row in data:
        pir = calculatePIR(row)
        pir_list.append(pir)

    # calculate pir average benign examples
    pb = sum(pir_list)/float(len(pir_list))
    print("Pb = ", pb)
    if args.test:
        X_attack,y_attack,_= datasets.csv2numpy('data/attack/mimicus_F_mimicry.csv')
        data = scaler.transform(X_attack)
        check = 0
        for row in data:
            pir = calculatePIR(row)
            print(pir)
            if pir > pb:
                check = check + 1
        print("Detect: "+str(check)+"/"+str(len(data)))     
    else:
        count = 0
        data = scaler.transform(X)
        for row in data:
            pir = calculatePIR(row)
            evalute = 'benign file'
            if pir > pb:
                evalute = 'malicious file'
            print("File "+file_names[count]+" is "+evalute)
            count = count + 1
    return 0

if __name__ == "__main__":
    sys.exit(main())








