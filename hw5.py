from pandas import read_csv
import pymorphy2
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
globvar = 0
import pandas as pd
def vec(s):
    coder.fit_transform(s)
def f_tokenizer(s):
    global globvar  # Needed to modify global copy of globvar
    globvar += 1
    print(globvar)
    morph = pymorphy2.MorphAnalyzer()

    t = s.split(' ')

    f = []
    for j in t:
        m = morph.parse(j.replace('.',''))
        if (len(m) != 0):
            wrd = m[0]
            if wrd.tag.POS not in ('NUMR','PREP','CONJ','PRCL','INTJ'):
                f.append(wrd.normal_form)
    return f
df = read_csv("D:\\MyML\\less5\\data\\train.csv",sep='\t')
print("load")
dftest = df#.head(20)
#dftest['normname']=dftest['name'].map(f_tokenizer)
dftest['data'] = dftest['name']#+" " + dftest['description']
coder = HashingVectorizer(tokenizer=f_tokenizer, n_features=150)
print("convert")


#dftest['vectorname']=dftest['normname'].map(vec)

data = dftest['data'].tolist()
myarray = np.asarray(data)
trn  =coder.fit_transform(myarray)
print("fit")
i = 0;
target = dftest['target']

print("startfit")
#grid_search = RandomizedSearchCV(model, param_distributions=param_dist,n_iter=n_iter_search, cv=3)
#grid_search.fit(trn, target)
#mymodel = grid_search.best_estimator_
print("===================================================================================")
#print(grid_search.best_params_)
#print(grid_search.best_score_)
print("===================================================================================")
#TRNtrain, TRNtest, TARtrain, TARtest = train_test_split(trn, target, test_size=0.25)
#mlymodel = model.fit(TRNtrain, TARtrain)
#print('roc_auc_score: ', roc_auc_score(TARtest, mymodel.predict(TRNtest)))
params= {'max_features': ['auto', 'sqrt', 'log2', None],
                                    'max_depth': range(3, 25),
                                    'criterion': ['gini', 'entropy'],
                                    'splitter': ['best', 'random'],
                                    'min_samples_leaf': range(1, 20),
                                    }
modelC=LogisticRegression(random_state=42,max_iter=20)
param_grid = {'C': [0.0001,0.001,0.01, 0.1, 1, 10, 100,1000,10000,100000], 'penalty': ['l1', 'l2']}
grid = GridSearchCV(modelC, cv=3,scoring='roc_auc',param_grid=param_grid)
grid.fit(trn, target)
#random_search.fit(trn, target)
print('Best estimator is {} with score {} using params {}'.format(modelC.__class__.__name__, grid.best_score_, grid.best_params_))
print("endfit")
mymodel = grid.best_estimator_
dft = read_csv("D:\\MyML\\less5\\data\\test.csv",sep='\t')
dft = dft#.head(10)
dft['data'] = dft['name']#+" " + dft['description']

dataTestT = dft['data'].tolist()
myarrayTestT = np.asarray(dataTestT)
Xtest  =coder.fit_transform(myarrayTestT)

target = mymodel.predict(Xtest)
res = pd.Series(target)
dft['target']=res
dfres=dft[['id','target']]

dfres.to_csv('D:\MyML\\less5\\results2.csv',sep=',', index=False)
print("End")
