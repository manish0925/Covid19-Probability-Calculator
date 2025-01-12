import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(Data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(Data))
    test_set_size=int(len(Data)*ratio)
    test_indecies=shuffled[:test_set_size]
    train_indecies=shuffled[test_set_size:]
    return Data.iloc[train_indecies], Data.iloc[test_indecies]

if __name__=="__main__":

    df=pd.read_csv('Data.csv')
    train, test=data_split(df, 0.2)
    x_train=train[['fever', 'bodypain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    x_test=test[['fever', 'bodypain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    y_train=train[['InfectionPro']].to_numpy().reshape(2390,)
    y_test=test[['InfectionPro']].to_numpy().reshape(597,)
    clf=LogisticRegression()
    clf.fit(x_train, y_train)

    file=open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()
    
