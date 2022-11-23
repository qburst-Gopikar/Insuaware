import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
import os
import json
import joblib
from google.cloud import storage
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

insuranceDF = pd.read_csv("gs://logistic_regression_testing/gender_pred_model/data/md5_matches.csv")
insuranceDF_taxonomy = pd.read_csv("gs://logistic_regression_testing/gender_pred_model/data/taxonomy_labelled.csv").loc[:, ['id','name']]

insuranceDF = insuranceDF.assign(list_id=insuranceDF.list_id.str.strip('[]').str.strip().str.split(',')).explode('list_id')
insuranceDF = insuranceDF.assign(list_id=insuranceDF.list_id.str.strip())

insuranceDF_final= pd.merge(insuranceDF, insuranceDF_taxonomy, left_on='list_id', right_on='id',how='inner').loc[:, ['md5_hash','name']]
insuranceDF_final = insuranceDF_final.drop_duplicates()

insuranceDF_final = insuranceDF_final.groupby('md5_hash').agg({'name':lambda x: list(x)})
insuranceDF_final = insuranceDF_final['name'].str.join('|').str.get_dummies()

X = insuranceDF_final.drop(['Life'],axis=1)
Y = insuranceDF_final.Life

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=5)

neural_net = MLPClassifier(hidden_layer_sizes=(10),activation='logistic',solver='lbfgs', max_iter=1500)
neural_net.fit(X_train,Y_train)

filename='model.joblib'
joblib.dump(neural_net,filename)

# #upload model to cloud storage
model_dir = os.environ['AIP_MODEL_DIR']
storage_path = os.path.join(model_dir,filename)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(filename)




