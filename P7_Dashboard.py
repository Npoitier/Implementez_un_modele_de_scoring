#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import streamlit as st
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular
from zipfile import ZipFile
import requests
from io import StringIO 

@st.cache
def load_data():
    chemin = 'C:/Users/nisae/OneDrive/Documents/jupyter_notebook/P7_Poitier_Nicolas/Dashboard/'
    chemin = 'https://raw.githubusercontent.com/Npoitier/Implementez_un_modele_de_scoring/main/'

    Liste_des_prets = chemin + 'data/Dashboard_submitt_values.csv'
    data = pd.read_csv(Liste_des_prets, index_col=0, encoding ='utf-8')
    target = data[['TARGET']].copy()
    list_columns = data.columns.tolist()
    list_columns = [col for col in list_columns if col != 'TARGET']
    data = data[list_columns].copy() 
    return chemin, data, target

def load_preprocessing(chemin, model_name):
    #filename = chemin + 'average_precision_score/' + 'TL_SN_pipe'+model_name+'_final_preprocess_model.sav'
    filename = 'TL_SN_pipe'+model_name+'_final_preprocess_model.sav'
    model = pickle.load(open(filename, 'rb'))
    return model

def load_model(chemin, model_name):
    #file = open(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', "r")
    #file = pd.read_csv(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', header=0, encoding ='utf-8')
    file = pd.read_csv(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', header=None, names=['col_name'], encoding ='utf-8')
    features = pd.Series(file['col_name']).tolist()

    #features = []
    #for line in file :    
    #    features.append(line.replace('\n',''))
    if 	model_name == 'RandomForestClassifier':
        seuil = 0.1
    else :
        seuil = 0.5

    #filename = chemin + 'average_precision_score/'+'TL_SN_pipe' + model_name +'_final_model.sav''./' +'/average_precision_score/'+ 
    filename = 'TL_SN_pipe' + model_name +'_final_model.sav'
    model = pickle.load(open(filename, 'rb'))

    return model, features, seuil

def prediction(model_name, data, id_pret):
    model, features, seuil = load_model(chemin, model_name)
    X=data[features].copy()    
    preproc = load_preprocessing(chemin, model_name)
    X_transform = preproc.transform(X[X.index == int(id_pret)])
    list_colonnes = preproc.get_feature_names_out().tolist()
    list_colonnes = pd.Series(list_colonnes).str.replace('quanti__','').str.replace('remainder__','').str.replace('quali__','').tolist()
    X_transform = pd.DataFrame(X_transform,columns=list_colonnes)
    X_transform = X_transform.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) 
    predic_classe = model.predict_proba(X_transform)[:,1]
    score = (predic_classe > seuil).astype(int)
    return int(score)
    
def shap_importance(model_name,id_pret,res_model_name,df_shap_values):
    if model_name != res_model_name:
        df_shap_values = pd.read_csv(chemin + 'data/' +model_name+"_shap_values.csv",
                                 index_col=0, encoding ='utf-8')
    #height = list(df_shap_values.iloc[id_pret])
    height = df_shap_values[df_shap_values.index == int(id_pret)]
    height = np.array(height.T)
    height = height[:,0].tolist()
    #somme = np.sum(height)
    maxi = np.max(np.abs(height))
    mini = np.min(np.abs(height))
    bars = df_shap_values.columns.tolist()
    bars = [bars[x] for x in range(len(height)) if np.abs(height[x]) >= maxi*0.05]
    height = [height[x] for x in range(len(height)) if np.abs(height[x]) >= maxi*0.05]
    fig = plt.figure(figsize=(10,len(bars)//2))
    plt.plot([0,0], [-1, len(bars)], color='darkblue', linestyle='--')
    #print(maxi,somme,len(bars),mini)
    y_pos = np.arange(len(bars))
    clrs = ['green' if (x > 0) else 'red' for x in height ]
    plt.barh(y_pos, height, color =clrs)
    plt.yticks(y_pos, bars)
    plt.title('With Shap '+model_name+' prêt '+str(id_pret))
    #plt.show()
    return model_name,df_shap_values,fig
    
def lime_importance(chemin, model_name):
    model, features, seuil = load_model(chemin, model_name)
    X=data[features].copy()   
    preproc = load_preprocessing(chemin, model_name)    
    X_transform = preproc.transform(X) 
    list_colonnes = preproc.get_feature_names_out().tolist()
    list_colonnes = pd.Series(list_colonnes).str.replace('quanti__','').str.replace('remainder__','').str.replace('quali__','').tolist()
    X_transform = pd.DataFrame(X_transform,columns=list_colonnes)
    X_transform = X_transform.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) 
    list_colonnes = X_transform.columns.tolist()
    X_transform.index = X.index
    explainer = lime_tabular.LimeTabularExplainer(X_transform, mode="classification",
                                              class_names=["Solvable", "Non Solvable"],
                                              feature_names=list_colonnes,
                                                 discretize_continuous=False)
    expdt0 = explainer.explain_instance(np.array(X_transform[X_transform.index == int(id_pret)].T).ravel(),
                                        model.predict_proba ,num_features=len(list_colonnes))
    test = np.array(expdt0.local_exp.get(1))
    list_cols = [list_colonnes[int(i)] for i in test[:,0]]
    height = test[:,1].tolist() 
    height.reverse()    
    #somme = np.sum(height)
    maxi = np.max(np.abs(height))
    mini = np.min(np.abs(height))
    bars = list_cols
    bars.reverse()
    bars = [bars[x] for x in range(len(height)) if np.abs(height[x]) >= maxi*0.05]
    height = [height[x] for x in range(len(height)) if np.abs(height[x]) >= maxi*0.05]
    fig = plt.figure(figsize=(10,len(bars)//2))
    plt.plot([0,0], [-1, len(bars)], color='darkblue', linestyle='--')
    #print(maxi,somme,len(bars),mini)
    y_pos = np.arange(len(bars))
    clrs = ['green' if (x > 0) else 'red' for x in height ]
    plt.barh(y_pos, height, color =clrs)
    plt.yticks(y_pos, bars)
    plt.title('With lime '+model_name+' prêt '+str(id_pret))
    return fig
    

res_model_name = ''
df_shap_values = None
chemin, data, target = load_data()

# sidebar

list_model = ['RandomForestClassifier','LGBMClassifier']
model_name = st.sidebar.selectbox("Model", list_model)
# model, features, seuil = load_model(chemin,model_name)

list_prets = data.index.values
id_pret = st.sidebar.selectbox("Loan ID", list_prets)

# 
st.write("Loan ID selection : ", id_pret)
st.write("model :",model_name)

st.header("Customer information display")
st.write("Gender : ", data.loc[data.index == int(id_pret),"CODE_GENDER"].values[0])
st.write("Age : {:.0f} ans".format(abs(int(data.loc[data.index == int(id_pret),"DAYS_BIRTH"].values[0]))//365))
st.write("Family status : ", data.loc[data.index == int(id_pret),"NAME_FAMILY_STATUS"].values[0])
st.write("Number of children : {:.0f}".format(data.loc[data.index == int(id_pret),"CNT_CHILDREN"].values[0]))

st.subheader("Income (USD)")
st.write("Income total : {:.0f}".format(data.loc[data.index == int(id_pret),"AMT_INCOME_TOTAL"].values[0]))
st.write("Credit amount : {:.0f}".format(data.loc[data.index == int(id_pret),"AMT_CREDIT"].values[0]))
st.write("Credit annuities : {:.0f}".format(data.loc[data.index == int(id_pret),"AMT_ANNUITY"].values[0]))
st.write("Amount of property for credit : {:.0f}".format(data.loc[data.index == int(id_pret),"AMT_GOODS_PRICE"].values[0]))

score = prediction(model_name, data, id_pret)
if score == 0:
	prediction = "Loan "+str(id_pret)+" granted"
else :
	prediction = "Loan "+str(id_pret)+" refused"
	
st.subheader(prediction)

res_model_name,df_shap_values,fig = shap_importance(model_name,id_pret,res_model_name,df_shap_values)
st.pyplot(fig)

fig_lime = lime_importance(chemin, model_name)
st.pyplot(fig_lime)

