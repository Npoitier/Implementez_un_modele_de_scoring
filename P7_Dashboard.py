#!/usr/bin/env python
# coding: utf-8

# from pathlib import Path
import pandas as pd
import streamlit as st
# import pickle
# import re
import numpy as np
import matplotlib.pyplot as plt
# from lime import lime_tabular
# from zipfile import ZipFile
import requests
# from io import StringIO
import json
import os
from PIL import Image

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



def prediction(model_name, metric, id_pret):
    dicostr = requests.get("https://modele-de-scoring-api.herokuapp.com/predict/"+model_name+"/metric/"+metric+"/indice/"+str(id_pret))
    dico = json.loads(dicostr.text)
    score = int(dico.get('score'))
    classe = int(dico.get('classe'))
    fiabilite = float(dico.get('fiabilite'))
    return score, classe, fiabilite
    
def get_importance(model_name, metric, id_pret, method):
    
    if method == 'featureimportance':
        dicostr = requests.get("https://modele-de-scoring-api.herokuapp.com/"+method+"/"+model_name+"/metric/"+metric)
        Title = 'model: '+model_name+' metric: '+metric+' features importance'
    else:
        dicostr = requests.get("https://modele-de-scoring-api.herokuapp.com/"+method+"/"+model_name+"/metric/"+metric+"/indice/"+str(id_pret))
        Title = 'With '+method+' '+model_name+' prêt '+str(id_pret)
        
    dico = json.loads(dicostr.text)
    height = []
    bars = []
    for i,v in dico.items():
        bars.append(i)
        height.append(v)
    fig = plt.figure(figsize=(10,len(bars)//2))
    plt.plot([0,0], [-1, len(bars)], color='darkblue', linestyle='--')
    #print(maxi,somme,len(bars),mini)
    y_pos = np.arange(len(bars))
    clrs = ['green' if (x > 0) else 'red' for x in height ]
    plt.barh(y_pos, height, color =clrs)
    plt.yticks(y_pos, bars)
    plt.title(Title)
    return fig
    
def get_graphlist():
    images = r"C:\Users\nisae\OneDrive\Documents\GitHub\Implementez_un_modele_de_scoring\pictures"
    images = "./pictures/"
    extension = '.jpg'
    graph_list = []
    for root, dirs_list, files_list in os.walk(images):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                graph = file_name.split('.')[0].replace('_',' ')
                graph_list.append(graph)
    return graph_list, images

def graph_picture(images,graph_name):
    image_name = images+"/"+graph_name.replace(' ','_')+'.jpg'
    image= Image.open(image_name)
    return image


chemin, data, target = load_data()

# sidebar

list_model = ['RandomForestClassifier','LGBMClassifier']
model_name = st.sidebar.selectbox("Model", list_model)
# model, features, seuil = load_model(chemin,model_name)

list_metric = ['average_precision_score','BAW_metrique']
metric = st.sidebar.selectbox("Optimisation Metric", list_metric)

list_prets = data.index.values
id_pret = st.sidebar.selectbox("Loan ID", list_prets)

# 
st.write("Loan ID selection : ", id_pret)
st.write("model :",model_name)

st.header("Data information")
list_graph, images = get_graphlist()
graph_name = st.selectbox("Graph", list_graph)
image = graph_picture(images,graph_name)
st.image(image)

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

score, classe, fiabilite = prediction(model_name, metric, id_pret)
if score == 0:
    if score == classe :
        prediction = "Loan "+str(id_pret)+" granted, it's a good predict"
    else :
        prediction = "Loan "+str(id_pret)+" granted, it's a bad predict"
else :
    if score == classe :
        prediction = "Loan "+str(id_pret)+" refused, it's a good predict"
    else :
        prediction = "Loan "+str(id_pret)+" refused, it's a bad predict"
        
	
st.subheader(prediction)
st.write("model's confidence about ranking "+str(round(fiabilite,2))+'%')

if st.sidebar.checkbox("Show shap explaination ?"):
    fig = get_importance(model_name, metric, id_pret, 'shap')
    st.pyplot(fig)
if st.sidebar.checkbox("Show lime explaination ?"):
    fig_lime = get_importance(model_name, metric, id_pret, 'lime')
    st.pyplot(fig_lime)
    
if st.sidebar.checkbox("Show model features importance ?"):
    fig_gen = get_importance(model_name, metric, id_pret, 'featureimportance')
    st.sidebar.pyplot(fig_gen)
