import streamlit as st
from pyngrok import ngrok
import json
from urllib.request import urlopen
# from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
import pickle

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

def main() :

    @st.cache
    def load_data():
        data = pd.read_csv('data/application_sample.csv',
           index_col='SK_ID_CURR', encoding ='utf-8')
        sample = pd.read_csv('data/X_train_sample.csv', index_col='SK_ID_CURR',
                             encoding ='utf-8')
        description = pd.read_csv("data/features_description.csv", 
            usecols=['Row', 'Description'], index_col=0,
                encoding= 'unicode_escape')
        target = data.iloc[:, -1:]
        return data, sample, target, description

    def load_model():
        '''loading the trained model'''
        pickle_in = open('model/XGB_clf_model_f.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf
 
    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                    round(data["AMT_INCOME_TOTAL"].mean(), 2),
                        round(data["AMT_CREDIT"].mean(), 2)]
 
        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]
        targets = data.TARGET.value_counts()
 
        return nb_credits, rev_moy, credits_moy, targets
# 
    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/-365), 2)
        return data_age
# 
    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    @st.cache
    def load_prediction(sample, id):
        # X=sample.iloc[:, :-1]
        X=sample.iloc[:,2:]	
        #Appel de l'API : 
#        API_url = "http://127.0.0.1:5000/credit/" + str(id)
        API_url = "https://flask-app-fya.herokuapp.com/credit/" + str(id)
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        score = API_data['proba']
        return score
 
    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = KMeans(n_clusters=2).fit(sample)
#        knn = knn_train(sample)
        return knn
# 
    def load_kmeans(sample, id, knn):
        index = sample[sample.index == int(id)].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
#        st.write(data_client)
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        df_neighbors.drop(['index'], axis=1, inplace=True)
#        df_neighbors.reset_index()
        return df_neighbors.iloc[:,2:].sample(10)
# 
    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        data_client['index'] = int(id)
#        data_client.drop(['index'], axis=1, inplace=True)
        return data_client.iloc[:,2:]
# 
    #Loading data……
    data, sample, target, description = load_data()
    id_client = sample.index.values
    clf = load_model()

    #######################################
    # SIDEBAR
    #######################################

    #Title display
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Évaluation de Crédit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Aide à la décision</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #Customer ID selection
    st.sidebar.header("**Informations Générales**")
 
    #Loading selectbox
    chk_id = st.sidebar.selectbox("ID Client", id_client)

    #Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)
 
    ### Display of information in the sidebar ###
    #Number of loans in the sample
    st.sidebar.markdown("<u>Nombre de prêts traités:</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)
 
    #Average income
    st.sidebar.markdown("<u>Revenu Moyen (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)
 
    #AMT CREDIT
    st.sidebar.markdown("<u>Montant de prêt Moyen (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
#     
    #PieChart
    st.sidebar.markdown("<u>Performance générale</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)
#         
    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    #Display Customer ID from Sidebar
    st.write("***ID Client sélectionné:***", chk_id)


    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Informations Client**")
# 
# ######################################################################
    if st.checkbox("Afficher informations client ?"):

        infos_client = identite_client(data, chk_id)
        st.write("**Gender : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/-365)))
        st.write("**Situation Familiale : **", infos_client["NAME_FAMILY_STATUS"].values[0])        
        st.write("**Nombre enfants : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
 
        #Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="green", linestyle='--')
        ax.set(title='Age Clients', xlabel='Age(Annee)', ylabel='')
        st.pyplot(fig)
       #Infor sur revenu       
        st.subheader("*Revenu (USD)*")
        st.write("**Revenu total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Montant du prêt : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Rentes du crédit : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("**Montant du bien objet du prêt : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
#         
        #Income distribution plot
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Revenu Clients', xlabel='Revenu < 200K (USD)', ylabel='')
        st.pyplot(fig)
          
        #Relationship Age / Income Total interactive plot 
        data_sk = data.reset_index(drop=False)
# 		
        data_sk = data_sk.loc[data_sk['AMT_INCOME_TOTAL'] < 500000, :]
		
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH']/-365).round(1)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])
  
        fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relation Age / Revenu", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))
#   
        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Revenu", title_font=dict(size=18, family='Verdana'))
# 
        st.plotly_chart(fig)
     
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
# 
    #Customer solvability display
    st.header("**Analyse Dossier Client**")
    prediction = load_prediction(sample, chk_id)
    st.write("**Probabilité de défaut : **{:.0f} %".format(round(float(prediction)*100, 2)))
# 
    #Compute decision according to the best threshold
 
    if prediction <= 0.10 :
        decision = "<font color='green'>**PRET ACCEPTE**</font>"
		
    else:
        decision = "<font color='red'>**PRET REFUSE**</font>"
# 
    st.write("**Decision** *(Avec seuil de 10%)* **: **", decision, unsafe_allow_html=True)
    st.markdown("<u>Données Client :</u>", unsafe_allow_html=True)
    shows = identite_client(data, chk_id)
    gb = GridOptionsBuilder.from_dataframe(shows)
    gb.configure_grid_options(domLayout='autoHeight')
    gb.configure_pagination()
    gridOptions = gb.build()
    AgGrid(shows, gridOptions=gridOptions)		
# 
# ###################################################################################
    #Feature importance / description
    if st.checkbox("ID Client {:.0f} Importance des variables?".format(chk_id)):
        shap.initjs()
        X = sample.iloc[:, :-1]
        X = X[X.index == chk_id]
        number = st.slider("Choisir le nonbre de variables …", 0, 20, 5)
# 
        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.Explainer(clf)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)
#         
        if st.checkbox("Besoins d'info detaillée sur variables ?") :
            list_features = description.index.to_list()
            feature = st.selectbox('Liste des variables …', list_features)
            st.table(description.loc[description.index == feature][:1])
#        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
#         
    #Similar customer files display
    chk_voisins = st.checkbox("Affichier les dossiers de clients similaires ?")
# 
# #######################################################################
# DEBUT Modif 29/12/2021
# #######################################################################
    if chk_voisins:
#         knn = load_knn(sample)
        st.markdown("<u>Liste des 10 dossiers proches :</u>", unsafe_allow_html=True)

        # Extraire l'observation du client choisi 
        X = sample.iloc[:, :-1]
        x_new = X[X.index == chk_id]
#        x_new = pd.DataFrame(sample.iloc[chk_id:chk_id+1,])
        # Definir un ficbhier ne contenant pas le client choisi
        samples = X.drop(chk_id)
        # Definir le modele de Nearest observations, avec une observation la + proche
        neigh_model = NearestNeighbors(n_neighbors=1)
        # Fiter le modele
        neigh_model.fit(samples)
        # Extraire les 10 observations les + proches 
        neigh_indices = neigh_model.kneighbors(x_new, 10, return_distance=False)
        # Afficher les observations 
        index_list = neigh_indices[0]
        shows = data.loc[data.index[index_list]]
        drop_cols = [0,2]
        shows = shows.drop(shows.columns[drop_cols], axis=1)

#        AgGrid(shows, height=500, fit_columns_on_grid_load=True)
  
        gb = GridOptionsBuilder.from_dataframe(shows)
#        gb.configure_grid_options(domLayout='autoHeight')
        gb.configure_pagination()
        gridOptions = gb.build()
        AgGrid(shows, gridOptions=gridOptions)		
# 
        st.markdown("<i>Target 1 = Client en Défault</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
# #######################################################################
# FIN Modif 29/12/2021
# #######################################################################
# 
# For debugging only .....
# 
#     st.markdown("<i>FOR DEBUGGING ONLY .....</i>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
