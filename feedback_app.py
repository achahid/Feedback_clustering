




# pipreqs .
# This will create requirements.txt file at the current directory.

import streamlit_authenticator as stauth
import nltk
import streamlit_ext as ste
import nltk_download_utils
from googletrans import Translator
import pandas as pd
import time
import warnings
import datetime
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from deep_translator import GoogleTranslator
import sys
# import huggingface_hub.snapshot_download
from huggingface_hub import snapshot_download
from sklearn.cluster import AgglomerativeClustering
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
import re
from nltk.tokenize import word_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.simplefilter(action='ignore', category=FutureWarning)
from sentence_transformers import SentenceTransformer, util

import xlsxwriter
from io import BytesIO

import streamlit as st



pd.options.mode.chained_assignment = None

now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")



#### FUNCTIONS #####

def translate_to_english(text_list):
    # translator = Translator(service_urls=['translate.google.com'])
    translator = Translator(service_urls=['translate.google.com'],
                            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)', proxies=None, timeout=None)
    translated_text = []
    count = 0
    for text in text_list:
        try:
            result = translator.translate(text, dest='en').text
            translated_text.append(result)
            count += 1
            if count % 1000 == 0:
                print('time to sleep 5 sec')
                time.sleep(5)
        except Exception as e:
            translated_text.append("Translation failed")
    return translated_text

def keyowrds_removal(df, list_to_remove):
    for key in range(len(list_to_remove)):
        df['keyword_eng'] = df['keyword_eng'].str.replace(list_to_remove[key], '')
    return df

# IMPORTANT FUNCTIONS:
def data_preprocessing(df):
    # Make all column to lower case.
    df.columns = map(str.lower, df.columns)
    if 'keyword' not in df.columns:
        # print('ERROR: PLEASE CHECK IF YOUR DATA CONTAINS keyword COLUMN')
        st.error('Please ensure that your data includes the column **KEYWORD**', icon="🚨")
        sys.exit(1)

    if 'id' not in df.columns:
        df['id'] = range(len(df))
        print('id is added to the data')

    if 'keyword_eng' not in df.columns:
        with st.spinner('**The keywords are in the process of being translated to ENGLISH. Please hold on ...** '):
            df = df[['id', 'keyword']].copy()
            df.dropna(inplace=True)
            # Adding 'digit-' prefix for the rows that contains digits only as GoogleTranslator can not
            # translate digits only.
            df["keyword"] = df["keyword"].apply(lambda x: 'digit-' + x if x.isdigit() else x)
            # print("The keywords are in the process of being translated to ENGLISH. Please hold on ... ")
            my_list = df["keyword"].to_list()
            # df["Keyword_eng"] = translate_to_english(my_list)
            df["keyword_eng"] = df["keyword"].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(x))
            # remove the added prefix from the rows
            df["keyword_eng"] = df["keyword_eng"].apply(lambda x: x.replace("digit-", "") if x.startswith("digit-") else x)
            df["keyword"] = df["keyword"].apply(lambda x: x.replace("digit-", "") if x.startswith("digit-") else x)
        st.success('**The translation process is finished, we are now moving on to the clustering process.***')

    # Splits the data into short and long tail keywords:
    df_org = df.copy()
    df['keyword_eng'] = df['keyword_eng'].astype(str)
    df['keyword'] = df['keyword'].astype(str)

    df_new = df[['id', 'keyword', 'keyword_eng']].copy()
    # Remove the next words.
    keywords_quistions = ['what is', 'why is', 'what', 'why', 'how much', 'how long', 'how many',
                          'how do', 'how to', 'when', 'when is', 'where is']

    df_new = keyowrds_removal(df_new, keywords_quistions)

    df_new['strings_only'] = df_new['keyword'].str.replace('\d+', '', regex=True)
    df_new.insert(2, "words", df_new['strings_only'].str.split(), True)
    df_new.insert(3, 'amount', [len(v) for v in df_new.words], True)
    short_tail_df = df_new[df_new.amount < 2]
    short_tail_df = short_tail_df.drop(['strings_only', 'words', 'amount'], axis=1)
    short_tail_df.reset_index(drop=True, inplace=True)
    ID = short_tail_df.id
    long_tail_df = df_new[~df_new.id.isin(ID)][['id', 'keyword', 'keyword_eng']].copy()

    print(" *** There were {} SHORT TAIL keywords, and {} LONG TAIL keywords".format(short_tail_df.shape[0],
                                                                                     long_tail_df.shape[0]))
    return long_tail_df, short_tail_df, df_org

def K_MEANS_TRANSFORMATION(processed_df,  start_cluster, end_cluster, steps):

    # Define the list of feedback sentences
    feedback_list = processed_df.keyword_eng.to_list()

    # Generate sentence embeddings for each feedback sentence
    embeddings = model.encode(feedback_list)
    dic = {}


    for cl_num in range(start_cluster, end_cluster, steps):
        try:
            kmeans = KMeans(n_clusters=cl_num,random_state=10)
            cluster_assignment = kmeans.fit_predict(embeddings)

            clustered_sentences = {}
            clustered_sentences_id = {}


            for sentence_id, cluster_id in enumerate(cluster_assignment):
                if cluster_id not in clustered_sentences:
                    clustered_sentences[cluster_id] = []

                if cluster_id not in clustered_sentences_id:
                    clustered_sentences_id[cluster_id] = []

                clustered_sentences[cluster_id].append(feedback_list[sentence_id])
                # clustered_sentences_id[cluster_id].append(ID[sentence_id])

            your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
            dft = your_df_from_dict.transpose()
            df_results = pd.melt(dft, value_vars=dft.columns)
            df_results.dropna(inplace=True)
            df_results.rename(columns={'variable': 'cluster', 'value': 'keyword_eng'}, inplace=True)
            dic[cl_num] = df_results

        except Exception as e:
            print(e)
            continue

    return dic


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def dfs_xlsx(data_list):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    for sheet_name, df in data_list.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def AgglomerativeClustering_algo(model_name_topics,keywords_df):

    dic={}
    feedback_list = keywords_df.keyword_eng.to_list()
    embedder = SentenceTransformer(model_name_topics)
    corpus_embeddings = embedder.encode(feedback_list)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(feedback_list[sentence_id])

    your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
    dft = your_df_from_dict.transpose()

    df_results = pd.melt(dft, value_vars=dft.columns)
    df_results.dropna(inplace=True)
    df_results.rename(columns={'variable': 'clusters', 'value': 'feedback'}, inplace=True)
    dic['Aglo_clusters'] = df_results
    return dic

def option_to_model(level_number,options):
  try:
    return options[level_number]
  except Exception as e:
    return e

# These are some model options for sentence transformers:f
option_models = {
    "<select>": '<select>',
    "General Base": 'all-mpnet-base-v2',
    "General Roberta": 'all-distilroberta-v1',
    "General miniML_L12": 'all-MiniLM-L12-v2',
    "General miniML_L6": 'all-MiniLM-L6-v2',
    "Medics": 'pritamdeka/S-PubMedBert-MS-MARCO',
    'Education and training': 'bert-base-nli-mean-tokens',
    'Finance':'roberta-base-nli-mean-tokens',
}


html_temp = """
<div style="background-color:blue;padding:1.5px">
<h1 style="color:white;text-align:center;"> FEEDBACK CLUSTERING APP</h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)


# provide a color for buttons.
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)


# --- USER AUTHENTICATION ---
names = ['User Unknown','Lars van Tulden', 'Helena Geginat', 'abdelhak chahid','Michael van den Reym','Mitchell Pijl','Nhu Nguyen']
usernames = ['admin','ltulden', 'hgeginat', 'achahid','mreym','mpijl','nnguyen']
passwords = ['io123#$','123#$123', '123#$123', '123#$123','123#$123','123#$123','123#$123']

hashed_passwords = stauth.Hasher(passwords).generate()
authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
name, authentication_status, username = authenticator.login('Login', 'sidebar')



if st.session_state["authentication_status"]:

    authenticator.logout("Logout","sidebar")
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')
    st.sidebar.text('version Jan 2023')

    st.warning("Please ensure that your data includes the column **KEYWORD** :eye-in-speech-bubble: ")
    uploaded_file_cl = st.file_uploader("Upload data", type=['csv'])


    if uploaded_file_cl is not None:

        # keywords_df = pd.read_csv(uploaded_file_cl,encoding='latin-1')
        # max_value = np.trunc(keywords_df.shape[0] - 2).astype(int)
        # # long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)
        # st.dataframe(keywords_df)
        keywords_df = pd.read_csv(uploaded_file_cl, encoding='latin-1')
        # long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)
        # data_download = convert_df(processed_data)
        # ste.download_button("Press to Download", data_download, "translated_data.csv")


    load_K_means = st.button('GENERATE CLUSTERS: K-MEANS' )

    if load_K_means:
        long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)

        with st.spinner('**The K-MEANS clustering algorithm is currently in operation. Please hold on ...**'):

            model_name = 'all-MiniLM-L6-v2'
            model = SentenceTransformer(model_name)

            max_cluster = max(3,np.trunc(keywords_df.shape[0] * 0.1).astype(int))
            min_cluster = max(1,np.trunc(max_cluster / 2).astype(int))
            steps = max(1,np.trunc((max_cluster - min_cluster) / 3).astype(int))

            data_list = K_MEANS_TRANSFORMATION(processed_data, start_cluster=5, end_cluster=15, steps=5)
            preffix = 'CLUSTER_id_'
            new_dict = {(preffix + str(key)): value for key, value in data_list.items()}
            data_list = new_dict

            df_xlsx = dfs_xlsx(data_list)

            st.write("""
            <p style="background-color: #FEC929; color: black; padding: 10px;"> 
            Further examination is recommended for the subsequent clusters..
            </p>
            """, unsafe_allow_html=True)

            st.subheader("Download data")
            ste.download_button(label='Download Results',
                               data=df_xlsx,
                               file_name='K_MEANS_clustering.xlsx')

    ### AGLOMERATIVE  :
    model_name = ["<select>", "General Base", "General Roberta", "General miniML_L12", "General miniML_L6",
                  "Medics", "Education and training", "Finance"]

    select_box = st.selectbox('Select a model Transformer', options=model_name)
    selected_option = option_to_model(select_box,option_models)
    load_transformers = st.button('GENERATE CLUSTERS: TRANSFORMERS')


    if load_transformers and select_box != '<select>':
        st.write('You selected model:', selected_option)
        long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)

        with st.spinner('**The Model Aglomerative clustering algorithm is currently running. Please hold on...**'):


            data_list = AgglomerativeClustering_algo(selected_option, processed_data)

            preffix = 'CLUSTER_id_'
            new_dict = {(preffix + str(key)): value for key, value in data_list.items()}
            data_list = new_dict
            df_xlsx = dfs_xlsx(data_list)

            st.write("""
                        <p style="background-color: #FEC929; color: black; padding: 10px;">
                        Further examination is recommended for the subsequent clusters..
                        </p>
                        """, unsafe_allow_html=True)

            st.subheader("Download data")
            ste.download_button(label='Download Results',
                               data=df_xlsx,
                               file_name='Aglomerative_clustering.xlsx')




elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')


if st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')














