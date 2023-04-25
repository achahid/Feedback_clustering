





# pipreqs .
# This will create requirements.txt file at the current directory.


import streamlit_authenticator as stauth
import nltk
import csv
import io
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

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'



pd.options.mode.chained_assignment = None

now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")


# FUNCTIONS:

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

def stemmList(my_list):
    # the stemmer requires a language parameter
    snow_stemmer = SnowballStemmer(language='english')
    # porter_stemmer = PorterStemmer()

    nltk.download('punkt')
    stemmed_list = []
    for l in my_list:
        words = l.split(" ")
        stem_words = []
        # print(l)
        for word in words:
            x = snow_stemmer.stem(word)
            stem_words.append(x)
        key = " ".join(stem_words)
        # print(key)
        stemmed_list.append(key)
    return stemmed_list

# IMPORTANT FUNCTIONS:
def data_preprocessing(df):
    # Make all column to lower case.
    df.columns = map(str.lower, df.columns)

    if 'keyword' not in df.columns:
        print('ERROR: PLEASE CHECK IF YOUR DATA CONTAINS keyword COLUMN')
        # st.error('Please ensure that your data includes the column **KEYWORD**', icon="🚨")
        sys.exit(1)

    if 'id' not in df.columns:
        df['id'] = range(len(df))
        print('id is added to the data')

    if 'keyword_eng' not in df.columns:
        with st.spinner('**The keywords are in the process of being translated to ENGLISH. Please hold on ...** '):
            # df = df[['id', 'keyword']].copy()
            df = df[df['keyword'].notna()]
            # Adding 'digit-' prefix for the rows that contains digits only as GoogleTranslator can not
            # translate digits only.
            df["keyword"] = df["keyword"].apply(lambda x: 'digit-' + x if x.isdigit() else x)
            # print("The keywords are in the process of being translated to ENGLISH. Please hold on ... ")
            # my_list = df["keyword"].to_list()
            # df["keyword_eng"] = translate_to_english(my_list)
            df["keyword_eng"] = df["keyword"].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(x))
            df.dropna(subset=['keyword_eng'], inplace=True)  # remove NONE that was produced when trying to translate strange
                                                  # characters like :"????"
                                                  # remove the added prefix from the rows
            df["keyword_eng"] = df["keyword_eng"].apply(lambda x: x.replace("digit-", "") if x.startswith("digit-") else x)
            df["keyword"] = df["keyword"].apply(lambda x: x.replace("digit-", "") if x.startswith("digit-") else x)
        st.success('**The translation process is finished, we are now moving on to the clustering process.***')

    # Splits the data into short and long tail keywords:
    df = df.dropna(subset=['keyword_eng'])
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

def labelling_clusters(df, cluster_num, n):
    df_cluster = df[df["cluster"] == cluster_num]
    keywords_list = df_cluster.keyword_eng.to_list()
    words = [word_tokenize(i) for i in keywords_list]
    words_list = sum(words, [])

    # Make the words singular
    singular_words = [wnl.lemmatize(wrd) for wrd in words_list]
    singular_words_lower = list(map(lambda x: x.lower(), singular_words))

    # Remove stop words
    stop_words = nltk.corpus.stopwords.words('english')
    clean_words = [word for word in singular_words_lower if word not in stop_words]
    clean_words_0 = [re.sub('[^a-zA-Z0-9]+', "", i) for i in clean_words]
    clean_words_1 = [item for item in clean_words_0 if not item.isdigit()]
    clean_words_2 = [x for x in clean_words_1 if x]

    # Calculate the frequency of each word
    fdist = nltk.FreqDist(clean_words_2)
    # Rank the words by frequency
    keywords = sorted(fdist, key=fdist.get, reverse=True)
    keywords_1 = [' '.join(keywords[:n])]
    return keywords_1


def K_MEANS_TRANSFORMATION(processed_df, n_clusters):
    """
    This function will generate many clusters using TRANSFORMERS models and K-means.
    :param processed_df : a preprocessed dataframe.
    :param n_clusters : the number of clusters you want to generate.
    :return:
    """
    # Define the list of feedback sentences
    keywords_list = processed_df.keyword_eng.to_list()
    ID = processed_df.id.to_list()
    # Generate sentence embeddings for each feedback sentence
    embeddings = model.encode(keywords_list)

    kmeans = KMeans(n_clusters, random_state=10)
    cluster_assignment = kmeans.fit_predict(embeddings)

    clustered_sentences = {}
    clustered_sentences_id = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        if cluster_id not in clustered_sentences_id:
            clustered_sentences_id[cluster_id] = []

        clustered_sentences[cluster_id].append(keywords_list[sentence_id])
        clustered_sentences_id[cluster_id].append(ID[sentence_id])

    your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
    dft = your_df_from_dict.transpose()
    df_results = pd.melt(dft, value_vars=dft.columns)
    df_results.dropna(inplace=True)
    df_results.rename(columns={'variable': 'cluster', 'value': 'keyword_eng'}, inplace=True)
    # we need sentence id.
    your_id = pd.DataFrame.from_dict(clustered_sentences_id, orient='index')
    dt_id = your_id.transpose()
    df_id = pd.melt(dt_id, value_vars=dt_id.columns)
    df_id.dropna(inplace=True)
    # Assign now the id to the clustered data:
    df_results['id'] = df_id['value'].astype(int)
    df_results = df_results[['id', 'keyword_eng', 'cluster']]
    return df_results

def GENERATING_LABELS(df_clusters, n):
    """
    This function will cluster generate labels for each cluster.
    :param df_clusters : clustered dataframe.
    :param n : the number of words each label will contain.
    :return:
    """
    print('GENERATING LABELS')
    df = df_clusters.copy()
    clusters_amount = max(df.cluster.unique())
    for num_cl in range(clusters_amount + 1):
        keyword_label = labelling_clusters(df, cluster_num=num_cl, n=n)
        cl_lables = ''.join(keyword_label)
        df.loc[df.cluster == num_cl, 'labels'] = cl_lables

    return df

def TOPICS_CLUSTERING(df_clusters):
    """
    This function will cluster the TOPICS and label them.
    :param df_clusters:
    :return:
    """
    # df_clusters = dk_1.copy()
    df_clusters.reset_index(drop=True, inplace=True)
    clusters_labels = df_clusters.labels
    ID = df_clusters.id.to_list()
    model_name_topics = 'all-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name_topics)
    corpus_embeddings = embedder.encode(clusters_labels)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform AgglomerativeClustering clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    clustered_sentences_id = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        if cluster_id not in clustered_sentences_id:
            clustered_sentences_id[cluster_id] = []

        clustered_sentences[cluster_id].append(clusters_labels[sentence_id])
        clustered_sentences_id[cluster_id].append(ID[sentence_id])

    your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
    dft = your_df_from_dict.transpose()

    df_results = pd.melt(dft, value_vars=dft.columns)
    df_results.dropna(inplace=True)
    df_results.rename(columns={'variable': 'cluster', 'value': 'keyword_eng'}, inplace=True)
    df_clusters_final = GENERATING_LABELS(df_results, n=1)
    df_clusters_final.rename(columns={'cluster': 'TOPIC_ID', 'keyword_eng': 'SUBTOPIC', 'labels': 'TOPIC'},
                             inplace=True)

    # we need sentence id.
    your_id = pd.DataFrame.from_dict(clustered_sentences_id, orient='index')
    dt_id = your_id.transpose()
    df_id = pd.melt(dt_id, value_vars=dt_id.columns)
    df_id.dropna(inplace=True)
    # Assign now the id to the clustered data:
    df_clusters_final['id'] = df_id['value'].astype(int)
    df_clusters_final = df_clusters_final[['id', 'SUBTOPIC', 'TOPIC', 'TOPIC_ID']]

    return df_clusters_final


def THEMES_GENERATOR(df):
    'function to generate topics for each sentence/feedback and cluster the topics, hence clustering the feedback'

    sentences = df.keyword_eng.to_list()
    topics = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(pos_tags, binary=False)

        # Collect all NER chunks
        ner_chunks = []
        for subtree in tree.subtrees():
            if subtree.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
                ner_chunks.append(' '.join([word for word, tag in subtree.leaves()]))

        # Collect all noun phrases
        noun_phrases = []
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('NN') or tag == 'VBZ' or tag == 'WP' or tag == 'NNS' or tag == 'VB':
                if i > 0 and pos_tags[i - 1][1] == 'DT':
                    noun_phrases.append(' '.join([pos_tags[i - 1][0], word]))
                elif i > 0 and pos_tags[i - 1][1] == 'NN':
                    noun_phrases.append(' '.join([pos_tags[i - 1][0], word]))
                elif i > 1 and pos_tags[i - 2][1] == 'DT':
                    noun_phrases.append(' '.join([pos_tags[i - 2][0], pos_tags[i - 1][0], word]))
                elif i > 1 and pos_tags[i][1] == 'NN':
                    noun_phrases.append(' '.join([pos_tags[i - 2][0], pos_tags[i - 1][0], word]))
                elif i > 1 and pos_tags[i][1] == 'NNS':
                    noun_phrases.append(' '.join([pos_tags[i - 2][0], pos_tags[i - 1][0], word]))
                elif i > 1 and pos_tags[i][1] == 'VBZ':
                    noun_phrases.append(' '.join([pos_tags[i - 2][0], pos_tags[i - 1][0], word]))
                elif i > 1 and pos_tags[i][1] == 'VB':
                    noun_phrases.append(' '.join([pos_tags[i - 2][0], pos_tags[i - 1][0], word]))

        # Combine NER chunks and noun phrases, and keep only the top two
        all_topics = ner_chunks + noun_phrases
        sorted_topics = sorted(all_topics, key=lambda x: -len(x))
        top_topics = sorted_topics[:1]
        topics.append(top_topics)

    topics_list = [','.join(sublist) for sublist in topics]
    topics_list_low = [element.lower() for element in topics_list]

    # initialize data of lists.
    data = {'id': df.id.to_list(),
            # 'keyword_eng': sentences,
            'themes': topics_list_low}
    topic_df = pd.DataFrame(data)
    results_df = df.merge(topic_df, on='id', how='left')
    return results_df


def RESULTS_OUTPUT(df_final, first_columns):
    """"
    FUNCTION:  will adjsut the order of your columns.
    :param df_clusters_final : final clustered data.
    :param  first_column_names : columns that you want to be at first order.
    :return : clean result output. columns in order.
    """

    # get the current column names
    column_names = list(df_final.columns)

    # remove the specified columns from the list (if they exist)
    for column_name in first_columns:
        if column_name in column_names:
            column_names.remove(column_name)

    # insert the specified columns at the beginning of the list (in reverse order)
    for column_name in reversed(first_columns):
        column_names.insert(0, column_name)

    # reorder the columns using the reindex method
    df = df_final.reindex(columns=column_names)
    return df

def AGGLOMERATIVE(processed_df,selected_model):
    """ this function we will cluster the feedback/sentences/keywords without pre-specifying the number of clusters...
       hence different approach then previous algo..."""


    feedback_list = processed_df.keyword_eng.to_list()
    ID = processed_df.id.to_list()
    embedder = SentenceTransformer(selected_model)
    corpus_embeddings = embedder.encode(feedback_list)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform k-means clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    clustered_sentences_id = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        if cluster_id not in clustered_sentences_id:
            clustered_sentences_id[cluster_id] = []

        clustered_sentences[cluster_id].append(feedback_list[sentence_id])
        clustered_sentences_id[cluster_id].append(ID[sentence_id])

    your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
    dft = your_df_from_dict.transpose()
    df_results = pd.melt(dft, value_vars=dft.columns)
    df_results.dropna(inplace=True)
    df_results.rename(columns={'variable': 'cluster', 'value': 'keyword_eng'}, inplace=True)
    # we need sentence id.
    your_id = pd.DataFrame.from_dict(clustered_sentences_id, orient='index')
    dt_id = your_id.transpose()
    df_id = pd.melt(dt_id, value_vars=dt_id.columns)
    df_id.dropna(inplace=True)
    # Assign now the id to the clustered data:
    df_results['id'] = df_id['value'].astype(int)
    df_results = df_results[['id', 'keyword_eng', 'cluster']]
    return df_results


def CLUSTERING_K_MEANS_TRANSFORMATION(processed_data, start_cluster, end_cluster, steps):
    dic = {}

    for cl_num in range(start_cluster, end_cluster, steps):
        try:

            df_0 = K_MEANS_TRANSFORMATION(processed_df=processed_data, n_clusters=cl_num)
            df_1 = GENERATING_LABELS(df_0, n=2)
            df_2 = TOPICS_CLUSTERING(df_1)
            clusters_df = df_2.merge(processed_data, on='id', how='left')
            first_column_names = ['id', 'keyword', 'keyword_eng', 'SUBTOPIC', 'TOPIC', 'TOPIC_ID']
            topics_df = THEMES_GENERATOR(clusters_df)
            dic[cl_num] = RESULTS_OUTPUT(topics_df, first_column_names)

        except Exception as e:
            print(e)
            continue

    return dic

def CLUSTERING_AGGLOMERATIVE(processed_df, selected_model):
    """This function is to perform clustering based on AGGLOMERATIVE algorithm
    :param : processed_df processed original data.
    :param : selected_model for performing clustering
    """
    dic = {}
    df_0 = AGGLOMERATIVE(processed_df , selected_model)
    df_1 = GENERATING_LABELS(df_0, n=2)
    df_2 = TOPICS_CLUSTERING(df_1)
    clusters_df = df_2.merge(processed_df, on='id', how='left')
    first_column_names = ['id', 'keyword', 'keyword_eng', 'SUBTOPIC', 'TOPIC', 'TOPIC_ID']
    topics_df = THEMES_GENERATOR(clusters_df)
    dic['aglomerative'] = RESULTS_OUTPUT(topics_df, first_column_names)

    return dic


def option_to_model(level_number,options):
  try:
    return options[level_number]
  except Exception as e:
    return e

def dfs_xlsx(data_list,selected_encoding):
    output = BytesIO()
    # writer = pd.ExcelWriter(output, engine='xlsxwriter')
    writer = pd.ExcelWriter(output, engine='xlsxwriter',options={'encoding': selected_encoding})
    for sheet_name, df in data_list.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    results_data = output.getvalue()
    return results_data



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

option_encoding= {
    "UTF-8": 'utf-8',
    "LATIN": 'latin-1'
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
    st.sidebar.text('version Feb 2023')

    st.warning("Please ensure that your data includes the column **KEYWORD** :eye-in-speech-bubble: ")
    encoding_name = ["UTF-8" , "LATIN"]

    select_box = st.selectbox('SELECT AN APPROPRIATE ENCODING', options=encoding_name)
    selected_encoding = option_to_model(select_box,option_encoding)

    uploaded_file_cl = st.file_uploader("Upload data I", type=['csv'])

    if uploaded_file_cl is not None:
        file_contents = uploaded_file_cl.getvalue()
        file = io.StringIO(file_contents.decode(selected_encoding))
        try:
            dialect = csv.Sniffer().sniff(file.read(1024))
            file.seek(0)
            reader = csv.reader(file, dialect)
            headers = next(reader)

        except:
            file.seek(0)
            reader = csv.reader(file, delimiter=',')
            headers = next(reader)

        keywords_df = pd.read_csv(file, delimiter=dialect.delimiter, header=None, names=headers).dropna(axis=1,  how='all')
        st.dataframe(keywords_df)





    model_name = ["<select>", "General Base", "General Roberta", "General miniML_L12", "General miniML_L6",
                  "Medics", "Education and training", "Finance"]

    select_box = st.selectbox('SELECT A MODEL AND RUN ONE OF THE NEXT ALGORITHMS', options=model_name)
    selected_option = option_to_model(select_box,option_models)

    load_trans_K_means = st.button('GENERATE CLUSTERS: TRANSFORMERS & K-MEANS' )

    if load_trans_K_means:
        long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)

        with st.spinner('**The TRANSFORMERS clustering algorithm is currently in operation. Please hold on ...**'):

            model = SentenceTransformer(selected_option)

            max_cluster = max(3,np.trunc(keywords_df.shape[0] * 0.1).astype(int))
            min_cluster = max(1,np.trunc(max_cluster / 2).astype(int))
            steps = max(1,np.trunc((max_cluster - min_cluster) / 3).astype(int))

            data_list = CLUSTERING_K_MEANS_TRANSFORMATION(processed_data, start_cluster=min_cluster,
                                                          end_cluster=max_cluster, steps=steps)

            preffix = 'CLUSTER_id_'
            new_dict = {(preffix + str(key)): value for key, value in data_list.items()}
            data_list = new_dict

            df_xlsx = dfs_xlsx(data_list, selected_encoding)

            st.write("""
            <p style="background-color: #FEC929; color: black; padding: 10px;"> 
            Further examination is recommended for the subsequent clusters..
            </p>
            """, unsafe_allow_html=True)

            st.subheader("Download data")
            ste.download_button(label='Download Results',
                               data=df_xlsx,
                               file_name='TRANSFOMERS_K_MEANS.xlsx')

    ### AGGLOMERATIVE:

    load_agglomerative = st.button('GENERATE CLUSTERS: AGGLOMERATIVE')

    if load_agglomerative and select_box != '<select>':
        st.write('You selected model:', selected_option)
        long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)

        with st.spinner('**The AGGLOMERATIVE Model clustering algorithm is currently running. Please hold on...**'):

            data_list = CLUSTERING_AGGLOMERATIVE(processed_df=processed_data, selected_model=selected_option)
            preffix = 'id_1'
            new_dict = {(str(key) + preffix ): value for key, value in data_list.items()}
            data_list = new_dict
            df_xlsx = dfs_xlsx(data_list, selected_encoding)

            st.write("""
                        <p style="background-color: #FEC929; color: black; padding: 10px;">
                        Further examination is recommended for the subsequent clusters..
                        </p>
                        """, unsafe_allow_html=True)

            st.subheader("Download data")
            ste.download_button(label='Download Results',
                               data=df_xlsx,
                               file_name='Agglomerative.xlsx')




elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')


if st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')







