




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
            # df = df[['id', 'keyword']].copy()
            df.dropna(inplace=True)
            # Adding 'digit-' prefix for the rows that contains digits only as GoogleTranslator can not
            # translate digits only.
            df["keyword"] = df["keyword"].apply(lambda x: 'digit-' + x if x.isdigit() else x)
            # print("The keywords are in the process of being translated to ENGLISH. Please hold on ... ")
            # my_list = df["keyword"].to_list()
            # df["Keyword_eng"] = translate_to_english(my_list)
            df["keyword_eng"] = df["keyword"].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(x))
            df = df.mask(df.eq('None')).dropna()  # remove NONE that was produced when trying to translate strange
                                                  # characters like :"????"
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

def topics_generator(df, df_clusters, clusters_labels):
    """
    This function is
    :param df: keywords dataframe
    :param df_clusters:
    :param clusters_labels:
    :return:
    """

    # model_name_topics = 'paraphrase-MiniLM-L6-v2'
    model_name_topics = 'all-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name_topics)
    corpus_embeddings = embedder.encode(clusters_labels)

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

        clustered_sentences[cluster_id].append(clusters_labels[sentence_id])

    your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
    dft = your_df_from_dict.transpose()

    df_1 = pd.melt(dft, value_vars=dft.columns)
    df_1.dropna(inplace=True)
    df_1.rename(columns={'variable': 'TOPICS', 'value': 'SUB_TOPICS'}, inplace=True)

    df_clusters.rename(columns={'labels': 'SUB_TOPICS'}, inplace=True)
    df_clusters_pre = df_clusters.merge(df_1, on='SUB_TOPICS', how='left')
    df_clusters_final = df_clusters_pre.merge(df[['id', 'keyword']], on='id', how='left')
    df_clusters_final = df_clusters_final[['id', 'keyword_eng', 'semantic_score', 'SUB_TOPICS', 'TOPICS']]
    topics = df_clusters_final.TOPICS.max()
    print('*** GENERATING {} TOPICS USING AGGLOMERATIVE CLUSETERING '.format(topics))
    return df_clusters_final

def clusters_generator_cosine(df,  labels):
    print('*** Keyword clustering using SENTENCE TRANSFORMERS...')
    df.reset_index(drop=True, inplace=True)
    sentences1 = labels
    sentences2 = df.keyword_eng
    id = df.id

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    score = []
    SEN_1 = []
    SEN_2 = []
    ID = []

    # Output the pairs with their score
    for i in range(len(sentences2)):
        for j in range(len(sentences1)):
            score.append(cosine_scores[j][i].item())
            SEN_1.append(sentences1[j])
            SEN_2.append(sentences2[i])
            ID.append(id[i])

    # initialize data of lists.
    data = {'id': ID,
            'semantic_score': score,
            'labels': SEN_1,
            'keyword_eng': SEN_2
            }

    df = pd.DataFrame(data)
    dt = df.loc[df.groupby(['keyword_eng', 'id'])['semantic_score'].idxmax()]
    dt.sort_values('labels', inplace=True)
    return (dt)

def CLUSTERING_K_MEANS(processed_df, long_tail_df, short_tail_df, start_cluster, end_cluster, steps, cutoff):
    'this function uses CountVectorizer (words frequency) then kmeans'

    global num_cl
    textlist = long_tail_df.keyword_eng.to_list()
    textlist_stem = stemmList(textlist)
    text_data = pd.DataFrame(textlist_stem)
    # Bag of words
    vectorizer_cv = CountVectorizer(analyzer='word')
    X_cv = vectorizer_cv.fit_transform(textlist_stem)
    dic = {}
    LABELS = {}

    for cl_num in range(start_cluster, end_cluster, steps):

        try:
            kmeans = KMeans(n_clusters=cl_num, random_state=10)
            kmeans.fit(X_cv)
            result = pd.concat([text_data, pd.DataFrame(X_cv.toarray(), columns=vectorizer_cv.get_feature_names_out())], axis=1)
            result['cluster'] = kmeans.predict(X_cv)
            result.rename(columns={0: 'Keyword_ENG_stemmed'}, inplace=True)
            df_results = result[['Keyword_ENG_stemmed', 'cluster']].copy()
            df_results.insert(0, "id", long_tail_df.id.values, True)
            df_results.insert(1, "keyword_eng", textlist, True)

            for num_cl in range(cl_num + 1):
                keyword_label = labelling_clusters(df_results, cluster_num=num_cl, n=2)
                cl_lables = ''.join(keyword_label)
                df_results.loc[df_results.cluster == num_cl, 'labels'] = cl_lables

            # Similarity score calculation between LABELS AND KEYWORD ENGLISH.
            # to have an idea how far a keyword from specific cluster.
            sentences1 = df_results.labels
            sentences2 = df_results.keyword_eng

            # Compute embedding for both lists
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)

            # Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            y = []
            # Output the pairs with their score
            for i in range(len(sentences1)):
                y.append(cosine_scores[i][i].item())

            df_results['semantic_score'] = y

            df_results.drop(['Keyword_ENG_stemmed', 'cluster'], inplace=True, axis=1)
            labels = df_results.labels.unique()
            if short_tail_df.shape[0] != 0 :
                df_clusters = clusters_generator_cosine(short_tail_df, labels=labels)
                clusters_short = df_clusters[df_results.columns.values.tolist()]
                df_clusters_all = pd.concat([df_results, clusters_short], ignore_index=True)
            else:
               df_clusters_all = df_results

            # Generating some statistics:
            z = df_clusters_all.groupby(['labels'])['semantic_score'].mean()
            A = z[z < cutoff]
            final_clusters = topics_generator(df=long_tail_df, df_clusters=df_clusters_all, clusters_labels=labels)
            # Adding columns van original data to the results
            df_org = processed_df.drop(['keyword_eng'], axis=1)
            final_clusters = final_clusters.merge(df_org, on='id', how='left')
            dic[cl_num] = final_clusters
            LABELS[cl_num] = A.index.values

        except Exception as e:
            print(e)
            continue

    return (dic,LABELS)

def K_MEANS_TRANSFORMATION(processed_df,  start_cluster, end_cluster, steps):
    'function that uses the transfomers then k-means'

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
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                if cluster_id not in clustered_sentences:
                    clustered_sentences[cluster_id] = []

                clustered_sentences[cluster_id].append(feedback_list[sentence_id])


            your_df_from_dict = pd.DataFrame.from_dict(clustered_sentences, orient='index')
            dft = your_df_from_dict.transpose()
            df_results = pd.melt(dft, value_vars=dft.columns)
            df_results.dropna(inplace=True)
            df_results.rename(columns={'variable': 'cluster', 'value': 'keyword_eng'}, inplace=True)
            df_results['id'] = range(len(df_results))
            clusters_amount =  max(df_results.cluster.unique())

            for num_cl in range(clusters_amount + 1):
                keyword_label = labelling_clusters(df_results, cluster_num=num_cl, n=2)
                cl_lables = ''.join(keyword_label)
                df_results.loc[df_results.cluster == num_cl, 'labels'] = cl_lables

            print('GENERATING THEMES')
            dic[cl_num] = THEMES_GENERATOR(df_results)

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
    results_data = output.getvalue()
    return results_data

def AgglomerativeClustering_algo(model_name_topics,df):
    """this function we will cluster the feedback/sentences/keywords without pre-specifying the number of clusters...
       hence different approach then previous algo..."""

    dic = {}
    feedback_list = df.keyword_eng.to_list()
    embedder = SentenceTransformer(model_name_topics)
    corpus_embeddings = embedder.encode(feedback_list)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform k-means clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
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
    df_results.rename(columns={'variable': 'cluster', 'value': 'keyword_eng'}, inplace=True)
    df_results['id'] = range(len(df_results))
    clusters_amount = max(df_results.cluster.unique())

    for num_cl in range(clusters_amount + 1):
        keyword_label = labelling_clusters(df_results, cluster_num=num_cl, n=2)
        cl_lables = ''.join(keyword_label)
        df_results.loc[df_results.cluster == num_cl, 'labels'] = cl_lables

    print('GENERATING THEMES')
    dic['Agglomerative_clusters'] = THEMES_GENERATOR(df_results)
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
    encoding_name = ["UTF-8" , "LATIN" ]

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
        except:
            file.seek(0)
            reader = csv.reader(file, delimiter=',')
        data = [row for row in reader]
        keywords_df = pd.DataFrame(data[1:], columns=data[0])
        st.dataframe(keywords_df)

    # if uploaded_file_cl is not None:
    #     # file_buffer = io.StringIO(uploaded_file_cl.getvalue().decode("utf-8"))
    #     with open(uploaded_file_cl, newline='', encoding=selected_encoding) as file:
    #         dialect = csv.Sniffer().sniff(file.read(1024))
    #         file.seek(0)
    #         reader = csv.reader(file, dialect)
    #         data = [row for row in reader]
    #     keywords_df = pd.DataFrame(data[1:], columns=data[0])

        # keywords_df = pd.read_csv(uploaded_file_cl, encoding=selected_option)
        # st.dataframe(keywords_df)


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

            data_list = K_MEANS_TRANSFORMATION(processed_data, start_cluster=min_cluster, end_cluster=max_cluster, steps=steps)
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
                               file_name='TRANSFOMERS_K_MEANS.xlsx')

    ### AGGLOMERATIVE:

    load_agglomerative = st.button('GENERATE CLUSTERS: AGGLOMERATIVE')

    if load_agglomerative and select_box != '<select>':
        st.write('You selected model:', selected_option)
        long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)

        with st.spinner('**The AGGLOMERATIVE Model clustering algorithm is currently running. Please hold on...**'):


            data_list = AgglomerativeClustering_algo(selected_option, processed_data)

            preffix = 'id_'
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
                               file_name='Agglomerative.xlsx')




elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')


if st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')





#old versions:
    # load_K_means = st.button('GENERATE CLUSTERS: K-MEANS' )
    #
    # if load_K_means:
    #     long_tail_df, short_tail_df, processed_data = data_preprocessing(keywords_df)
    #     with st.spinner('**The K-MEANS clustering algorithm is currently in operation. Please hold on ...**'):
    #
    #         model_name = 'all-MiniLM-L6-v2'
    #         model = SentenceTransformer(model_name)
    #
    #         max_cluster = max(3,np.trunc(keywords_df.shape[0] * 0.1).astype(int))
    #         min_cluster = max(1,np.trunc(max_cluster / 2).astype(int))
    #         steps = max(1,np.trunc((max_cluster - min_cluster) / 3).astype(int))
    #
    #         cut_off = 0.5
    #         data_list, labs = CLUSTERING_K_MEANS(processed_data, long_tail_df, short_tail_df, start_cluster=min_cluster,
    #                                        end_cluster = max_cluster, steps=steps, cutoff=cut_off)
    #
    #
    #         preffix = 'CLUSTER_id_'# ff
    #         new_dict = {(preffix + str(key)): value for key, value in data_list.items()}
    #         data_list = new_dict
    #
    #         new_labs = {(preffix + str(key)): value for key, value in labs.items()}
    #         labs = new_labs
    #         noisy_clusters = pd.DataFrame.from_dict(labs, orient='index')
    #         noisy_clusters = noisy_clusters.transpose()
    #         noisy_clusters = noisy_clusters.fillna(value='')
    #         data_list['Noisy_clusters'] = noisy_clusters
    #         df_xlsx = dfs_xlsx(data_list)
    #
    #         st.write("""
    #         <p style="background-color: #FEC929; color: black; padding: 10px;">
    #         Further examination is recommended for the subsequent clusters..
    #         </p>
    #         """, unsafe_allow_html=True)
    #         # st.balloons()
    #
    #         st.dataframe(noisy_clusters)
    #         st.subheader("Download data")
    #         ste.download_button(label='Download Results',
    #                            data=df_xlsx,
    #                            file_name='K_MEANS_clustering.xlsx')










