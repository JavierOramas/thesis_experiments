
############################### Library Imports #############################################################

from collections import Counter
import pandas as pd 
import numpy as np 

# data visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap

# sklearn 
from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline
# from sklearn.manifold import TSNE
# from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score, classification_report

# visualization
from pyod.models.ecod import ECOD
from yellowbrick.cluster import KElbowVisualizer
import lightgbm as lgb
import prince

# Embeddings for clustering
from pipeline.embeddings.basic_embeddings import Embedding

# Outlier detection
from pyod.models.ecod import ECOD

# NLTK
from nltk.corpus import stopwords
import nltk

# LLM
from ctransformers import AutoModelForCausalLM

################################################################################################

class TopicModel:

    def __init__(self) -> None:
        self.topics = None
        self.n_topics = None
        self.clusters_centroids = None

    def generate_embeddings(self, sentences):
        
        model = Embedding().get_llm()

        output = model.encode(sentences=sentences,
                show_progress_bar=True,
                normalize_embeddings=True)

        df_embedding = pd.DataFrame(output)
        
        return df_embedding

    def detect_outliers(self, df_embedding):
        clf = ECOD()
        clf.fit(df_embedding)


        out = clf.predict(df_embedding) 
        df_embedding["outliers"] = out

        # df["outliers"] = out

        df_embedding_no_out = df_embedding[df_embedding["outliers"] == 0]
        df_embedding_no_out = df_embedding_no_out.drop(["outliers"], axis = 1)


        df_embedding_with_out = df_embedding.copy()
        df_embedding_with_out = df_embedding_with_out.drop(["outliers"], axis = 1)

        return df_embedding_no_out, df_embedding_with_out

    def detect_optimal_k(self, df, lim_sub=2, lim_sup=15, visualize=False):
        km = KMeans(init="k-means++", random_state=0, n_init="auto")
        visualizer = KElbowVisualizer(km, k=(lim_sub, lim_sup), locate_elbow=True)
 
        visualizer.fit(df)
        
        if visualize:    # Fit the data to the visualizer
            visualizer.show()
        
        self.clusters_centroids = visualizer.elbow_value_
        return visualizer.elbow_value_

    def topic_words(self, sentences, clusters, k):
        
        #  we will strip stopwords for a better topic model representation
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))


        data = pd.DataFrame(columns=["sentences", "topic"], data=zip(sentences, clusters))
        topic_bow = {}

                # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
        llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id="/mnt/DATA/THESIS/playground/models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
            model_file="/mnt/DATA/THESIS/playground/models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
            model_type="mistral", 
            gpu_layers=20)
        
        for i in range(0, k):
            # Initialize a Counter for each topic
            topic_bow[i] = Counter()
            for j in data[data.topic == i].sentences:
                # Tokenize the sentence
                words = j.split()
                # Remove stop words and punctuation
                words = [word for word in words if word not in stop_words and word.isalpha()]
                # Add the words to the topic bag of words
                topic_bow[i].update(words)


            # Get the 10 most common words in the topic
            topic_bow[i] = Counter(topic_bow[i]).most_common(20)
            
        for idx,i in topic_bow.items():
            text = ", ".join(j[0] for j in i if isinstance(j[0], str) )

            topic = llm(f"""
                        You will be given some words and you should create a topic name to match those words. No number or blank string is allowed.
                        
                        text: PC, laptop, monitor, AI, internet.
                        topic: tech.
                        
                        text: hospital, aspirin, blood, injection, doctor
                        topic: Medicine.

                        text: {text},
                        topic: """, repetition_penalty=100.0)
            print(">", topic)
            topic_bow[idx] = (i, topic)
        
        return topic_bow
    
    def get_topics(self, sentences):

        #  Embed sentences
        df_embedding = self.generate_embeddings(sentences)

        # detect outliers 
        df_embeddings_no_out, df_embeddings_with_out = self.detect_outliers(df_embedding)

        # find the optimal number of topics in the corpus (Elbow)
        optimal_k = self.detect_optimal_k(df_embeddings_with_out)

        print("Optimal number of topics", optimal_k)

        # Create clusters
        clusters = KMeans(n_clusters=optimal_k, init = "k-means++").fit(df_embeddings_no_out)
        self.clusters_centroids = clusters.cluster_centers_
        
        clusters_predict = clusters.predict(df_embeddings_no_out)

        topics = self.topic_words(sentences, clusters_predict, optimal_k)

        self.topics = topics

        return topics