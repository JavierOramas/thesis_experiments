from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def davies_blouldin_score(df_embedding_no_out, clusters_predict, verbose=False):
    """
    The Davies Bouldin index is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances.
    The minimum value of the DB Index is 0, whereas a smaller value (closer to 0) represents a better model that produces better clusters.
    """
    score = davies_bouldin_score(df_embedding_no_out,clusters_predict)
    
    if verbose:
        print(f"Davies bouldin score: {score}")
    
    return score

def davies_blouldin_score(df_embedding_no_out, clusters_predict, verbose=False):
    """
    Calinski Harabaz Index -> Variance Ratio Criterion.
    Calinski Harabaz Index is defined as the ratio of the sum of between-cluster dispersion and of within-cluster dispersion.
    The higher the index the more separable the clusters.
    """
    score = calinski_harabasz_score(df_embedding_no_out,clusters_predict)
    
    if verbose:
        print(f"Calinski Score: {score}")
    
    return score

def davies_blouldin_score(df_embedding_no_out, clusters_predict, verbose=False):
    """
    The silhouette score is a metric used to calculate the goodness of fit of a clustering algorithm, but can also be used as a method for determining an optimal value of k (see here for more).
    Its value ranges from -1 to 1.
    A value of 0 indicates clusters are overlapping and either the data or the value of k is incorrect.
    1 is the ideal value and indicates that clusters are very dense and nicely separated.
    """
    score = silhouette_score(df_embedding_no_out,clusters_predict)
    
    if verbose:
        print(f"Silhouette Score: {score}")
    
    return score