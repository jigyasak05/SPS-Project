import pandas as pd
import numpy as np
import scipy
from surprise import Reader, Dataset, SVD
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


def get_similar_movies(title):
    
    df2= pd.read_csv('data/dataset_short.csv')
    

    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    #Replace NaN with an empty string
    df2['overview'] = df2['overview'].fillna('')

    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df2['overview'])
    

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    #Construct a reverse map of indices and movie titles
    indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    movie_names= df2['title'].iloc[movie_indices]

    #get data corresponding to each movie from database:
    column_names = ['id', 'title', 'year', 'vote_count', 'vote_average', 'popularity', 'overview']

    result = pd.DataFrame(columns = column_names)
    for movie_name in movie_names:
        result = result.append(df2.loc[df2['title'] == movie_name])


    #print(result)
    return result.to_dict(orient='records')

    



