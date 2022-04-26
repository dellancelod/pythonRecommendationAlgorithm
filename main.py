from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

def get_title_from_index(df, index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(df, title):
    return df[df.title == title]["index"].values[0]


if __name__ == "__main__":
    df = pd.read_csv("movie_dataset.csv") # creating pandas DataFrame object from csv table
    features = ['keywords', 'cast', 'director', 'genres'] # specifying the features that we will compare movies with

    for feature in features:
        df[feature] = df[feature].fillna('') # filling all NaNs(pandas 'missing data' marker) with blank string
   
    # applying combined_features() method over each rows of dataframe and storing the combined string in “combined_features” column
    df["combined_features"] = df.apply(combine_features,axis=1) 
    
    # creating cosine similarity matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])
    cos_sim = cosine_similarity(count_matrix)
    
    # reading movie from console and finding all similarity scores for it
    movie_input = input("Enter a movie to see recommendations for: ")
    movie_index = get_index_from_title(df, movie_input)
    
    # enumerating the row of similarity scores for given movie
    similar_movies = list(enumerate(cos_sim[movie_index]))
    
    # sort similar movies in descending order and discard the first element
    # since it's the given movie
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],
                                   reverse=True)[1:]
    
    # filter 5 similar movies to given
    i=0
    print("Top 5 similar movies to " + movie_input + " are:\n")
    for element in sorted_similar_movies:
        print(get_title_from_index(df, element[0]))
        i = i + 1
        if i > 5:
            break
    
    
    
    
    