from flask import Flask, render_template, url_for, flash, redirect
from popular_movies import popular_movies
from similar_movies import get_similar_movies
from image import get_poster
import pandas as pd
import pprint

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

df=pd.read_csv('data/dataset_short.csv')
popular_movies= popular_movies()
#print(type(popular_movies[0]))

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=popular_movies, category='Popular Movies')


@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/detail/<int:id>", methods=['GET', 'POST'])
def detail(id):
    #print(df.loc[df['id'] == id])
    #df[df['id']==27205].to_dict('records')[0]
    movie=df.loc[df['id'] == id].to_dict(orient='records')[0]
    #pprint.pprint(movie)
    #print(movie['title'])
    link= get_poster(movie['title'])
    #print(link)
    return render_template('detail.html', title= movie['title'], post=movie, link=link)

@app.route("/similar_movies/<movie_name>")
def similar_movies(movie_name):
    similar_movies = get_similar_movies(movie_name)
    #print(len(similar_movies))
    #pprint.pprint(similar_movies[0])
    #print(movie['title'])
    return render_template('similar_movies.html', title= movie_name, posts=similar_movies, movie_name=movie_name)

@app.route("/filter_movies/<genre>")
def filter(genre):
    if(genre=='All'):
        genre_df=pd.read_csv(f'data/dataset_short.csv')
        genre_df=genre_df.head(200)
    else:    
        genre_df=pd.read_csv(f'data/genre_{genre}.csv')

    filter_results= genre_df.to_dict(orient='records')
    return render_template('home.html', posts=filter_results, category=genre)






if __name__ == '__main__':
    app.run(debug=True)
