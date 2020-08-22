from flask import Flask, render_template, url_for, flash, redirect
from popular_movies import popular_movies
from image import get_poster

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]
popular_movies= popular_movies()


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=popular_movies)


@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/detail/<int:index>", methods=['GET', 'POST'])
def detail(index):
    movie=popular_movies[index]
    print(movie['title'])
    link= get_poster(movie['title'])
    print(link)
    return render_template('detail.html', title= movie['title'], post=movie, link=link)





if __name__ == '__main__':
    app.run(debug=True)
