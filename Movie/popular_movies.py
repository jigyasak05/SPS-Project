import pandas as pd 
import numpy as np 
from ast import literal_eval

df=pd.read_csv('data/movies_metadata.csv')
df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.95)

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


def popular_movies():
	
	df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
	qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres', 'overview']]
	qualified['vote_count'] = qualified['vote_count'].astype('int')
	qualified['vote_average'] = qualified['vote_average'].astype('int')
	qualified['wr'] = qualified.apply(weighted_rating, axis=1)
	qualified = qualified.sort_values('wr', ascending=False).head(250)
	result= qualified.head(100)
	ans= result.to_dict(orient='records')
	return ans


#if __name__ == '__main__':
#	main()