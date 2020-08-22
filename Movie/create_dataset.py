import pandas as pd 
import numpy as np 
from genre_wise import genre_res
from popular_movies import popular_movies

genres= ['Action', 'Thriller', 'Drama', 'Crime', 'Science Fiction', 'Adventure', 'Fantasy', 'Romance', 'Comedy', 'Mystery', 'Family']

column_names = ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'overview']

final_dataframe = pd.DataFrame(columns = column_names)

popular_movies= popular_movies()
final_dataframe = final_dataframe.append(popular_movies)
for genre in genres:
	movies= genre_res(genre)
	print(genre)
	print(movies.size)
	print(movies.shape)
	final_dataframe = final_dataframe.append(movies)
	print(final_dataframe.shape)
	movies.to_csv('%s.csv',genre)
	print("----------------------------------------------------------------------------------")
	


print(final_dataframe)

print(final_dataframe.shape)
#remove duplicates
#final_dataframe.drop_duplicates(inplace=True) 

final_dataframe.loc[final_dataframe.astype(str).drop_duplicates().index]
print(final_dataframe.shape)
#the list elements are still list in the final results.
final_dataframe.loc[final_dataframe.astype(str).drop_duplicates().index].loc[0,'title']
print(final_dataframe.shape)
#convert to csv
final_dataframe.to_csv('dataset_short.csv') 

