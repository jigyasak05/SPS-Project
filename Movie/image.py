import requests
import pprint
import json

def get_poster(name):

	url = "https://imdb8.p.rapidapi.com/title/find"

	querystring = {"q": name}

	headers = {
	    'x-rapidapi-host': "imdb8.p.rapidapi.com",
	    'x-rapidapi-key': "9fc5fa1be3msh6dda2679ff9b9a5p1c27afjsn70a89581d46a"
	    }

	response = requests.request("GET", url, headers=headers, params=querystring)

	#print(type(response))
	#print(response["results"])
	x= (response.text)
	dictionary = json.loads(x)
	return dictionary["results"][0]['image']['url']