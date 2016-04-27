import requests
from requests_oauthlib import OAuth1
from StringIO import StringIO
import json

fname = "app.config"
CON = json.load(file(fname, 'r'))


class Fetch_news(object):
    def __init__(self, consumer_key=None, consumer_secret=None, access_token_key=None, access_token_secret=None):
        self.auth = OAuth1(consumer_key, consumer_secret, access_token_key, access_token_secret)

    def fetch_tweets(self, params=None):                
        url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
        self.response = requests.get(url, params=params, auth=self.auth, timeout=10)


def main(): 
 

    key = "pNIf0Li8bjKMUsnT4ComTffsv"
    secret = "JaOOLVD5u9FzVWcd1Nh5JV7VhIuIuzK5sGIFuSYsabinbz1dYs"
    token = "30206317-iwHVDope9UlGiIVMfJsfhPxpMEnD5MEx1oS6ipE6J"
    token_secret ="6UAHstx8axzsXINpdGeRKdWDalCeB4t6kvzBLztGhDXf6"
    
    
    news = Fetch_news(key, secret, token, token_secret)
     
    id = {account["screen_name"]: 0 for account in CON["accounts"]}

  
    while True:
        try:
            for account in CON["accounts"]:
                screen_name = account["screen_name"]
                file_name = account["file"]

                params = {'screen_name':screen_name, 'exclude_replies':'true', 'count':'200'}
                if id[screen_name] > 0:
                    params['max_id'] = id[screen_name]
                news.fetch_tweets(params)
                tweets = json.load(StringIO(news.response.content))
               
                if news.response.status_code != 200:
                    print news.response.status_code
                    continue

                with file(file_name, "a+") as output_stream:
                    for t in tweets:
                        output_stream.write(json.dumps(t))
                        output_stream.write("\n")
                        if id[screen_name] == 0:
                            id[screen_name] = t['id']-1
                        else:
                            id[screen_name] = min(id[screen_name], t['id']-1)

        except Exception as e:
            print "Error"


if __name__ == "__main__":
    main()

