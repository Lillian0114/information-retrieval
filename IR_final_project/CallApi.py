import json
import tweepy

# XX please input your oath key
ACCESS_TOKEN = 'XXXX'
ACCESS_SECRET = 'XXXX'
CONSUMER_KEY = 'XXXX'
CONSUMER_SECRET = 'XXXX'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)


api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# TRACKING_KEYWORDS = ['Marketing']
# OUTPUT_FILE = ['Marketing.txt']
TRACKING_KEYWORDS = [['Politics'],['Education'],['Health'],['Marketing'],['Music'],['News'],['Sport'],['Technology'],['Pets'],['Food'],['Family']]
OUTPUT_FILE = ['Politics.txt','Education.txt','Health.txt','Marketing.txt','Music.txt','News.txt','Sport.txt','Technology.txt','Pets.txt','Food.txt','Family.txt']
TWEETS_TO_CAPTURE = 200 
LANGUAGES = ['en']

class MyStreamListener(tweepy.StreamListener):

    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open(OUTPUT_FILE[i], "w")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        
        if self.num_tweets <= TWEETS_TO_CAPTURE:
            if self.num_tweets % 100 == 0: # see some progress...
                print('Numer of {} tweets captured so far: {}'.format(TRACKING_KEYWORDS[i], self.num_tweets))
            return True
        else:
            return False

        self.file.close()

    def on_error(self, status):
        print(status)

for i in range(len(TRACKING_KEYWORDS)):
    l = MyStreamListener(i)

    stream = tweepy.Stream(auth, l)

    # LOCATIONS = [-79.762152, 40.496103,	-71.856214,	45.01585 ]     #new york
    # [43.7188915224134, -79.53740244133613,43.6703,-79.3867 tornoto        

    # Filter Twitter Streams to capture data by the keywords:
    # stream.filter(track=TRACKING_KEYWORDS[i], languages=LANGUAGES,locations=LOCATIONS)
    stream.filter(track=TRACKING_KEYWORDS[i], languages=LANGUAGES)