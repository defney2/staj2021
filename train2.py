import pandas as pd
import numpy as np
import seaborn as sns
import tweepy
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

columns = ['statuses_count', 'followers_count', 'friends_count',
'favourites_count', 'listed_count', 'default_profile',
'profile_use_background_image', 'verified', 'tweet_freq',
'followers_growth_rate', 'friends_growth_rate',
'favourites_growth_rate', 'listed_growth_rate',
'followers_friend_ratio', 'screenname_length',
'num_digits_in_screenname', 'name_length', 'name_digits_in_name',
'description_length', 'isbot']

dataset = pd.read_csv("dataframee2.csv", usecols = columns)

def binary_map(x):
    return x.map({True: 1, False: 0})

altered_var = ['default_profile',
'profile_use_background_image',
'verified']

dataset[altered_var] = dataset[altered_var].apply(binary_map)
dataset.head()
dataset.columns
dataset.shape
dataset.describe()
dataset.isnull().sum()
#create new dataset where the null values are dropped and check that there are no null values left
dataset1 = dataset.dropna()
dataset1.isnull().sum()

train = dataset1.drop(['isbot'], axis = 1)
test = dataset1['isbot']

X_train, X_test, y_train, y_test = train_test_split(train, test, train_size = 0.8, test_size = 0.2, random_state = 2)
regr = LinearRegression()
regr.fit(X_train, y_train)
pred = regr.predict(X_test)

#since linear regression gave a very low accuracy score I moved onto try a different model: the random forest model
regr.score(X_test, y_test)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


metrics.accuracy_score(y_test, y_pred)


consumer_key = "783ZxxFfcUkh4y5eG88JtgVG0"
consumer_secret = "10JlCd50Hpw0yICiTsHnYgNfcdiQGKxtDMM8ZRnpNsASPVHmug"
access_token = "921665261853466624-sYD4ZDpFA8CNHyeeKKwL2ZA1wwLrBtp"
access_token_secret = "TFRNyf6zlsvkKGi1j91bPdPr9LpRZ42Ta0TG7SLXlPmrm"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

val = input("Enter the twitter account screenname: ")

user = api.get_user(screen_name=val)
screenname = user.screen_name
name = user.name

statuses_count = user.statuses_count
followers_count = user.followers_count
friends_count = user.friends_count
favourites_count = user.favourites_count
listed_count = user.listed_count
default_profile = user.default_profile
if default_profile == True:
    default_profile = 1
else:
    default_profile = 0
profile_use_background_image = user.profile_use_background_image
if profile_use_background_image == True:
    profile_use_background_image = 1
else:
    profile_use_background_image = 0
verified = user.verified
if verified == True:
    verified = 1
else:
    verified = 0


current_time = datetime.now(timezone.utc)
created = user.created_at
naive = current_time.replace(tzinfo=None)
diff = current_time - created
sec = diff.total_seconds()
user_age = sec / 3600
    
tweet_freq = user.statuses_count / user_age
followers_growth_rate = user.followers_count / user_age
friends_growth_rate = user.friends_count / user_age
favourites_growth_rate = user.favourites_count / user_age
listed_growth_rate = user.listed_count / user_age
followers_friend_ratio = user.followers_count / user.friends_count
screenname_length = len(screenname)
num_digits_in_screenname = sum(c.isdigit() for c in screenname)
name_length = len(name)
name_digits_in_name = sum(c.isdigit() for c in name)
description_length = len(user.description)

array = [statuses_count, followers_count, friends_count, favourites_count, listed_count, default_profile, profile_use_background_image, verified, tweet_freq, followers_growth_rate, friends_growth_rate, favourites_growth_rate, listed_growth_rate, followers_friend_ratio, screenname_length, num_digits_in_screenname, name_length, name_digits_in_name, description_length ]

clf.predict([[statuses_count, followers_count, friends_count, favourites_count, listed_count, default_profile, profile_use_background_image, verified, tweet_freq, followers_growth_rate, friends_growth_rate, favourites_growth_rate, listed_growth_rate, followers_friend_ratio, screenname_length, num_digits_in_screenname, name_length, name_digits_in_name, description_length ]])

rf_probs1 = clf.predict_proba([[statuses_count, followers_count, friends_count, favourites_count, listed_count, default_profile, profile_use_background_image, verified, tweet_freq, followers_growth_rate, friends_growth_rate, favourites_growth_rate, listed_growth_rate, followers_friend_ratio, screenname_length, num_digits_in_screenname, name_length, name_digits_in_name, description_length ]])[:, 1]

rf_probs0 = clf.predict_proba([[statuses_count, followers_count, friends_count, favourites_count, listed_count, default_profile, profile_use_background_image, verified, tweet_freq, followers_growth_rate, friends_growth_rate, favourites_growth_rate, listed_growth_rate, followers_friend_ratio, screenname_length, num_digits_in_screenname, name_length, name_digits_in_name, description_length ]])[:, 0]

print("Probability human: ", rf_probs0)
print("Probability bot: ", rf_probs1)
