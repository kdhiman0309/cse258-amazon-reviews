# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:27:10 2017

@author: kdhiman
"""

# In[]
import gzip
import numpy as np
from collections import defaultdict
import scipy.optimize
from sklearn.utils import shuffle
import string
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import copy
from nltk.sentiment.vader import allcap_differential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.stem.porter import *

# In[]
count_pun = lambda l1,l2: sum([1 for x in l1 if x in l2])

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])
# In[]
def round(x):
    return np.round(x)
    #return x
# In[]
# In[]
def f_mae(theta, X, y, lam):
  theta = theta.reshape((len(theta),1))

  error = np.sum(np.abs(y - np.dot(X,theta))); # MAE
  #error =  error/len(y) + lam * np.sum(np.abs(theta)) #L1
  error =  error/len(y) + lam * np.sum(np.abs(theta*theta)) #L2
  return error
# NEGATIVE Derivative of log-likelihood
def fprime_mae(theta, X, y, lam):
  theta = theta.reshape((len(theta),1))
  dl = np.dot(np.sign(np.dot(X, theta) - y).T, X).T #MAE
  #dl = 2 * np.dot((np.dot(X, theta) - y).T, X).T #MSE
  dl = dl/len(y)
  #dl += lam * np.sign(theta); #L1
  dl += 2* lam * (theta); #L2
  return dl

# In[]
def f_mse(theta, X, y, lam):
  theta = theta.reshape((len(theta),1))

  #error = np.sum(np.abs(y - np.dot(X,theta))); # MAE
  error = np.sum(np.square(y - np.dot(X,theta))); #MSE
  #error =  error/len(y) + lam * np.sum(np.abs(theta)) #L1
  error =  error/len(y) + lam * np.sum(np.abs(theta*theta)) #L2
  return error
# NEGATIVE Derivative of log-likelihood
def fprime_mse(theta, X, y, lam):
  theta = theta.reshape((len(theta),1))
  #dl = np.dot(np.sign(np.dot(X, theta) - y).T, X).T #MAE
  dl = 2 * np.dot((np.dot(X, theta) - y).T, X).T #MSE
  dl = dl/len(y)
  #dl += lam * np.sign(theta); #L1
  dl += 2* lam * (theta); #L2
  return dl
  
# In[]
data = np.array(list(readGz('train.json.gz')))
shuffle(data)
data_size = 150000
data_train = data[:data_size]
data_valid = data[data_size:]
userHelpfulTest = defaultdict(list)
# In[]
rating_div = 1
price_div = 1
outOf_div = 1
words_div = np.max([len(d['reviewText'].split()) for d in data_train])
sent_div = 1

# In[]
'''
rating_div = 1.0
price_div = 1000
outOf_div = 500
words_div = 1500
sent_div = 880
'''
# In[]
'''
ratingsPerItem_Train = defaultdict(list)
ratingsPerItem_Val = defaultdict(list)

ratingsPerUser_Train = defaultdict(list)
ratingsPerUser_Val = defaultdict(list)

reviewsPerUser = defaultdict(int)

for d in data_train:
    ratingsPerItem_Train[d['itemID']].append(d['rating']);
    ratingsPerUser_Train[d['reviewerID']].append(d['rating']);
for d in data_valid:
    ratingsPerItem_Val[d['itemID']].append(d['rating']);
    ratingsPerUser_Val[d['reviewerID']].append(d['rating']);
'''
# In[]
avgHelpFullUserTemp = defaultdict(list)
for d in data_train:
    if(d['helpful']['outOf']>10):
        avgHelpFullUserTemp[d['reviewerID']].append(d['helpful'])
avgHelpFullUser = defaultdict(float)
avgHelpFullUserGlobal=0
for d in avgHelpFullUserTemp:
    sum_help = 0
    sum_outof = 0;
    for x in avgHelpFullUserTemp[d]:
        sum_help += x['nHelpful']
        sum_outof += x['outOf']
    avgHelpFullUser[d] = sum_help/sum_outof
    avgHelpFullUserGlobal += avgHelpFullUser[d]
avgHelpFullUserGlobal = avgHelpFullUserGlobal / len(avgHelpFullUser)
del avgHelpFullUserTemp
# In[]
# avg rating per item
avgRatingPerItemTemp = defaultdict(list)
global_avg_rating = 0;
for d in data_train:
    avgRatingPerItemTemp[d['itemID']].append(d['rating'])
    global_avg_rating += d['rating']
global_avg_rating = global_avg_rating / len(data_train)
avgRatingPerItem = defaultdict(float)
for d in avgRatingPerItemTemp:
    sum_ = 0;
    for x in avgRatingPerItemTemp[d]:
            sum_ += x;
    sum_ = sum_ / len(avgRatingPerItemTemp[d])
    avgRatingPerItem[d] = sum_    
del avgRatingPerItemTemp
# In[]

# In[]
# avg help vs year
'''
avgHelpYearTemp = defaultdict(list)
for d in data_train:
    if(d['helpful']['outOf']>5):
        year = int(d['reviewTime'].split(" ")[2])
        avgHelpYearTemp[year].append(d['helpful'])
    
avgHelpYear = defaultdict(float)
for d in avgHelpYearTemp:
    sum_help = 0
    sum_outof = 0;
    for x in avgHelpYearTemp[d]:
        sum_help += x['nHelpful']
        sum_outof += x['outOf']
    avgHelpYear[d] = sum_help/sum_outof
'''
# In[]

wordCount = defaultdict(int)
punctuation = set(string.punctuation)
stopword = stopwords.words('english')
stemmer = PorterStemmer()

for d in data_train:
  if(d['helpful']['outOf']>10):
      r = ''.join([c for c in d['reviewText'].lower() if not c in punctuation])
      for w in r.split():
        if w not in stopword:
            w = stemmer.stem(w)
            wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]

### Sentiment analysis

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)
# In[]

corpus = [d['reviewText'] for d in data_train if d['helpful']['outOf']>10]
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(corpus)
counts = X.toarray()
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(counts)
tfidf = tfidf.toarray()
idf = transformer.idf_

# In[]
# Verify
temp = []
for c in counts:
    temp.append(c * idf);
temp = np.array(temp)
temp = normalize(temp)
# In[]
def getXy(D, test=False, valid=False):
    X = []
    y = []
    outOf = []
    nHelp = []
    
    for l in D:
        if l['helpful']['outOf']!=0 and (test==True or (valid==True and l['helpful']['outOf'] > 0) or (test==False and l['helpful']['outOf'] > 10)):
            e = []
            #e.append(1) # bias
            '''
            e.append(1 if l['rating'] < 1.1 else 0) # rating
            e.append(1 if l['rating'] > 4.1 else 0) # rating
            rating_round = np.round(l['rating'])
            for i in range(1,5,1):
                if i==rating_round:
                    e.append(1)
                else:
                    e.append(0)
            
            e.append(l['rating']/rating_div) # rating
            e.append(l['rating']*l['rating']) # rating
            e.append((l['rating']-4.0522)) # rating
            e.append(np.abs(l['rating']-4.0522)) # rating
            #e.append(abs(l['rating']-4.0522)) # rating
            
            userId = l['reviewerID']
            if userId in reviewsPerUser:    
                e.append(reviewsPerUser[userId])
                #e.append(1)
            else:
                e.append(0)
                #e.append(0)
            
            if 'price' in l:
                e.append(l['price']/price_div) # rating
                e.append(1);
            else:
                e.append(0)
                e.append(0)
            #e.append(np.abs(l['rating']-4.0522)) # rating
            
            e.append(l['helpful']['outOf']/outOf_div) # out of
            e.append(1 if l['helpful']['outOf'] < 50 else 0)
            e.append(1 if l['helpful']['outOf'] >= 50 and l['helpful']['outOf'] < 200 else 0)
            e.append(1 if l['helpful']['outOf'] >= 200 and l['helpful']['outOf'] < 400 else 0)
            e.append(1 if l['helpful']['outOf'] >= 400 else 0)
            #e.append(1 if l['helpful']['outOf'] <= 100 and l['helpful']['outOf'] > 50 else 0)
            #e.append(1 if l['helpful']['outOf'] <= 150 and l['helpful']['outOf'] > 100 else 0)
            #e.append(1 if l['helpful']['outOf'] <= 200 and l['helpful']['outOf'] > 100 else 0)
            #e.append(1 if l['helpful']['outOf'] <= 300 and l['helpful']['outOf'] > 200 else 0)
            #e.append(1 if l['helpful']['outOf'] <= 400 and l['helpful']['outOf'] > 200 else 0)
            #e.append(1 if l['helpful']['outOf'] <= 500 and l['helpful']['outOf'] > 400 else 0)
            #e.append(1 if l['helpful']['outOf'] > 400 else 0)
            
            #e.append(((l['helpful']['outOf'])/outOf_div)*(l['helpful']['outOf']/outOf_div)) # out of
            #e.append(len(l['reviewText'].split())/words_div) # review length
            review_words = l['reviewText'].split()
            e.append(len(review_words)/words_div)
            e.append(len([word for word in review_words if word[0].isupper() ]))
            #e.append(len(l['reviewText'].split('.'))/sent_div)
            #e.append(count_pun(l['reviewText'], set(string.punctuation)))
            #e.append(allcap_differential(l['reviewText'].split())) #is any word all caps
            
            year = int(l['reviewTime'].split(" ")[2])
            #e.append(2017-year)
            #e.append((1488062202-l['unixReviewTime'])/2592000) # age in months
            #e.append(avgHelpYear[year])
            for i in range(11):
                if(2004+i==year):
                    e.append(1)
                else:
                    e.append(0)
                    #e.append((2014-(int(l['reviewTime'].split(" ")[2])))) # year
            #e.append(1-(l['unixReviewTime']))
            tf = vectorizer.transform([l['reviewText']]).toarray()
            tfidf = tf * idf
            tfidf = normalize(tfidf)
            for i in tfidf[0]:
                 e.append(i)   
            #np.concatenate(e, tfidf[0])
            
            feat = [0]*len(words)
            r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation])
            for w in r.split():
                if w in words:
                    feat[wordId[w]] += 1
            for f in feat:
                e.append(f)
            
            
            
            if l['reviewerID'] in avgHelpFullUser:
                e.append(avgHelpFullUser[l['reviewerID']])
                e.append(1)
            else:
                e.append(0)
                e.append(0)
            
            ''' 
            e.append(l['rating']) # rating
            e.append(np.square(l['rating'])) # rating
            e.append(1 if l['rating'] < 2.1 else 0) # rating
            e.append(1 if l['rating'] > 3.9 else 0) # rating
            e.append(l['helpful']['outOf']) # out of
            e.append(np.square(l['helpful']['outOf'])) # out of
            review_words = l['reviewText'].split()
            e.append(allcap_differential(l['reviewText'].split())) #is any word all caps
            e.append(sum([c=='!' for c in review_words]))
            e.append(len(review_words)/words_div)
            e.append((l['rating']-4.0522)) # rating
            e.append(np.abs(l['rating']-4.0522)) # rating
            
            year = int(l['reviewTime'].split(" ")[2])
            '''
            if 'price' in l:
                e.append(l['price']) # rating
            else:
                e.append(0)
            '''
            for i in range(11):
                if(2003+i==year):
                    e.append(1)
                else:
                    e.append(0)
                    #e.append((2014-(int(l['reviewTime'].split(" ")[2])))) # year
            if l['reviewerID'] in avgHelpFullUser:
                e.append(avgHelpFullUser[l['reviewerID']])
            else:
                e.append(avgHelpFullUserGlobal)
            if l['itemID'] in avgRatingPerItem:
                e.append(avgRatingPerItem[l['itemID']])
            else:
                e.append(global_avg_rating)
            e.append(len(l['reviewText'].split('.'))/sent_div)
            e.append(count_pun(l['reviewText'], set(string.punctuation)))
            e.append(allcap_differential(l['reviewText'].split())) #is any word all caps
            
            e.append((1488062202-l['unixReviewTime'])/2592000) # age in months
            
            '''
            feat = [0]*len(words)
            r = ''
            r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation])
            
            for w in r.split():
                w1 = stemmer.stem(w)
                if w1 in words:
                    feat[wordId[w1]] += 1
            #for f in feat:
                #e.append(f)
            #e  = np.concatenate(e, feat)
            e = e + feat
            '''
            '''
            tf = vectorizer.transform([l['reviewText']]).toarray()
            tfidf = tf * idf
            #tfidf = normalize(tfidf)
            for i in tfidf[0]:
                 e.append(i)   
            '''
            X.append(e)
            if test==False:
                y.append([l['helpful']['nHelpful']/l['helpful']['outOf']]); # nHelfFull/outOf
                outOf.append([l['helpful']['outOf']])
                nHelp.append([l['helpful']['nHelpful']])
            else:
                userHelpfulTest[l['reviewerID']+"_"+l['itemID']].append(e);
    X = np.array(X)
    y = np.array(y)
    outOf = np.array(outOf)
    nHelp = np.array(nHelp)
    return X, y, outOf, nHelp
# In[]
X_train, y_train, outOfTrain, nHelpTrain = getXy(data_train)
X_valid, y_valid, outOfValid, nHelpValid = getXy(data_valid, False, True)
# In[]
#clf = Ridge(3700.1)
#clf = Ridge(alpha=50.5005, normalize=True, solver='lsqr', fit_intercept = True)
clf = Ridge(alpha=450.1005, fit_intercept = True)
theta = [0] * len(X_train[0])
def train():    # In[]                      
    #lam = 0.016
    #lam=0.125
    '''
    lam = 0.002
    
    theta = [0] * len(X_train[0])
    theta = np.array(theta)
    
    theta = theta.reshape((len(theta),1))
    theta = 1.1*(np.random.random((len(theta),1)) - 0.5);
    '''
    #theta,l,info = scipy.optimize.fmin_l_bfgs_b(f_mse, theta, fprime_mse, args = (X_train, y_train, lam))
    #In[]
    clf.fit(X_train,y_train)
    predict = clf.predict(X_valid)
    #predict = np.dot(X_valid, theta);
    predict = predict.reshape((predict.shape[0],1))
    mae = np.sum(np.abs(nHelpValid - round(predict * outOfValid)) )/len(data_valid);
    print("MAE=",mae);
    theta = (clf.coef_)
    return theta, mae;
min_mae=100
theta_min = 0
for i in range(1):
    theta, mae =  train()
    if min_mae > mae:
        mae = min_mae
        theta_min = theta
# In[]
data_train_ = data

avgHelpFullUserTemp = defaultdict(list)
for d in data_train:
    if(d['helpful']['outOf']>1):
        avgHelpFullUserTemp[d['reviewerID']].append(d['helpful'])
avgHelpFullUser = defaultdict(float)
for d in avgHelpFullUserTemp:
    sum_help = 0
    sum_outof = 0;
    for x in avgHelpFullUserTemp[d]:
        sum_help += x['nHelpful']
        sum_outof += x['outOf']
    avgHelpFullUser[d] = sum_help/sum_outof

# In[]
'''
data_train_ = data
X_train_, y_train_, outOfTrain_, nHelpTrain_ = getXy(data_train_)
X_valid_, y_valid_, outOfValid_, nHelpValid_ = getXy(data_valid, False, True)

theta_, mae_ = train()
'''
# In[]

avg_helpful = np.sum(nHelpTrain) / np.sum(outOfTrain);
userHelpfulTest = defaultdict(list)
data_test = np.array(list(readGz('test_Helpful.json.gz')))
X_test, y_test, outOfTest, _t = getXy(data_test, True)
'''
ratingsPerItem_Test = defaultdict(list)
ratingsPerUser_Test = defaultdict(list)

for d in data_test:
    ratingsPerItem_Test[d['itemID']].append(d['rating']);
    ratingsPerUser_Val[d['reviewerID']].append(d['rating']);

'''
# In[]

predictions = open("predictions_Helpful.txt", 'w')
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,outOf = l.strip().split('-')
  outOf = int(outOf)
  key = u+"_"+i
  if key in userHelpfulTest:
      ll =np.array(userHelpfulTest[key][0])
      #print(ll)
      x = clf.predict(ll.reshape(1,-1))
      x = x[0][0]
      #predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(round(np.dot(ll, theta)*outOf)) + '\n')
      predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(round(x*outOf)) + '\n')
  else: 
      predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf*avg_helpful) + '\n')
  
predictions.close()