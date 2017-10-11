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
from sklearn.ensemble import RandomForestClassifier
# In[]
# Functions
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

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
punctuation = set(string.punctuation)
stopwordList = stopwords.words('english')

def reviewCounts(s):    
    words = s.split()
    nWords = len(words)
    nSentences = len(s.split("."))
    nChars = len(s)
    nPunctuations  = len([w for w in words if w in punctuation])
    nExclamations = len([w for w in words if w=='!'])
    nAllCaps = allcap_differential(words)
    nTitleWords = len([w for w in words if w[0].isupper()])
    return {"nWords":nWords, "nSentences":nSentences, 
            "nChars":nChars, "nPunc7tuations":nPunctuations,
            "nExclamations":nExclamations,"nAllCaps":nAllCaps,
            "nTitleWords":nTitleWords}

min_year = 2003
max_year = 2014         
def oneHotYear(e, year):
    for y in range(min_year, max_year, 1):
        if(y==year):
            e.append(1)
        else:
            e.append(0)
    return e
def getYear(year_str):
    return int(year_str.split(" ")[2])

def reviewAgeInMonths(timeStamp):
    return (1420070400-timeStamp)/2592000
               
# In[]
# Parameters
minOutOf=30
maxOutOf=600
num_words = 1000
total_train_size = 100000
# In[]
# Read data
data = np.array(list(readGz('train.json.gz')))
shuffle(data)
data_train = data[:total_train_size]
data_valid = data[total_train_size:]
del data
# In[]
# Pre-Process
avg_rating=0
avg_helpful=0
train_size = 0;
total_nHelpful=0;
total_outOf=0;

itemRatings = defaultdict(list)
itemHelpful = defaultdict(list)
itemHelpful_nHelpful = defaultdict(int)
itemHelpful_outOf = defaultdict(int)
userRatings = defaultdict(list)
userHelpful = defaultdict(list)

avgUserRatings = defaultdict(float)
avgUserHelpful = defaultdict(float)

avgItemRatings = defaultdict(float)
avgItemHelpful = defaultdict(float)

max_review_words = 0;

for d in data_train:
    outOf =  d['helpful']['outOf']
    if(outOf>=minOutOf):
        nHelpful = d['helpful']['nHelpful']
        rating = d['rating']
        userID = d['reviewerID']
        itemID = d['itemID']

        train_size += 1
        avg_rating += rating
        total_nHelpful += nHelpful
        total_outOf += outOf
        itemRatings[itemID].append(rating)
        itemHelpful[itemID].append(d['helpful'])
        userRatings[userID].append(rating)
        userHelpful[userID].append(d['helpful'])
        max_review_words = max(max_review_words, reviewCounts(d['reviewText'])['nWords'])
avg_rating = avg_rating / train_size
avg_helpful = total_nHelpful / total_outOf
del total_nHelpful
del total_outOf
del train_size
#In[]
for userID, ratings in userRatings.items():
    avgUserRatings[userID] = sum(ratings)/len(ratings)
    
for itemID, ratings in itemRatings.items():
    avgItemRatings[itemID] = sum(ratings)/len(ratings)
    
for userID, helpful in userHelpful.items():
    avgUserHelpful[userID] = sum([d['nHelpful'] for d in helpful]) \
                            / sum([d['outOf'] for d in helpful])
                            
for itemID, helpful in itemHelpful.items():
    avgItemHelpful[itemID] = sum([d['nHelpful'] for d in helpful]) \
                            / sum([d['outOf'] for d in helpful])
    
del outOf, d, userID, itemID, helpful, ratings
# In[]
# Unigram
wordCount = defaultdict(int)

stemmer = PorterStemmer()
for d in data_train:
    if d['helpful']['outOf']>=minOutOf:
        r = ''.join([c for c in d['reviewText'].lower() if not c in punctuation])
        for w in r.split():
            if w not in punctuation and w not in stopwordList:
                w_stem = stemmer.stem(w)
                wordCount[w_stem] +=1
counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:num_words]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)
del words, counts, wordCount
# In[]
corpus = [d['reviewText'] for d in data_train if d['helpful']['outOf']>=minOutOf]
vectorizer = CountVectorizer(max_features=num_words, stop_words='english')
X = vectorizer.fit_transform(corpus)
counts = X.toarray()
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(counts)
tfidf = tfidf.toarray()
idf = transformer.idf_
# In[]
def addUnigramTfIdf(e, s):
    tf = vectorizer.transform([s]).toarray()
    tfidf = tf * idf
    tfidf = normalize(tfidf)
    for i in tfidf[0]:
         e.append(i)   
    return e

# In[]
def getXy(D, train=False, valid=False, test=False):
    X = []
    y = []
    nHelpList = []
    outOfList = []
    userItemMap = defaultdict(list)
    for d in D:
        if(d['helpful']['outOf']>0):
            if(train):
                if(d['helpful']['outOf'] >= minOutOf and d['helpful']['outOf'] < maxOutOf):
                    X.append(buildModel(d))
                    y.append([d['helpful']['nHelpful']/d['helpful']['outOf']])
            elif(valid):
                if(d['helpful']['outOf'] >= minOutOf and d['helpful']['outOf'] < maxOutOf):
                    X.append(buildModel(d))
                    nHelpList.append(d['helpful']['nHelpful'])
                    outOfList.append(d['helpful']['outOf'])
            elif(test):
                userItemMap[d['reviewerID']+"_"+d['itemID']].append(buildModel(d)) 
    X = np.array(X)
    y = np.array(y)   
    nHelpList = np.array(nHelpList).reshape(-1,1)
    outOfList = np.array(outOfList).reshape(-1,1)
    return X, y, nHelpList, outOfList, userItemMap

# In[]
def getXy2(D, train=False, valid=False, test=False):
    X = []
    y = []
    nHelpList = []
    outOfList = []
    userItemMap = defaultdict(list)
    for d in D:
        if(d['helpful']['outOf']>0):
            if(train):
                if(d['helpful']['outOf'] >= minOutOf):
                    X.append(buildModel(d))
                    y.append([d['helpful']['nHelpful']/d['helpful']['outOf']])
            elif(valid):
                X.append(buildModel(d))
                nHelpList.append(d['helpful']['nHelpful'])
                outOfList.append(d['helpful']['outOf'])
            elif(test):
                userItemMap[d['reviewerID']+"_"+d['itemID']].append(buildModel(d)) 
    X = np.array(X)
    y = np.array(y)   
    nHelpList = np.array(nHelpList).reshape(-1,1)
    outOfList = np.array(outOfList).reshape(-1,1)
    return X, y, nHelpList, outOfList, userItemMap
# In[]
def addUnigram(e, s):
    r = ''.join([c for c in s.lower() if not c in punctuation])
    feat = [0] * num_words
    for w in r.split():
        if w not in stopwordList:
            w_stem = stemmer.stem(w)
            if w_stem in wordId:    
                feat[wordId[w_stem]] +=1
    e = e + feat
    return e
# In[]
def oneHotOutOf(e, outOf):
    t = []
    if(outOf >= 30):
        t.append(1 if outOf <=40 and outOf > 20 else 0)
        t.append(1 if outOf <=60 and outOf > 40 else 0)
        t.append(1 if outOf <=150 and outOf > 60 else 0)
        t.append(1 if outOf > 150 else 0)
    else:
        t.append(1 if outOf <=10 else 0)
        t.append(1 if outOf <=20 and outOf > 10 else 0)
        t.append(1 if outOf <=40 and outOf > 20 else 0)
    return e + t


# In[]
def buildModel(d):
    e = []
    e.append(1)
    if(d['helpful']['outOf'] >= 30):
        e.append(d['rating'])
        e.append(d['rating']-avg_rating)
        e.append(np.abs(d['rating']-avg_rating))
        
        #e.append(d['helpful']['outOf']/maxOutOf)
        
        rtAttr = reviewCounts(d['reviewText'])
        e.append(rtAt4tr['nWords']/max_review_words)
        #e.append(rtAttr['nSentences'])
        e.append(rtAttr['nExclamations'])
        e.append(rtAttr['nAllCaps'])
        e = oneHotOutOf(e, d['helpful']['outOf'])
        e = oneHotYear(e, getYear(d['reviewTime']))
        #e = addUnigram(e, d['reviewText'])
        e = addUnigramTfIdf(e, d['reviewText'])
        userID, itemID = d['reviewerID'], d['itemID']
        '''
        if userID in avgUserRatings:
            e.append(avgUserRatings[userID])
        else:
            e.append(avg_rating)
        
        if userID in avgUserHelpful:
            e.append(avgUserHelpful[userID])
        else:
            e.append(avg_helpful)
        
        
        if itemID in avgItemHelpful:
            e.append(avgItemHelpful[itemID])
        else:
            e.append(avg_helpful)
        
        if itemID in avgItemRatings:
            e.append(avgItemRatings[userID])
        else:
            e.append(avg_rating)
        '''
    else:
        e.append(d['rating'])
        e.append(d['rating']-avg_rating)
        e.append(np.abs(d['rating']-avg_rating))
        
        #e.append(d['helpful']['outOf']/maxOutOf)
        
        rtAttr = reviewCounts(d['reviewText'])
        e.append(rtAttr['nWords']/max_review_words)
        #e.append(rtAttr['nSentences'])
        e.append(rtAttr['nExclamations'])
        e.append(rtAttr['nAllCaps'])
        e = oneHotOutOf(e, d['helpful']['outOf'])
        e = oneHotYear(e, getYear(d['reviewTime']))
        #e = addUnigram(e, d['reviewText'])
        #e = addUnigramTfIdf(e, d['reviewText'])
        '''
        userID, itemID = d['reviewerID'], d['itemID']
        if userID in avgUserRatings:
            e.append(avgUserRatings[userID])
        else:
            e.append(avg_rating)
        
        if userID in avgUserHelpful:
            e.append(avgUserHelpful[userID])
        else:
            e.append(avg_helpful)
        '''
        '''
        if itemID in avgItemHelpful:
            e.append(avgItemHelpful[itemID])
        else:
            e.append(avg_helpful)
        
        if itemID in avgItemRatings:
            e.append(avgItemRatings[userID])
        else:
            e.append(avg_rating)
        '''
    return e

# In[]
# generate X, y for Train and Valid
'''
minOutOf = 1
maxOutOf = 10
X_train, y_train, _t, _t, _t = getXy(data_train, train=True)
X_valid, _t, nHelpful_valid, outOf_valid, _t = getXy(data_valid, valid=True)
# In[]
clf = Ridge(alpha=10.15, fit_intercept = False, solver='lsqr')
clf.fit(X_train,y_train)
predict = clf.predict(X_valid)
theta = clf.coef_
mae = np.sum(np.abs(nHelpful_valid - np.round(predict * outOf_valid)))/len(predict)
print("MAE = ", mae)
'''
# In[]
# generate X, y for Train and Valid
minOutOf = 5
maxOutOf = 30
X_train_l, y_train_l, _t, _t, _t = getXy(data_train, train=True)
X_valid_l, _t, nHelpful_valid_l, outOf_valid_l, _t = getXy(data_valid, valid=True)
# In[]
clf_l = Ridge(alpha=0.15, fit_intercept = False, solver='lsqr')
clf_l.fit(X_train_l,y_train_l)
predict_l = clf_l.predict(X_valid_l)
theta_l = clf_l.coef_
mae = np.sum(np.abs(nHelpful_valid_l - np.round(predict_l * outOf_valid_l)))/len(predict_l)
print("MAE = ", mae)
# In[]
# generate X, y for Train and Valid
minOutOf = 30
maxOutOf = 600
X_train_h, y_train_h, _t, _t, _t = getXy(data_train, train=True)
X_valid_h, _t, nHelpful_valid_h, outOf_valid_h, _t = getXy(data_valid, valid=True)
# In[]
clf_h = Ridge(alpha=4.5, fit_intercept = False, solver='lsqr')
clf_h.fit(X_train_h,y_train_h)
predict_h = clf_h.predict(X_valid_h)
theta_h = clf_h.coef_
mae = np.sum(np.abs(nHelpful_valid_h - np.round(predict_h * outOf_valid_h)))/len(predict_h)
print("MAE = ", mae)

# In[]
minOutOf = 1
maxOutOf = 30
X_valid, _t, nHelpful_valid, outOf_valid, _t = getXy(data_valid, valid=True)
predict = clf_l.predict(X_valid)
mae = np.sum(np.abs(nHelpful_valid - np.round(predict * outOf_valid)))/len(predict)
print("MAE = ", mae)

# In[]
'''
lam = 0.5
    
theta = [0] * len(X_train[0])
theta = np.array(theta)

theta = theta.reshape((len(theta),1))
theta = 0.001*(np.random.random((len(theta),1)) - 0.5);

theta,l,info = scipy.optimize.fmin_l_bfgs_b(f_mae, theta, fprime_mae, args = (X_train, y_train, lam))
predict = np.dot(X_valid, theta);
predict = predict.reshape((predict.shape[0],1))
mae = np.sum(np.abs(nHelpful_valid - np.round(predict * outOf_valid)) )/len(data_valid);
print("MAE=",mae);
'''

# In[]
# TEST

data_test = np.array(list(readGz('test_Helpful.json.gz')))
# In[]
_t, _t, _t, _t, userItemMap = getXy(data_test, test=True)

predictions = open("predictions_Helpful.csv", 'w')
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,outOf = l.strip().split('-')
  outOf = int(outOf)
  key = u+"_"+i
  if key in userItemMap:
    #if(outOf <5):
    #    predict = clf.predict(userItemMap[key])
    #el
    if(outOf<30):
        predict = clf_l.predict(userItemMap[key])
    else:
        predict = clf_h.predict(userItemMap[key])
        
    predict = predict[0][0]
    #     predict = avg_helpful * outOf
    predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(round(predict*outOf)) + '\n')
  else:
      predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(round(0*outOf)) + '\n')
predictions.close()