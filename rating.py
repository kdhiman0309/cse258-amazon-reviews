# In[]
import gzip
import numpy as np
from collections import defaultdict
import scipy.optimize
from sklearn.utils import shuffle
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

# In[]
data = np.array(list(readGz('train.json.gz')))
N_train = 180000
data_train = data[:N_train]
data_valid = data[N_train:]
# In[]
userDict = defaultdict(list)
itemDict = defaultdict(list)
#list  = []
count=0

for l in data_train:
    userID,itemID = l['reviewerID'],l['itemID']
    userDict[userID].append({'rating':l['rating'], 'itemID':itemID})
    itemDict[itemID].append({'rating':l['rating'], 'userID':userID});
# In[]
#Q5
alpha_ = 0;
N = 0;
for itemID, users in itemDict.items():
    for u in users:
        alpha_ += u['rating']
        N += 1
alpha_ = alpha_ / N;
print("alpha=",alpha_)
# In[]
ratings_valid = np.array([d['rating'] for d in data_valid])
mse_valid = np.sum(np.square(ratings_valid - alpha_)) / len(ratings_valid)
print("MSE Valid=",mse_valid)
# In[]
def getAlpha(beta_user, beta_item):
    sum_ = 0;
    for itemID, users in itemDict.items():
        for u in users:
            sum_ += u['rating'] - (beta_user[u['userID']] + beta_item[itemID])
    sum_ = sum_ / N_train
    return sum_;
def getBetaUser(userID, alpha, beta_item, lamda):
    sum_ = 0;
    items = userDict[userID]
    #print(userID)
    #print(items)
    for item in items:
        sum_ += item['rating'] - (alpha + beta_item[item['itemID']])
    sum_ = sum_ / (lamda + len(items))
    return sum_
def getBetaItem(itemID, alpha, beta_user, lamda):
    sum_ = 0;
    users = itemDict[itemID]
    for user in users:
        sum_ += user['rating'] - (alpha + beta_user[user['userID']])
    sum_ = sum_ / (lamda + len(users))
    return sum_
    
# In[]
#Q6 
beta_user = defaultdict(float)
beta_item = defaultdict(float)
for itemID, users in itemDict.items():
    beta_item[itemID] = 1.0;
for userID, items in userDict.items():
    beta_user[userID] = 1.0
# In[]
iters = 150
lamda = 1.0;
alpha = 0
    
for i in range(iters):
    alpha = getAlpha(beta_user, beta_item)
    for uid, beta in beta_user.items():
        beta_user[uid] = getBetaUser(uid, alpha, beta_item, lamda)
    for iid, beta in beta_item.items():
        beta_item[iid] = getBetaItem(iid, alpha, beta_user, lamda)
#In[]
avg_beta_user = 0
avg_beta_item = 0;
for itemID, users in itemDict.items():
    avg_beta_item += beta_item[itemID]

avg_beta_item = avg_beta_item / len(beta_item)

for userID, items in userDict.items():
    avg_beta_user += beta_user[userID]
avg_beta_user = avg_beta_user / len(beta_user)
#In[]
def predict(userID, itemID):
    
    return alpha + beta_user.get(userID, avg_beta_user) + beta_item.get(itemID, avg_beta_item)
#In[]
sum_ = 0;
for l in data_valid:
    sum_ += np.square(l['rating'] - predict(l['reviewerID'], l['itemID']))
mse_valid = sum_/len(data_valid)
print("MSE Valid=",mse_valid)
# In[]

print("BetaUser Max:",max(beta_user, key=beta_user.get))
print("BetaUser Min:",min(beta_user, key=beta_user.get))

print("BetaItem Max:",max(beta_item, key=beta_item.get))
print("BetaItem Min:",min(beta_item, key=beta_item.get))

# In[]
iters = 150
lamda = 7.0;
alpha = 0
    
for i in range(iters):
    alpha = getAlpha(beta_user, beta_item)
    for uid, beta in beta_user.items():
        beta_user[uid] = getBetaUser(uid, alpha, beta_item, lamda)
    for iid, beta in beta_item.items():
        beta_item[iid] = getBetaItem(iid, alpha, beta_user, lamda)
#In[]
avg_beta_user = 0
avg_beta_item = 0;
for itemID, users in itemDict.items():
    avg_beta_item += beta_item[itemID]

avg_beta_item = avg_beta_item / len(beta_item)

for userID, items in userDict.items():
    avg_beta_user += beta_user[userID]
avg_beta_user = avg_beta_user / len(beta_user)
#In[]
def predict(userID, itemID):
    
    return alpha + beta_user.get(userID, avg_beta_user) + beta_item.get(itemID, avg_beta_item)
#In[]
sum_ = 0;
for l in data_valid:
    sum_ += np.square(l['rating'] - predict(l['reviewerID'], l['itemID']))
mse_valid = sum_/len(data_valid)
print("MSE Valid=",mse_valid)

# In[]
predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  predictions.write(u + '-' + i +","+ str(predict(u,i))+'\n')
  
predictions.close()