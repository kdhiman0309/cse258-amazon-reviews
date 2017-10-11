# In[]
import gzip
import numpy as np
from collections import defaultdict
import scipy.optimize
from sklearn.utils import shuffle
import copy
import random
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])
def getYear(year_str):
    return int(year_str.split(" ")[2])

# In[]
data = np.array(list(readGz('train.json.gz')))
#N_train = 1000
N_train = 100000
#shuffle(data)
data_train = data[:N_train]
data_valid = data[N_train:]
# In[]
userDict = defaultdict(list)
itemDict = defaultdict(list)
#list  = []

for l in data_train:
    userID,itemID = l['reviewerID'],l['itemID']
    userDict[userID].append({'rating':l['rating'], 'itemID':itemID})
    itemDict[itemID].append({'rating':l['rating'], 'userID':userID});
# In[]
'''
yearRatings = defaultdict(list)
avgRatingYear = defaultdict(float)
avgRating=0
for d in data_train:
    year = getYear(d['reviewTime'])
    yearRatings[year].append(d['rating'])
    avgRating+= d['rating']
avgRating = avgRating / len(data_train)
for y in yearRatings:
    s = 0
    for rating in yearRatings[y]:
        s+= rating
    s = s / len(yearRatings[y])
    avgRatingYear[y] = s - avgRating
# In[]
avgUserDict = defaultdict(float)
avgItemDict = defaultdict(float)

for userID, items in userDict:
    sum_ =0
    for item in items:
        sum_ += item['rating']
    sum_ = sum_ / len(items)
    avgUserDict[userID] = sum_

for itemID, users in itemDict:
    sum_ = 0
    for user in users:
        sum_ += user['rating']
    sum_ = sum_ / len(users)
    avgItemDict[itemID] = sum_     
'''
# In[]
beta_user = defaultdict(float)
beta_item = defaultdict(float)
K = 10
lamda = 7.0;
alpha = 0

for itemID, users in itemDict.items():
    beta_item[itemID] = 1.0;
for userID, items in userDict.items():
    beta_user[userID] = 1.0

# In[]
# Naive model
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
iters = 150
lamda = 7.0;
alpha = 0
    
for i in range(iters):
    alpha = getAlpha(beta_user, beta_item)
    for uid, beta in beta_user.items():
        beta_user[uid] = getBetaUser(uid, alpha, beta_item, lamda)
    for iid, beta in beta_item.items():
        beta_item[iid] = getBetaItem(iid, alpha, beta_user, lamda)

avg_beta_user = 0
avg_beta_item = 0;
for itemID, users in itemDict.items():
    avg_beta_item += beta_item[itemID]

avg_beta_item = avg_beta_item / len(beta_item)

for userID, items in userDict.items():
    avg_beta_user += beta_user[userID]
avg_beta_user = avg_beta_user / len(beta_user)
print("avg_beta_item=",avg_beta_item)
print("avg_beta_user",avg_beta_user)

#In[]
def predict(userID, itemID):  
    return alpha + beta_user.get(userID, avg_beta_user) + beta_item.get(itemID, avg_beta_item)
# In[]
sum_ = 0;
for l in data_valid:
    sum_ += np.square(l['rating'] - predict(l['reviewerID'], l['itemID']))
mse_valid = sum_/len(data_valid)
print("MSE Valid=",mse_valid)
# In[]
alpha_org = copy.deepcopy(alpha)
beta_user_org = copy.deepcopy(beta_user)
beta_item_org = copy.deepcopy(beta_item)
# In[]
'''
def getObjFunction():
    sum_ = 0
    for itemID, users in itemDict.items():
        for u in users:
            userID = u['userID']
            sum_ += np.square(alpha + beta_user[userID] + beta_item[itemID] - u['rating']) 

    sum_ += lamda * (np.sum(np.sqaure(beta_user))
                + np.sum(np.sqaure(beta_item))
                    + np.sum(np.sqaure(gamma_item))
                       + np.sum(np.sqaure(gamma_user)))
'''
def getAlphaDerv():
    sum_ = 0;
    N = 0
    for itemID, users in itemDict.items():
        for u in users:
            N += 1
            userID = u['userID']
            sum_ +=  (alpha+beta_user[userID] + beta_item[itemID] 
                        + np.dot(gamma_user[userID],gamma_item[itemID]) \
                    - u['rating'])
    sum_ = 2 * sum_ / N
    return  sum_;
def getBetaUserDerv(userID):
    sum_ = 0;
    items = userDict[userID]
    #print(userID)
    #print(items)
    for item in items:
        sum_ += ((alpha + beta_item[item['itemID']] + beta_user[userID] 
                    + np.dot(gamma_user[userID],gamma_item[itemID])) \
                 - item['rating'])
    sum_ = 2 * (sum_/len(items)) + 2 * lamda * beta_user[userID] 
    #sum_ = sum_ / (lamda + len(items)) 
    return sum_
def getBetaItemDerv(itemID):
    sum_ = 0;
    users = itemDict[itemID]
    for user in users:
        sum_ += (alpha + beta_user[user['userID']] + beta_item[itemID]
                    + np.dot(gamma_user[userID],gamma_item[itemID]) \
                  - user['rating'])
    sum_ = 2 * (sum_/len(users)) + 2 * lamda * beta_item[itemID] 
    return sum_

def getGammaUserDerv(userID, k):
    sum_ = 0;
    items = userDict[userID]
    for item in items:
        sum_ += gamma_item[item['itemID']][k] * ((alpha 
                    + beta_item[item['itemID']]
                    +  beta_user[userID]
                    + np.dot(gamma_user[userID],gamma_item[itemID])) \
                 - item['rating'])
    sum_ = 2 * (sum_/len(items)) + 2 * lamda * gamma_user[userID][k]
    return sum_

def getGammaItemDerv(itemID, k):
    sum_ = 0;
    users = itemDict[itemID]
    for user in users:
        sum_ += gamma_user[user['userID']][k] * ((alpha 
                    + beta_user[user['userID']] 
                    + beta_item[itemID]
                    + np.dot(gamma_user[userID],gamma_item[itemID])) \
                  - user['rating'])
    sum_ = 2 * (sum_/len(users)) + 2 * lamda * gamma_item[itemID][k] 
    return sum_
    
def getAvgBeta(beta):
    sum_ = 0
    for k,v in beta.items():
        sum_ += v
    sum_ = sum_ / len(beta)
    return sum_
def getAvgGamma(gamma):
    sum_ = [0]*K
    for u,v in gamma.items():
        for i in range(len(sum_)):
            sum_[i] += v[i]
    for i in range(len(sum_)):
        sum_[i] = sum_[i]/len(gamma)
    return sum_
    
# In[]
def evaulate(data_):
    mse = 0;
    for d in data_:
        mse += np.square(d['rating'] - predict(d['reviewerID'], d['itemID']))
    return np.sqrt(mse/len(data_))
def anealing(i, eta):
    return eta / (1+i/T)
    #return eta
def momentum(var, grad, accu_grad, eta_, mu_):
    new_del = accu_grad * mu_ - eta_ * grad
    var = var + new_del 
    accu_grad = new_del
    return var, accu_grad

    
def updateAlpha(eta):
    global alpha, delta_alpha
    alpha_ = alpha
    alpha += mu * delta_alpha;
    grad = getAlphaDerv()
    delta_alpha = mu * delta_alpha  - eta * grad
    alpha = alpha_ + delta_alpha

def updateBetaUser(userID, eta):
    global beta_user, delta_beta_user
    beta_ = beta_user[userID];
    beta_user[userID] += mu * delta_beta_user[userID]
    grad = getBetaUserDerv(userID)
    delta_beta_user[userID] = mu * delta_beta_user[userID] - eta * grad
    beta_user[userID] = beta_ + delta_beta_user[userID]

def updateBetaItem(itemID, eta):
    global beta_item, delta_beta_item
    beta_ = beta_item[itemID]
    beta_item[itemID] += mu * delta_beta_item[itemID]
    grad = getBetaItemDerv(itemID)
    delta_beta_item[itemID] = mu * delta_beta_item[itemID] - eta * grad
    beta_item[itemID] = beta_ + delta_beta_item[itemID]

def updateGammaUser(userID, k, eta):
    global gamma_user, delta_gamma_user
    gamma_ = gamma_user[userID][k]
    gamma_user[userID][k] += mu * delta_gamma_user[userID][k]
    grad = getGammaUserDerv(userID, k)
    delta_gamma_user[userID][k] = mu * delta_gamma_user[userID][k] - eta * grad
    gamma_user[userID][k] = gamma_ + delta_gamma_user[userID][k]

def updateGammaItem(itemID, k, eta):
    global gamma_item, delta_gamma_item
    gamma_ = gamma_item[itemID][k]
    gamma_item[itemID][k] += mu * delta_gamma_item[itemID][k]
    grad = getGammaItemDerv(itemID, k)
    delta_gamma_item[itemID][k] = mu * delta_gamma_item[itemID][k] - eta * grad
    gamma_item[itemID][k] = gamma_ + delta_gamma_item[itemID][k]

def predict(userID, itemID):
    global alpha, beta_user, beta_item, gamma_user, gamma_item, \
                avg_beta_item, avg_beta_user, avg_gamma_item, avg_gamma_user
    rating = alpha + beta_user.get(userID, avg_beta_user) \
                + beta_item.get(itemID, avg_beta_item) \
                    + np.asscalar(np.dot(gamma_item.get(itemID, avg_gamma_item),
                                         gamma_user.get(userID, avg_gamma_user)))
    '''
    rating =  alpha + beta_user.get(userID, 0) \
                + beta_item.get(itemID, 0) \
                    + np.asscalar(np.dot(gamma_item.get(itemID, [0]*K),
                                         gamma_user.get(userID, [0]*K)))
    '''
    rating = max(1, rating)
    rating = min(5, rating)
    return rating             
 
def gradientDecent(eta_a, eta_b, eta_g):
    global alpha, beta_user, beta_item, gamma_user, gamma_item, \
                avg_beta_item, avg_beta_user, avg_gamma_item, avg_gamma_user,\
                delta_gamma_user,delta_gamma_item,delta_beta_item,delta_beta_user,delta_alpha,\
                alpha_min, beta_user_min,beta_item_min,gamma_user_min, gamma_item_min
            
    min_rmse = 1000
    count = 0
    
    for i in range(iters):
        
        # alpha
        #new_del = delta_alpha * mu - anealing(i,eta_a) * getAlphaDerv()
        #alpha = alpha + new_del 
        #delta_alpha = new_del
        
        #alpha, delta_alpha = momentum(alpha, getAlphaDerv(), delta_alpha, anealing(i,eta_a), mu)
        updateAlpha(anealing(i,eta_a))
        for userID in beta_user:
            updateBetaUser(userID,anealing(i,eta_b))
            #beta_user[userID], delta_beta_user[userID] = momentum(beta_user[userID], 
            #                getBetaUserDerv(userID), delta_beta_user[userID], anealing(i,eta_b), mu)
            
            #new_del = delta_beta_user[userID] * mu - anealing(i,eta_b) * getBetaUserDerv(userID)
            #beta_user[userID] = beta_user[userID] + new_del
            #delta_beta_user[userID] = new_del
        for itemID in beta_item:
            updateBetaItem(itemID, anealing(i,eta_b))
            #beta_item[itemID], delta_beta_item[itemID] = momentum(beta_item[itemID], 
            #                getBetaItemDerv(itemID), delta_beta_item[itemID], anealing(i,eta_b), mu)
            #new_del = delta_beta_item[itemID] * mu - anealing(i,eta_b) * getBetaItemDerv(itemID)
            #beta_item[itemID] = beta_item[itemID] + new_del
            #delta_beta_item[itemID] = new_del
                
        for userID in gamma_user:
            for k in range(K):
                updateGammaUser(userID, k, anealing(i,eta_g))
                #gamma_user[userID][k], delta_gamma_user[userID][k] = momentum(gamma_user[userID][k], 
                #        getGammaUserDerv(userID, k), delta_gamma_user[userID][k], anealing(i,eta_g), mu)
                #new_del = delta_gamma_user[userID][k] * mu - anealing(i,eta_g) * getGammaUserDerv(userID, k)
                #gamma_user[userID][k] = gamma_user[userID][k] + new_del
                #delta_gamma_user[userID][k] = new_del
                    
        for itemID in gamma_item:
            for k in range(K):
                updateGammaItem(itemID, k, anealing(i,eta_g))
                #gamma_item[itemID][k], delta_gamma_item[itemID][k] = momentum(gamma_item[itemID][k], 
                #        getGammaItemDerv(itemID,k), delta_gamma_item[itemID][k], anealing(i,eta_g), mu)
                #new_del = delta_gamma_item[itemID][k] * mu - anealing(i,eta_g) * getGammaItemDerv(itemID,k)
                #gamma_item[itemID][k] = gamma_item[itemID][k] + new_del
                #delta_gamma_item[itemID][k] = new_del
        avg_beta_item = getAvgBeta(beta_item)
        avg_beta_user = getAvgBeta(beta_user)
        avg_gamma_item = getAvgGamma(gamma_item)
        avg_gamma_user = getAvgGamma(gamma_user)
        rmse = evaulate(data_valid)
        if(rmse < min_rmse):
            min_rmse = rmse
            count = 0
            alpha_min = copy.deepcopy(alpha)
            beta_user_min = copy.deepcopy(beta_user)
            beta_item_min = copy.deepcopy(beta_item)
            gamma_user_min = copy.deepcopy(gamma_user)
            gamma_item_min = copy.deepcopy(gamma_item)
        else:
            count += 1 
        if(count ==4):
            break
        print(i," RMSE=",rmse)
# In[]
'''
beta_user = defaultdict(float)
beta_item = defaultdict(float)
avg_beta_user = 0
avg_beta_item = 0;
alpha = avgRating
'''
# In[]
# In[]
'''
for u in itemDict:
    beta_item[u] = random.uniform(-0.05, 0.05)
    
for u in userDict:
    beta_user[u] = random.uniform(-0.05, 0.05)
'''
# In[]
alpha = copy.deepcopy(alpha_org)
beta_item = copy.deepcopy(beta_item_org)
beta_user = copy.deepcopy(beta_user_org)
K=5
gamma_user = defaultdict(list)
gamma_item = defaultdict(list)

delta_gamma_user = defaultdict(list)
delta_gamma_item = defaultdict(list)
delta_beta_item = defaultdict(list)
delta_beta_user = defaultdict(list)
delta_alpha = 0

for itemID, users in itemDict.items():
    delta_beta_item[itemID] = 0.0;
for userID, items in userDict.items():
    delta_beta_user[userID] = 0.0

for userID in userDict:
    gamma_user[userID] = [0]*K  #(np.random.random((K, 1)) - 0.5) * 0.001
    delta_gamma_user[userID] = [0]*K
for itemID in itemDict:
    gamma_item[itemID] = [0]*K #(np.random.random((K,1)) - 0.5) * 0.001
    delta_gamma_item[itemID] = [0]*K

avg_beta_item = getAvgBeta(beta_item)
avg_beta_user = getAvgBeta(beta_user)
avg_gamma_user = getAvgGamma(gamma_user)
avg_gamma_item = getAvgGamma(gamma_item)

alpha_min = copy.deepcopy(alpha)
beta_user_min = copy.deepcopy(beta_user)
beta_item_min = copy.deepcopy(beta_item)
gamma_user_min = copy.deepcopy(gamma_user)
gamma_item_min = copy.deepcopy(gamma_item)

print("avg_beta_item=",avg_beta_item)
print("avg_beta_user",avg_beta_user)

rmse = evaulate(data_valid)
print("RMSE=",rmse)    

for k,v in gamma_item.items():
    for i in range(len(v)):
        v[i] = random.uniform(-0.001, 0.001)

for k,v in gamma_user.items():
    for i in range(len(v)):
        v[i] = random.uniform(-0.001, 0.001)

print("Gradient Decent")
lamda = 6.0;
eta_a = 1.0e-5
eta_b = 1.0e-5
eta_g = 1.0e-1
T = 100
mu = 0.8
iters = 150
gradientDecent(eta_a, eta_b, eta_g)
# not present
# user items
# cal avg from those items
# ceil

alpha = copy.deepcopy(alpha_min)
beta_user = copy.deepcopy(beta_user_min)
beta_item = copy.deepcopy(beta_item_min)
gamma_user = copy.deepcopy(gamma_user_min)
gamma_item = copy.deepcopy(gamma_item_min)

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