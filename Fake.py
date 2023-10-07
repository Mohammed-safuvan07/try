#!/usr/bin/env python
# coding: utf-8

# # AI BASED SOLUTION FOR FLAGGING OF FALSE INFORMATION ON ONLINE PLATFORMS 

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("E:/file2/Desktop/new_newsdesk.csv")


# In[2]:


data = data.dropna(how = 'any', axis = 0)


# In[3]:


data.isnull().sum()


# In[4]:


data.label.value_counts()


# In[5]:


from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer


# In[6]:


from nltk.corpus import stopwords
import nltk


# In[7]:


stemming = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[8]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV,KFold


# In[9]:


X=data[['text']]
Y=data['label']


# In[10]:


X


# In[11]:


p=data['text']
print(p)


# In[12]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[13]:


print('x_train:',x_train.shape)
print('y_train:',y_train.shape)
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)


# In[14]:


X_train = x_train


# In[15]:


x_train.head()


# In[16]:


X_test = x_test


# In[17]:


y_train.head()


# In[18]:


X_train.head()


# In[19]:


X_test.head(10)


# In[20]:


y_test


# # Data Preprocessing

# In[21]:


def preprocess(pro):
    process = re.sub('[^a-zA-Z]'," ",pro)
    lowe = process.lower()
    tokens = lowe.split()
   
    stop = [lemmatizer.lemmatize(i) for i in tokens if i not in stopwords.words('English')]
    lemmas =pd.Series([ " ".join(stop),len(stop)])
    return lemmas


# In[22]:


px_train = X_train['text'].apply(preprocess)


# In[23]:


px_train.head()


# In[156]:


type(px_train)


# # Test data preprocessing

# In[25]:


px_test = X_test['text'].apply(preprocess)


# In[26]:


px_test.head()


# In[27]:


px_test.columns = ['clean_text','text_length']
px_test.head()


# In[28]:


px_train.columns = ['clean_text','text_length']
px_train.head()


# In[29]:


X_train = pd.concat([X_train,px_train],axis=1)
X_train.head()


# In[30]:


X_test = pd.concat([X_test,px_test],axis=1)


# In[31]:


X_test.head()


# In[32]:


from wordcloud import WordCloud


# In[33]:


y_train


# In[34]:


y_test


# In[35]:


real_n = X_train.loc[y_train=='REAL', :]
real_n.head()


# In[36]:


words = ' '.join(real_n['clean_text'])
clean_word = " ".join([word for word in words.split()])


# In[37]:


real_word = WordCloud(stopwords=stopwords.words("english"),
                     background_color='black',
                     width=1600,
                     height=800).generate(clean_word)


# In[38]:


plt.figure(1,figsize=(30,20))
plt.imshow(real_word)
plt.axis('off')
plt.show()


# In[39]:


fake_n = X_train.loc[y_train=='FAKE', :]
fake_n.head()


# In[40]:


words_f = ' '.join(fake_n['clean_text'])
clean_word_f = " ".join([word for word in words_f.split()])


# In[41]:


real_word_f = WordCloud(stopwords=stopwords.words("english"),
                     background_color='black',
                     width=1600,
                     height=800).generate(clean_word_f)


# In[42]:


plt.figure(1,figsize=(30,20))
plt.imshow(real_word_f)
plt.axis('off')
plt.show()


# # Tfidf Vectorizer

# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[44]:


tf_vector = TfidfVectorizer()


# In[45]:


X_train_t = tf_vector.fit_transform(X_train['clean_text'])


# In[46]:


(X_train_t)


# In[47]:


print('unique words:',len(tf_vector.vocabulary_))
print('Shape of input data:',X_train_t.shape)


# # Test data
# 

# In[48]:


X_test_tf = tf_vector.transform(X_test['clean_text'])


# In[49]:


X_test_tf


# # Label Encoding

# In[50]:


label = LabelEncoder()


# In[51]:


y_train = label.fit_transform(y_train)


# In[52]:


y_train


# In[53]:


Y_test = label.transform(y_test)


# In[54]:


Y_test


# # Logistic Regression Model

# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


models = LogisticRegression()


# In[57]:


max_iter = range(100, 500)
solver = ['lbfgs', 'newton-cg', 'liblinear']
warm_start = [True, False]
C = np.arange(0, 1, 0.01)
random_grid ={
    'max_iter' : max_iter,
    'warm_start' : warm_start,
    'solver' : solver,
    'C' : C,
}


# In[58]:


kf1 = KFold(n_splits=5, shuffle=True)


# In[59]:


#random_search = RandomizedSearchCV(models,parameter, n_iter=10, cv=kf1, n_jobs=-1)


# In[60]:


random_search = RandomizedSearchCV(estimator =models, param_distributions = random_grid,n_iter = 10, scoring = 'accuracy',
                                   n_jobs = -1,verbose = 1,random_state = 1,cv=kf1)


# In[61]:


random_search.fit(X_train_t,y_train)


# In[62]:


n_model = LogisticRegression(warm_start = random_search.best_params_['warm_start'],solver=random_search.best_params_['solver'],max_iter = random_search.best_params_['max_iter'],C=random_search.best_params_['C']) 


# In[63]:


n_model.fit(X_train_t,y_train)


# In[64]:


from sklearn.metrics import accuracy_score


# In[65]:


new_logi =n_model.predict(X_train_t)


# In[66]:


new_logi_train_accuracy = accuracy_score(new_logi,y_train)


# In[67]:


print('accuracy_score',new_logi_train_accuracy)


# In[68]:


l_train_score = random_search.predict(X_train_t)
l_train_accuracy = accuracy_score(l_train_score,y_train)


# In[69]:


print('train_accuracy:',l_train_accuracy)


# In[70]:


l_test_score = random_search.predict(X_test_tf)


# In[71]:


l_test_accuracy = accuracy_score(l_test_score,Y_test) 


# In[72]:


print('test_acccuracy:',l_test_accuracy)


# In[73]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[74]:


confusion_matrix = metrics.confusion_matrix(Y_test, l_test_score)
print(confusion_matrix)


# In[75]:


precision = precision_score(Y_test, l_test_score)
precision


# In[76]:


recall = recall_score(Y_test, l_test_score)
recall


# In[127]:


cmx_1=confusion_matrix(Y_test,l_test_score)
print("\nNo. of test samples : ",len(X_test))
print("\n Confustion Matrix : \n",cmx_1)
print("\nPerfomance measures are: \n",classification_report(Y_test, l_test_score))


# In[77]:


f1 = f1_score(Y_test, l_test_score)
print(f1)


# In[78]:


data1 = {'news':["ROME: Novak Djokovic knows it isnt model behavior when he loses his cool on the tennis court.Yet he just can't help himself.Exactly two weeks after he was defaulted from the US Open , and a day after he was warned by the chair umpire for breaking his racket, Djokovic received an obscenity warning midway through a 7-5, 6-3 win over Casper Ruud in the Italian Open semifinals on Sunday.As opposed to his previous two outbursts, this time there were fans in the stands who could clearly hear how Djokovic dealt with his frustration.What's more is that, with 1,000 spectators allowed into the Foro Italico for the first time this week, a large proportion of those in attendance were children.I don't want to do it, but when it comes, it happens, Djokovic said Saturday. That's how I, I guess, release sometimes my anger. And it's definitely not the best message out there, especially for the young tennis players looking at me. I don't encourage that - definitely.Djokovic's behavior once again overshadowed his performance, in a match where he had to save two set points when Ruud served for the first set at 5-4 - one of them with a delicate backhand drop-shot winner.The top-ranked Djokovic also served five aces in a single game to take a 6-5 lead in the first.Ruud, 21, the first Norwegian player to contest a Masters 1000 semifinal and a product of Rafael Nadal's academy, put up plenty of resistance and also produced the shot of the day: a leaping over-the-shoulder hook shot for a winner as he raced back to chase down a lob - earning a thumbs-up from Djokovic.The obscenity warning came in the third game of the second set, by which time he had a running dialogue with the chair umpire over a series of contested calls.Still, Djokovic improved to 30-1 this year. His only loss came when he was thrown out of the US Open for unintentionally hitting a line judge in the throat with a ball during his fourth-round match against Pablo Carreno Busta.In Djokovic's 10th Rome final - he has won four - he ll face either eighth-seeded Diego Schwartzman or 12th-seeded Denis Shapovalov.Schwartzman beat nine-time Rome champion Nadal late Friday.In the women's tournament, top-seeded Simona Halep reached her third Rome final by beating Garbine Muguruza 6-3, 4-6, 6-4 to improve her record in tennis' restart to 9-0.Muguruza struggled with her serve and double faulted on the final two points of the 2 hour, 16-minute match.Halep, who lost to Elina Svitolina in the 2017 and 2018 finals, will face either Karolina Pliskova or Marketa Vondrousova in Mondays championship match.The second-ranked Halep is 13-0 overall stretching back to February, when she won a title in Dubai. After the tour's five-month break due to the coronavirus pandemic, the Romanian returned by raising another trophy in Prague last month. She then skipped the US Open due to travel and health concerns."]}


# In[79]:


data=pd.DataFrame(data1)
data


# In[80]:


print(data)


# # new prediction

# In[81]:


news2 ={'news':['''Whether or not Christians should celebrate Halloween has been a controversial topic for decades. Some view dressing up, eating candy and enjoying the festivities harmless and innocent, while others view it as an offense to their faith. Americans spend nearly $6.9 billion yearly making it the second largest commercial holiday in the country. As commercialized as the celebration has become, many of its roots are completely paganist. Is this a cause for Christians to avoid the entire celebration?
This is a time of year filled with debate, but not necessarily politics. Many Christians are convinced that Halloween is a satanic holiday while the rest of the world has found their sweet spot complete with costumes and candy. Children and adults have the opportunity to dress in accordance with their imagination, confirming from its haunted history to modern festivities, this holiday is a big deal. With decorations, candy, parties, and costumes, the average American spends up to $75 in the spirit of celebration.
Halloween is the holiday that links the seasons of fall and winter. Reportedly, it originated with one of the ancient Celtic festivals; an event where people would wear various costumes and light bonfires in hopes of warding off roaming ghosts. However, by the late 1800s, Americans shifted the theory of Halloween into a holiday centered on community and fun events. The focus, for many, has transitioned from witchcraft and ghosts to neighborhood celebratory events. With the evolving of the focal point, should Christians change their stance to celebrate the holiday?
Despite having at least partial roots from a Christian tradition, the relationship between Halloween and Christians has long been complicated. On October 31, 1517, Martin Luther essentially started the Protestant Reformation in Wittenberg, Germany, when he nailed his 95 Theses to a door. Many of the early Christian groups that came to America rejected this holiday as pagan. The Protestant Reformation heavily influenced the Pilgrims, Puritans, Quakers, and Baptists causing the great majority to frown upon it. However, that did not prevent Halloween from finding its way to American shores.
In the eighth century, Pope Gregory III dedicated November 1 as a time to honor all saints and martyrs. The holiday became widely recognized as All Saintsâ€™ Day. The evening before was known as All Hallowsâ€™ Eve, which later became Halloween. The word â€œhallowâ€ originated from the Old English word for â€œholyâ€ and â€œeâ€™enâ€ is an abbreviation of â€œevening.â€ As such, Halloween represented the night before All Saints Day.
Over time, Halloween advanced into a secular, community-based holiday branded by child-friendly activities that include costumes, neighborhood trick-or-treating, and more recently, trunk-or-treating. Along with a variety of pumpkin-flavored foods, Parties for both children and adults have become a very common way to celebrate the holiday. Some Christians still choose to lock themselves indoors with the lights off, but others have found freedom in their faith and are at liberty to decide when and how to participate.
In multiple countries around the world, as the days grow shorter and the nights get colder, people continue to escort the winter season in with candy-coated gatherings and a wide range of costumes. Halloween is a celebration that allows people of all ages to participate. Nonetheless, the question remains, â€œShould Christians celebrate?â€
Due to the efforts of community leaders and parents, Halloween has lost most of its illogical and religious undertones and is now more about imagination than spooky interpretation. There is nothing sinful about a Christian dressing up and participating in fun, non-threatening, celebrations. As a result, many Christians find no harm in dressing in costumes, attending parties and festivals as well as allowing their children to participate in school and local activities.
By Cherese Jackson (Virginia)Sources:History: Halloween
Kidsville News: Around the World â€“ October 2015
Grace to You: Christians and Halloween
Photo Credits:
Top Image Courtesy of Billy Wilson â€“ Flickr License
Inline Image (1) Courtesy of Richard Vignola â€“ Flickr License
Inline Image (2) Courtesy of The Forum News â€“ Flickr License
Featured Image Courtesy of John Nakamura â€“ Flickr License christianity , halloween''']}


# In[82]:


data_news1 = pd.DataFrame(news2)


# In[83]:


data_news1.head()


# In[84]:


tnews1 = data_news1[['news']]


# In[85]:


tnews1


# In[86]:


def preprocess(pro):
    process = re.sub('[^a-zA-Z]'," ",pro)
    lowe = process.lower()
    tokens = lowe.split()
   
    stop = [lemmatizer.lemmatize(i) for i in tokens if i not in stopwords.words('English')]
    lemmas =pd.Series([ " ".join(stop)])
    return lemmas


# In[87]:


gnews1 = tnews1['news'].apply(preprocess)


# In[88]:


gnews1.columns=['news']
gnews1


# In[89]:


newtf1 = tf_vector.transform(gnews1['news']) 


# In[90]:



print(newtf1)


# In[91]:


import pickle


# In[92]:


file = 'logisticm1.sav'
pickle.dump(models,open(file,'wb'))


# In[93]:


model_2 = pickle.load(open('logisticm1.sav','rb'))


# In[94]:


file2 = 'tfidf1.sav'
pickle.dump(tf_vector,open(file2,'wb'))


# In[95]:


model_3 = pickle.load(open('tfidf1.sav','rb'))


# In[98]:


ansr1 =n_model.predict(newtf1)
ansr1


# In[100]:


confusion = metrics.confusion_matrix(Y_test, l_test_score)


# # SVM

# In[128]:


from sklearn.svm import SVC


# In[129]:


support = svm.SVC()


# In[130]:


#C =[.01, .1, 1, 5, 10, 100]

#gamma= [0, .01, .1, 1, 5, 10, 100],
#kernel= ["rbf",'linear','poly']
#random_state=[1]
C = range(0, 10)
gamma = ['scale', 'auto']
svm_param = {
    "C":C ,
    "gamma":gamma
   
}


# In[131]:


kf2 = KFold(n_splits=5, shuffle=True)


# In[132]:


random_search1 = RandomizedSearchCV(estimator =support, param_distributions = svm_param,n_iter = 5, scoring = 'accuracy',
                                   n_jobs = -1,verbose = 1,random_state = 1,cv=kf2)


# In[133]:


random_search1.fit(X_train_t,y_train)


# In[134]:


#best_params = random_search.best_estimator_.get_params()
#print(best_params)
print("Best hyperparameters: ", random_search1.best_params_)


# In[135]:


support_1 = svm.SVC(gamma=random_search1.best_params_['gamma'],C=random_search1.best_params_['C'])


# In[136]:


support_1.fit(X_train_t,y_train)


# In[137]:


train_score_svm2  = support_1.predict(X_train_t)


# In[138]:


train_accuracy_svm2 = accuracy_score(train_score_svm2,y_train)


# In[139]:


print('train_accuracy:',train_accuracy_svm2)


# In[140]:


support


# In[141]:


support.fit(X_train_t,y_train)


# In[142]:


from sklearn.metrics import accuracy_score


# In[143]:


train_score_1 = support.predict(X_train_t)
train_accuracy_1 = accuracy_score(train_score_1,y_train)


# In[144]:


print('train_accuracy:',train_accuracy_1)


# In[145]:


test_score_1 = support.predict(X_test_tf)


# In[146]:


test_accuracy_1 = accuracy_score(test_score_1,Y_test)


# In[147]:


print('test_acccuracy:',test_accuracy_1)


# In[148]:


news_1=X_train_t[1]


# In[152]:


prediction_1 = support.predict(news_1)
print(prediction_1)

if (prediction_1[0]==0):
    print('The news is real')
else:
    print('The news is fake')


# In[111]:


from sklearn.metrics import classification_report, confusion_matrix


# In[150]:


confusion = metrics.confusion_matrix(Y_test, test_score_1)


# In[151]:


cmx=confusion_matrix(Y_test,test_score_1)
print("\nNo. of test samples : ",len(X_test))
print("\n Confustion Matrix : \n",cmx)
print("\nPerfomance measures are: \n",classification_report(Y_test, test_score_1))


# # KNN

# In[102]:


from sklearn.neighbors import KNeighborsClassifier


# In[103]:


knn_model = KNeighborsClassifier(n_neighbors=5)


# In[104]:


knn_model.fit(X_train_t,y_train)


# In[105]:


knn_1_train_score = knn_model.predict(X_train_t)
knn_train_accuracy = accuracy_score(knn_1_train_score,y_train)


# In[106]:


print('train_accuracy:',knn_train_accuracy)


# In[107]:


knn_test_score = knn_model.predict(X_test_tf)


# In[108]:


knn_test_accuracy = accuracy_score(knn_test_score,Y_test)


# In[109]:


print('test_acccuracy:',knn_test_accuracy)


# In[112]:


cmx_2=confusion_matrix(Y_test,knn_test_score)
print("\nNo. of test samples : ",len(X_test))
print("\n Confustion Matrix : \n",cmx_2)
print("\nPerfomance measures are: \n",classification_report(Y_test, knn_test_score))


# # SBERT

# In[113]:


get_ipython().system(' pip install -U sentence-transformers')


# In[114]:


from sentence_transformers import SentenceTransformer, util


# In[115]:


model = SentenceTransformer('all-MiniLM-L6-v2')  


# In[116]:


X_train_b = model.encode(px_train['clean_text'].tolist())


# In[117]:


X_test_b = model.encode(px_test['clean_text'].tolist())


# # KNN-Model

# In[118]:


knn_model.fit(X_train_b,y_train)


# In[119]:


knn_bert_train_score = knn_model.predict(X_train_b)
knn_train_accuracy_bert = accuracy_score(knn_bert_train_score,y_train)


# In[120]:


knn_train_accuracy_bert


# # Logistic Regression - Model

# In[123]:


n_model.fit(X_train_b,y_train)


# In[124]:


logi_bert_train_score = n_model.predict(X_train_b)
logi_train_accuracy_bert = accuracy_score(logi_bert_train_score,y_train)


# In[125]:


logi_train_accuracy_bert


# In[ ]:





# In[153]:


support.fit(X_train_b,y_train)


# In[154]:


svm_bert_train_score = support.predict(X_train_b)
svm_train_accuracy_bert = accuracy_score(svm_bert_train_score,y_train)


# In[155]:


svm_train_accuracy_bert


# In[ ]:




