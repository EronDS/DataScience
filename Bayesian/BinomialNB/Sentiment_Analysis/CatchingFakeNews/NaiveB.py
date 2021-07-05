from nltk import tokenize
import nltk
import re
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder




df = pd.read_csv('FakeOrRealNews.csv')


news = df.text 
label = df.label


nltk.download('stopwords')
nltk.download('rslp')

def Preprocessing(instancia):
    instancia = re.sub(r'https\S+', "" , instancia).lower().replace('.', '').replace('-', '').replace(';' , '').replace(')' , '') 
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [i for i in instancia.split() if not i in stopwords]
    return (' '.join(words))




vectorizer = CountVectorizer(analyzer = 'word')
model = MultinomialNB()

le = LabelEncoder()
y = le.fit_transform(label)
le.inverse_transform([0 , 1]) # 0 = Fake , 1 = True  

text = [Preprocessing(i) for i in news]

#np.save('news.npy' , np.array(text)) #saving into Numpy array 
#type(text)

news = vectorizer.fit_transform(text)

x_train,x_test,y_train,y_test = train_test_split(news,y,test_size = .2)





model.fit(x_train,y_train)
ypred = model.predict(x_test)
yhat = y_test
clss = ['Fake' , 'Real']

print(metrics.classification_report(ypred,yhat))
print(pd.crosstab(ypred,yhat,rownames = ['Real'] , colnames = ['Predict']))




