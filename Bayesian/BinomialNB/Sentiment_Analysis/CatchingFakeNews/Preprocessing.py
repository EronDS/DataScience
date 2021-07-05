import pandas as pd 
import os 

os.listdir()

fake = pd.read_csv('Fake.csv')
fake

label = [] 
text = [] 

for i in range(len(fake.text)):
    label.append('Fake')
    text.append(fake.text[i])


true = pd.read_csv('True.csv')

for i in range(len(true.text)):
    label.append('True')
    text.append(true.text[i])



print(len(text), len(label))


df = pd.DataFrame()
df['text'] = text 
df['label'] = label


df.to_csv('FakeOrRealNews.csv')