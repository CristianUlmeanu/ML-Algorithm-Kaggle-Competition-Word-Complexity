import sys
import regex
import time

import openpyxl as openpyxl
import pyphen as pyp
import numpy as np
import pandas as pan

#Gaussian
from sklearn.naive_bayes import GaussianNB

#Fold Slit
from sklearn.model_selection import KFold


pan.set_option('display.width', None)
pan.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)


#Dale_Chall luat de la laborator
from dale_chall import DALE_CHALL as dale

from nltk.tokenize import word_tokenize
from nltk.tag import AffixTagger
from nltk.corpus import wordnet as wn

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

dtypes = {"sentence": "string", "token": "string", "complexity": "float64"}
train = pan.read_excel('train.xlsx',dtype=dtypes, keep_default_na=False)
test = pan.read_excel('test.xlsx', dtype=dtypes, keep_default_na=False)

#Check date
#print(train)
#print(test)


#Creare caracteristici cuvant

vowels = 'aeiouw'
dict = pyp.Pyphen(lang='en')

def is_dale_chall(token):
    if (int(token not in dale)):
        return 8
    else:
        return 2
def is_title(token):
    if (int(token.istitle())==1):
        return int(token.istitle())+3.5
    else:
        return 2
            #int(token.istitle())
def lungime(token):
    if(len(token)<5):
        return 7
    if(len(token)>4 and len(token)<8):
        return 0
    if(len(token)>7 and len(token)<11):
        return 2
    if(len(token)>10):
        return 3.5

def nr_vowels(token):
       nr = 0
       for chr in token:
           if chr in vowels:
               nr += 1
       return nr

def nr_syllables(token):
  return len(dict.inserted(token).split('-'))

def nr_synsets(token):
    #print("synsets",len(wn.synsets(token)))
    return len(wn.synsets(token))*1.5

def num_lemmas(token):
    return (sum([len(synset.lemmas()) for synset in wn.synsets(token)]))

def sufix_prefix_complex(token):
    unique_prefix_list=[]
    unique_sufix_list=[]
    nr_morfisme=0
    prefix_list=["che","neo","cere","morph","syn","hom","tetra","hybrid","re","dis","over","un","mis","out","be","co","de","fore","inter","pre","sub","trans","under","anti","auto","bi","counter","dis","ex","hyper","in","inter","kilo","mal","mega","mis","mini","mono","neo","out","poly","pseudo","re","semi","sub","super","sur","tele","tri","ultra","under","vice","non","dis","un","im"]
    sufix_list=["ise","ate","fy","en","tion","sion","er","ment","ant","ent","age","ai","ence","ense","ance","ery","ry","er","ism","ship","age","ity","ness","cy","al","ent","ive","ous","ful","less","able"]
    for prefix in prefix_list:
        if(prefix not in unique_prefix_list):
            unique_prefix_list.append(prefix)
    for sufix in sufix_list:
        if(sufix not in unique_sufix_list):
            unique_sufix_list.append(sufix)
    #print(len(unique_sufix_list))
    #print(len(unique_prefix_list))
    for idx in unique_prefix_list:
        if(token.find(idx,0,5)!=-1):
            nr_morfisme=nr_morfisme+1
            #print(idx)
            #print(nr_morfisme)
    for idx in unique_sufix_list:
        if(token.find(idx,len(token)-5,len(token))!=-1):
            nr_morfisme=nr_morfisme+1
            #print(idx)
            #print(nr_morfisme)
    if (nr_morfisme>2):
        return 4
    else:
        return 2


#Creare caracteristici corpus
def corpus_feature(corpus):
    d={"bible":[6], "europarl":[3],"biomed":[9]}
    return d[corpus]

#Creare caracteristici propozitie
def dale_chall_prop(sentence):
    counter=0
    prop=regex.split(r" |, |,\"|,|; |:'|\|\.",sentence)
    prop[len(prop)-1]=regex.split(r"\.|!|\?|;|'|\"",prop[len(prop)-1])[0]
    for index in prop:
        if(index.lower() not in dale):
            counter=counter+1
    if (counter/len(prop)>0.05):
        formula_dale = 0.1579 *(counter/len(prop)*100)+0.0496*len(prop)+3.6365-0.75
    else:
        formula_dale = 0.1579 *(counter/len(prop)*100)+0.0496*len(prop)-0.75
    if(formula_dale<6):
        return [2]
    elif(formula_dale>5.99 and formula_dale<11):
        return [5]
    elif(formula_dale>10.99):
        return [8]

def gunning_readablity(sentence):
    counter=0
    prop = regex.split(r" |, |,\"|,|; |:'|\|\.", sentence)
    prop[len(prop) - 1] = regex.split(r"\.|!|\?|;|'|\"", prop[len(prop) - 1])[0]
    for index in prop:
        if(len(dict.inserted(index).split('-'))>=3):
            counter=counter+1
    formula_gunning=0.4*(len(prop)+100*(counter/len(prop)))
    #print("formula gunning:",formula_gunning)
    #print("nr de cuvinte sunt:",len(prop))
    #print("silabele sunt",counter)
    if(formula_gunning>15):
        return [4]
    elif(formula_gunning<=15 and formula_gunning>11):
        return [3]
    elif(formula_gunning<=11):
        return [2]


def flesh_readability(sentence):
    silabe=0
    prop = regex.split(r" |, |,\"|,|; |:'|\|\.", sentence)
    prop[len(prop) - 1] = regex.split(r"\.|!|\?|;|'|\"", prop[len(prop) - 1])[0]
    for index in prop:
        silabe=silabe+len(dict.inserted(index).split('-'))
    formula_fresh = 206.835-1.015*(len(prop))-84.6*(silabe/len(prop))
    #print("formula fresh:",formula_fresh)
    #print("nr de cuvinte sunt:",len(prop))
    #print("silabele sunt",silabe)
    if(formula_fresh<=30):
        return [4]
    if(formula_fresh>30 and formula_fresh<=65):
        return [3]
    if(formula_fresh>65):
        return [2]

def diversitate_lexicala(sentence):
    cuvinte=0
    prop = regex.split(r" |, |,\"|,|; |:'|\|\.", sentence)
    prop[len(prop) - 1] = regex.split(r"\.|!|\?|;|'|\"", prop[len(prop) - 1])[0]
    for index in prop:
        counter=0
        for index2 in prop:
            if (index==index2):
                counter=counter+1
        if (counter==1):
            cuvinte=cuvinte+1
    formula_diversitate=cuvinte/len(prop)
    if(formula_diversitate>0.65):
        return 8
    elif(formula_diversitate<=0.65 and formula_diversitate>0.3):
        return 5
    else:
        return 2

#Unire caracteristici cuvant
def get_word_structure_features(token):
    features = []
    features.append(is_dale_chall(token))
    features.append(is_title(token))
    features.append(lungime(token))
        ##Pe Gausian creste rata de complexitate fara lungime
    features.append(nr_vowels(token))
        ##Pe Gausian creste rata de complexitate fara vocale
    features.append(nr_syllables(token))
        ##Pe Gausian creste rata de complexitate fara silabe
    features.append(nr_synsets(token))
    #features.append(sufix_prefix_complex(token))
        ##Nu este o implementare eficienta
    #features.append(num_lemmas(token))
        ##Nu este o implementare eficienta
    return np.array(features)

#Unire caracteristici prop
def get_prop_features(sentence):
    features = []
    features.extend(dale_chall_prop(sentence))
    feature_prop_lungime=(gunning_readablity(sentence)[0]+flesh_readability(sentence)[0])/2
    feature_prop_lungime=[feature_prop_lungime]
    features.extend(feature_prop_lungime)
    features.append(diversitate_lexicala(sentence))
    return features

#Generare caracteristici overall pentru un cuvant din rand
def feature_word(row):
    token = row['token']
    sentence = row['sentence']
    corpus = row['corpus']
    all_features = []
    all_features.extend(corpus_feature(corpus))
    all_features.extend(get_word_structure_features(token))
        ##print(type(get_word_structure_features(token)))
        ##print(type(corpus_feature(row['corpus'])))
        ##print(type(dale_chall_prop(row['sentence'])))
    all_features.extend(dale_chall_prop(sentence))
    #all_features.extend(get_prop_features(sentence))
    #print(np.array(all_features))
    return np.array(all_features)



#Generarea caracteristicilor overall pentr
# u tot setul de date
def feature_set_date(set_date):
    nr_of_features = len(feature_word(set_date.iloc[0]))
    #print("Nr Feature: ", nr_of_features)
    nr_of_examples = len(set_date)
    features = np.zeros((nr_of_examples, nr_of_features))
    for index, row in set_date.iterrows():
        row_ftrs = feature_word(row)
        features[index, :] = row_ftrs
    return features

#Testare
rezultat_procesare = feature_set_date(train) #X_train
complexitate=train['complex'].values #Y_train
#print(rezultat_procesare)
#print(complexitate)



#Generare submisii
submisie=feature_set_date(test)
#print (submisie)


#KNN test
timer_start=time.time()
model =KNeighborsClassifier(n_neighbors=5)
model.fit(rezultat_procesare, complexitate)
timer_sfarsit=time.time()
#print(timer_sfarsit-timer_start)
preds=model.predict(submisie)
#print (preds)
df = pan.DataFrame()
df['id'] = test.index + len(train) + 1
df['complex'] = preds
#df.to_csv('submission.csv', index=False)
count = 0
for index, row in df.iterrows():
    if(row['complex']==1):
        count=count+1
#print (count)


#Gausian Naive Base
timer_start=time.time()
model =GaussianNB()
model.fit(rezultat_procesare, complexitate)
timer_sfarsit=time.time()
#print(timer_sfarsit-timer_start)
preds=model.predict(submisie)
#print (preds)
df = pan.DataFrame()
df['id'] = test.index + len(train) + 1
df['complex'] = preds
df.to_csv('submission.csv', index=False)
count = 0
for index, row in df.iterrows():
    if(row['complex']==1):
        count=count+1
#print (count)

#10 Fold Cross Validation

def K_fold_cross_validation_and_confusion_matrix(splits,date_testare):
    split = KFold(n_splits=splits, shuffle=True)
    matrice_confuzie=[]
    for train_split, test_split in split.split(date_testare):
        # print('train: %s, test: %s' % (train.iloc[train_split], train.iloc[test_split]))
        # print("_________________________________________________")
        procesare_split = date_testare.iloc[train_split]
        procesare_split = procesare_split.reset_index()
        rezultat_procesare_split = feature_set_date(procesare_split)
        complexitate_split = date_testare.iloc[train_split]['complex'].values
        complexitate_true = date_testare.iloc[test_split]['complex'].values
        submisie_split = date_testare.iloc[test_split]
        submisie_split = submisie_split.reset_index()
        rezultat_submisie_split = feature_set_date(submisie_split)
        model = GaussianNB()
        model.fit(rezultat_procesare_split, complexitate_split)
        preds_split = model.predict(rezultat_submisie_split)
        print ("Eficiență validation: ",balanced_accuracy_score(complexitate_true, preds_split))
    ##Matrice de confuzie
        matrice_confuzie.append(confusion_matrix(complexitate_true,preds_split))
    print(sum(matrice_confuzie))


#K_fold_cross_validation_and_confusion_matrix(10,train)



