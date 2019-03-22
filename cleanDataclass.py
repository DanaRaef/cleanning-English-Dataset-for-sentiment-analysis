import nltk.classify.util
import pandas as pd
import numpy as np
import re
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tokenize import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
class cleanDataclass :

    negation_words = ['none',
                      'doesnt',
                      'nonetheless',
                      'not',
                      'non',
                      'nobody',
                      "no",
                      "without",
                      "never",
                      "neither",
                      "nor",
                      "non",
                      'but',
                      "nothing",
                      "noone",
                      "other"]

    def negationhandle(self,string):
        match1 = re.search(r'\b(?:not)\b (\S+)', string)
        match2 = re.search(r'\b(?:never)\b (\S+)', string)
        match3 = re.search(r'\b(?:no)\b (\S+)', string)
        match4 = re.search(r'\b(?:none)\b (\S+)', string)
        match5 = re.search(r'\b(?:noone)\b (\S+)', string)
        match6 = re.search(r'\b(?:nobody)\b (\S+)', string)
        match7 = re.search(r'\b(?:nothing)\b (\S+)', string)
        match8 = re.search(r'\b(?:dont)\b (\S+)', string)
        match9 = re.search(r'\b(?:isnt)\b (\S+)', string)
        match10 = re.search(r'\b(?:doesnt)\b (\S+)', string)
        match11 = re.search(r'\b(?:rarely)\b (\S+)', string)
        match12 = re.search(r'\b(?:cant)\b (\S+)', string)
        match13 = re.search(r'\b(?:couldnt)\b (\S+)', string)
        match14 = re.search(r'\b(?:shouldnt)\b (\S+)', string)
        match15 = re.search(r'\b(?:wasnt)\b (\S+)', string)
        match16 = re.search(r'\b(?:arent)\b (\S+)', string)
        match17 = re.search(r'\b(?:hasnt)\b (\S+)', string)
        match18 = re.search(r'\b(?:hadnt)\b (\S+)', string)
        match19 = re.search(r'\b(?:neither)\b (\S+)', string)
        match20 = re.search(r'\b(?:nor)\b (\S+)', string)
        match21 = re.search(r'\b(?:but)\b (\S+)', string)

        if match1:
            # Yes, process it
            string = string.replace(match1.group(1), 'NEG_' + match1.group(1))

        if match2:
            # Yes, process it
            string = string.replace(match2.group(1), 'NEG_' + match2.group(1))

        if match3:
            string = string.replace(match3.group(1), 'NEG_' + match3.group(1))

        if match4:
            string = string.replace(match4.group(1), 'NEG_' + match4.group(1))

        if match5:
            string = string.replace(match5.group(1), 'NEG_' + match5.group(1))

        if match6:
            string = string.replace(match6.group(1), 'NEG_' + match6.group(1))

        if match7:
            string = string.replace(match7.group(1), 'NEG_' + match7.group(1))

        if match8:
            string = string.replace(match8.group(1), 'NEG_' + match8.group(1))

        if match9:
            string = string.replace(match9.group(1), 'NEG_' + match9.group(1))

        if match10:
            string = string.replace(match10.group(1), 'NEG_' + match10.group(1))

        if match11:
            string = string.replace(match11.group(1), 'NEG_' + match11.group(1))

        if match12:
            string = string.replace(match12.group(1), 'NEG_' + match12.group(1))

        if match13:
            string = string.replace(match13.group(1), 'NEG_' + match13.group(1))

        if match14:
            string = string.replace(match14.group(1), 'NEG_' + match14.group(1))

        if match15:
            string = string.replace(match15.group(1), 'NEG_' + match15.group(1))

        if match16:
            string = string.replace(match16.group(1), 'NEG_' + match16.group(1))

        if match17:
            string = string.replace(match17.group(1), 'NEG_' + match17.group(1))

        if match18:
            string = string.replace(match18.group(1), 'NEG_' + match18.group(1))

        if match19:
            string = string.replace(match19.group(1), 'NEG_' + match19.group(1))

        if match20:
            string = string.replace(match20.group(1), 'NEG_' + match20.group(1))

        if match21:
            string = string.replace(match21.group(1), 'NEG_' + match21.group(1))

        return string

    stop_words = stopwords.words('english')
    for w in stop_words:
        if w in negation_words:
            stop_words.remove(w)
    stop_words.append('am')
    stop_words.append('i')
    stop_words.append('pm')
    stop_words.append('tonight')
    stop_words.append('tomorrow')
    stop_words.append('today')
    stop_words.append('soon')
    stop_words.append('yesterday')
    stop_words.append('week')
    stop_words.append('day')
    stop_words.append('morning')
    stop_words.append('evening')
    stop_words.append('yayayayayaay')
    stop_words.append('tasha')
    stop_words.append('sunday')
    stop_words.append('monday')
    stop_words.append('tuesday')
    stop_words.append('wednesday')
    stop_words.append('thursday')
    stop_words.append('friday')
    stop_words.append('saturday')
    stop_words.append('weekend')
    # print (stop_words)
    # lemmatizer
    lem = WordNetLemmatizer()

    # abbreviation dictionary
    CONTRACTION_MAP = {
        "omg": "oh my god",
        "wtf": "what the fuck",
        "gonna": "going to",
        "gotta":"got",
        "k": "okay",
        "ok": "okay",
        "ain't": "is not",
        "aren't": "are not",
        "can't": "can not",
        "cant": "can not",
        "can't've": "can not have",
        "cause": "because",
        "cuz": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "didnt": "did not",
        "doesn't": "does not",
        "doesnt": "does not",
        "don't": "do not",
        "dont": "do not",
        "hadn't": "had not",
        "hadnt": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "hasnt": "has not",
        "haven't": "have not",
        "havent": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "im": "i am",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "ive": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "lol": "laugh out loud",
        "ma'am": "madam",
        "maam": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "nope" :'no',
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "thx": "thank",
        "thanx": "thank",
        "thanks": "thank",
        "u": "you",
        "wasn't": "was not",
        "wasn": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "y": "why",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        "ya": 'yes',
        "fucker": "fuck",
        "fuckin": "fuck",
        "fucking": "fuck"
    }

    def clean (self,df):
        df_new= []
        df = [re.sub(r"(?:\@|https?\://)\S+", "", t) for t in df]
        df = [re.sub(r'(.)\1{2,}', r'\1', t) for t in df]
        df = [re.sub("[^a-zA-Z]", " ", t) for t in df]
        df = [t.lower() for t in df]
        for t in df :
            t = ' '.join([self.CONTRACTION_MAP.get(item, item) for item in t.split()])
            df_new.append(t)
        df = df_new
        df_new = []
        df = self.pos_tag(df)


        for t in df :
            t = ' '.join(item for item in t.split() if item not in self.stop_words)
            df_new.append(t)
        df = df_new
        df_new=[]


        df = [t.strip() for t in df]
        # df = [self.negationhandle(t) for t in df]

        # for t in df :
        #     t = ' '.join(self.lem.lemmatize(item) for item in t.split())
        #     df_new.append(t)
        # df = df_new


        print ("##############################")
        print (df)
        return df



    def pos_tag (self,df):
        tagged = []
        for text in df :
            tok = word_tokenize(text)
            tagged.append(nltk.pos_tag(tok))

        return(self.Join_tagged(tagged))

    def Join_tagged (self,listOflists):
        delimeter = " "
        sentences = []
        for l in listOflists:
            # print('l is ', l)

            temp =[]
            for word in l:
                if word[0] not in self.stop_words:
                    if word[0] in self.negation_words:
                        temp.append(word[0] + '_' + 'NEG')
                    else :
                        temp.append(word[0] + '_' + word[1])


            sentences.append(delimeter.join(temp))
        return sentences
df = pd.read_pickle('training.1600000.processed.noemoticon_cleaned.pkl')
cleanDataclass = cleanDataclass()
cleaned=cleanDataclass.clean(df['text'])
df['text']=cleaned
df.to_pickle('NEGTagged_POSTagged_NoLemma_STOPWORDSremoved_1Million.pkl')
