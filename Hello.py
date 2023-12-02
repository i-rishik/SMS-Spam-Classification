import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

ps = PorterStemmer()

def transform_data(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    res = []
    for i in text:
        if i.isalnum():
            res.append(i)
    
    text.clear()
    
    for i in res:
        if i not in stopwords.words('english') and i not in string.punctuation:
            text.append(i)
            
    res.clear()
    
    for i in text:
        res.append(ps.stem(i))
    
    return ' '.join(res)

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_data(input_sms)
    # 2. vectorize
    vector_input = cv.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")