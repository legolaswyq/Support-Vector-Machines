import numpy as np
import re
import string
from nltk.stem import *
import scipy.io
from sklearn import svm


def load_vocab_dict(filename):
    # vocab is a 2-D array, 1st column is the index, 2nd column is the word
    vocab = np.loadtxt(filename, dtype=str)
    vocab_dict = {}
    vocab_list = []
    # turn vocab into dict
    for item in vocab:
        vocab_dict[item[1]] = item[0]
        vocab_list.append(item[1])

    return vocab_dict, vocab_list


def preprocess_email(filename):
    with open(filename) as file:
        data = file.read()
        # lower all content
        data = data.lower()
        #  Strip all HTML
        data = re.sub('<[^<>]+>', ' ', data)
        #  Handle Numbers
        data = re.sub('[0-9]+', 'number', data)
        #  Handle URLS
        data = re.sub('(http|https)://[^\\s]*', 'httpaddr', data)
        #  Handle Email Addresses
        data = re.sub('[^\\s]+@[^\\s]+', 'emailaddr', data)
        #  Handle $ sign
        data = re.sub('[$]+', 'dollar', data)
        contents = data.split()
    # email preprocess
    # create a punctuation dict
    # str.maketrans('char to be replace', 'replacement', 'char to remove')
    table = str.maketrans('', '', string.punctuation)
    word_idx = []
    for i in range(len(contents)):
        # remove all punctuation
        contents[i] = contents[i].translate(table)
        # remove any non alphanumeric characters
        contents[i] = re.sub('[^a-zA-Z0-9]', '', contents[i])
        # Stem the word
        try:
            stemmer = PorterStemmer()
            contents[i] = stemmer.stem(contents[i])
        except:
            print("error")
            contents[i] = ''

        if len(contents[i]) < 1:
            continue

        if contents[i] in vocab_dict:
            word_idx.append(int(vocab_dict[contents[i]])-1)

    return word_idx


def extract_email_feature(word_idx):
    m = 1899
    features = np.zeros(m, dtype=int)
    for idx in word_idx:
        features[int(idx)] = 1
    return features.reshape(1,-1)


filename = "vocab.txt"
vocab_dict, vocab_list = load_vocab_dict(filename)
# email preprocess
filename1 = "emailSample2.txt"
word_idx = preprocess_email(filename1)
features = extract_email_feature(word_idx)

filename2 = "spamTrain.mat"
filename3 = "spamTest.mat"
# load train data
train_data = scipy.io.loadmat(filename2)
X = train_data['X']
y = train_data['y'].flatten()
# load test data
test_data = scipy.io.loadmat(filename3)
X_test = test_data["Xtest"]
y_test = test_data["ytest"].flatten()
# train model with training set
svm_model = svm.SVC(kernel="linear", C=0.1)
svm_model.fit(X, y)
# predict with testing set and correctness
predict = svm_model.predict(X_test)
correctness = sum(predict == y_test) / len(y_test) * 100
# print top 15 weights
weights = svm_model.coef_.ravel()
sorted_weights = sorted(enumerate(weights), key=lambda i: i[1], reverse=True)
for item in sorted_weights[:15]:
    print(vocab_list[item[0]], item[1])


print(svm_model.predict(features))