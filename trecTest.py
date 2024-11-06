from email.parser import Parser
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# pip install transformers torch pandas scikit-learn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

'''
TREC's data set is provided by file, with spam and ham marked by an index for each message in a file. 
The following code extracts the body of the message.
Then it outputs all the body and tags of the message to a file for the next step in converting the bag model.
'''
filetype="utf-8"
# get body of email
def get_body(content):
    if content.is_multipart():
        return get_body(content.get_payload(0))
    else:
        return content.get_payload(None, decode=True)

if __name__ == '__main__':
    with open("./trec06p/full/index", encoding=filetype) as file, \
         open("pre_email.txt", "a+", encoding=filetype, errors='ignore') as fout:
        
        line = file.readline()
        num = 0
        while line:
            line = line.rstrip()
            temp = line.split(' ', 1)
            print(f"Processing file {num}/{37822}", end=" ")
            num += 1
            
            path = "/Users/yiyi/Desktop/finalNLP/trec06p/" + temp[1].lstrip('..')
            with open(path, encoding=filetype, errors='ignore') as ftemp:
                text = ftemp.read()
            email = Parser().parsestr(text)
            body = get_body(email).decode(filetype, errors='ignore').replace('\n', '').replace('\r', '').replace('\t', '').strip()
            text = f"{temp[0]}\t{body}\n"
            fout.write(text)
            line = file.readline()
    
    print("Dataset PreProcessing Finish!")

# read pre_email.txt
'''
Change to uci SMS Spam Collection
'''
# total_email = pd.read_table('./uci_spam/SMSSpamCollection.txt', sep='\t', names=['label', 'mem'])
total_email = pd.read_table('./pre_email.txt', sep='\t', names=['label', 'mem'])
# data preprocessing
total_email.dropna(inplace=True)  # drop any NaN values
total_email['label'] = total_email.label.map({'ham': 0, 'spam': 1})  # normal email is 0 and spam is 1
total_count = total_email.shape[0]
spam_email = np.count_nonzero(total_email['label'].values)  # the number of spam email
print("Total number of Email:", total_count)
print("Spam Email:", spam_email)
print("Normal Email:", total_count - spam_email)

# train_test_split() function provided by sklearn splits the data into training and test sets
stratify=total_email['label'] # enable hierarchical splitting, and the ratio of spam to ham in the test set remains the same as in the training set
x_train, x_test, y_train, y_test = train_test_split(total_email['mem'], total_email['label'], random_state = 1,
                                                     stratify = total_email['label'])
print('size of training set: {}'.format(x_train.shape[0]))
print('size of test set: {}'.format(x_test.shape[0]))

#convert to the sparse matrix
count_vector = CountVectorizer(stop_words='english')
train_data = count_vector.fit_transform(x_train)
test_data = count_vector.transform(x_test)

# tcr score
def tcr_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp = cm[0][0]
    fp = cm[1][0]
    fn = cm[0][1]
    K = 2   
    tcr = (tp + fp) / (K * fn + fp)
    return tcr

# naive bayes
naive_bayes = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
naive_bayes.fit(train_data, y_train)
pred = naive_bayes.predict(test_data)

f1 = f1_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
accuracy = accuracy_score(y_test, pred)
na_tcr = tcr_score(y_test, pred)

print(f'************ Multinomial Naive Bayes ************')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print(f'TCR Score: {na_tcr}')

# Bernoulli naive Bayes model data fitting
b_naive_bayes = BernoulliNB(alpha=1.0, class_prior=None, fit_prior=True)
b_naive_bayes.fit(train_data, y_train)
b_pred = b_naive_bayes.predict(test_data)

b_f1 = f1_score(y_test, b_pred)
b_precision = precision_score(y_test, b_pred)
b_recall = recall_score(y_test, b_pred)
b_accuracy = accuracy_score(y_test, b_pred)
b_tcr = tcr_score(y_test, b_pred)

print(f'************ Bernoulli Bayes ************')
print(f'F1 Score: {b_f1}')
print(f'Precision: {b_precision}')
print(f'Recall: {b_recall}')
print(f'Accuracy: {b_accuracy}')
print(f'TCR Score: {b_tcr}')

# Complement naive Bayes model data fitting
c_naive_bayes = ComplementNB(alpha=1.0, class_prior=None, fit_prior=True)
c_naive_bayes.fit(train_data, y_train)
c_pred = c_naive_bayes.predict(test_data)

c_f1 = f1_score(y_test, c_pred)
c_precision = precision_score(y_test, c_pred)
c_recall = recall_score(y_test, c_pred)
c_accuracy = accuracy_score(y_test, c_pred)
c_tcr = tcr_score(y_test, c_pred)

print(f'************ Complement Bayes ************')
print(f'F1 Score: {c_f1}')
print(f'Precision: {c_precision}')
print(f'Recall: {c_recall}')
print(f'Accuracy: {c_accuracy}')
print(f'TCR Score: {c_tcr}')

# KNN Algorithm
k_neighbor = KNeighborsClassifier(n_neighbors=1, weights='uniform')
k_neighbor.fit(train_data, y_train)
knn_pred = k_neighbor.predict(test_data)

k_f1 = f1_score(y_test, knn_pred)
k_precision = precision_score(y_test, knn_pred)
k_recall = recall_score(y_test, knn_pred)
k_accuracy = accuracy_score(y_test, knn_pred)
k_tcr = tcr_score(y_test, knn_pred)

print(f'************ KNN Algorithm ************')
print(f'F1 Score: {k_f1}')
print(f'Precision: {k_precision}')
print(f'Recall: {k_recall}')
print(f'Accuracy: {k_accuracy}')
print(f'TCR Score: {k_tcr}')

# SVN Algorithm
svm_clf = LinearSVC()
svm_clf.fit(train_data, y_train)
svm_pred = svm_clf.predict(test_data)

s_f1 = f1_score(y_test, svm_pred)
s_precision = precision_score(y_test, svm_pred)
s_recall = recall_score(y_test, svm_pred)
s_accuracy = accuracy_score(y_test, svm_pred)
s_tcr = tcr_score(y_test, svm_pred)

print(f'************ SVN Algorithm ************')
print(f'F1 Score: {s_f1}')
print(f'Precision: {s_precision}')
print(f'Recall: {s_recall}')
print(f'Accuracy: {s_accuracy}')
print(f'TCR Score: {s_tcr}')

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data, y_train)
dt_pred = decision_tree.predict(test_data)

dt_f1 = f1_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_tcr = tcr_score(y_test, dt_pred)

print(f'************ Decision Tree ************')
print(f'F1 Score: {dt_f1}')
print(f'Precision: {dt_precision}')
print(f'Recall: {dt_recall}')
print(f'Accuracy: {dt_accuracy}')
print(f'TCR Score: {dt_tcr}')

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(train_data, y_train)
rf_pred = random_forest.predict(test_data)

rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_tcr = tcr_score(y_test, rf_pred)

print(f'************ Random Forest ************')
print(f'F1 Score: {rf_f1}')
print(f'Precision: {rf_precision}')
print(f'Recall: {rf_recall}')
print(f'Accuracy: {rf_accuracy}')
print(f'TCR Score: {rf_tcr}')

# Gradient Boosting
gdbt = GradientBoostingClassifier()
gdbt.fit(train_data, y_train)
gdbt_pred = gdbt.predict(test_data)

gdbt_f1 = f1_score(y_test, gdbt_pred)
gdbt_precision = precision_score(y_test, gdbt_pred)
gdbt_recall = recall_score(y_test, gdbt_pred)
gdbt_accuracy = accuracy_score(y_test, gdbt_pred)
gdbt_tcr = tcr_score(y_test, gdbt_pred)

print(f'************ Gradient Boosting ************')
print(f'F1 Score: {gdbt_f1}')
print(f'Precision: {gdbt_precision}')
print(f'Recall: {gdbt_recall}')
print(f'Accuracy: {gdbt_accuracy}')
print(f'TCR Score: {gdbt_tcr}')

# Neural Network
mlp = MLPClassifier(solver='lbfgs', activation='logistic')
mlp.fit(train_data, y_train)
nn_pred = mlp.predict(test_data)

nn_f1 = f1_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_tcr = tcr_score(y_test, nn_pred)

print(f'************ Neural Network ************')
print(f'F1 Score: {nn_f1}')
print(f'Precision: {nn_precision}')
print(f'Recall: {nn_recall}')
print(f'Accuracy: {nn_accuracy}')
print(f'TCR Score: {nn_tcr}')

