import nltk
import matplotlib.pyplot as plt
import re
import random

pos_reviews=open('positive.txt','r').read()
neg_reviews=open('negative.txt','r').read()

documents=[] # contains around 10000 reviews both postive and negative.
for review in pos_reviews.split('\n'):
    documents.append((review,'pos'))
for review in neg_reviews.split('\n'):
    documents.append((review,'neg'))

random.shuffle(documents) #shuffling the document list because the reviews were in the order
#of postive reviews first and then negative reviews which we don't want. so shuffled.

#All the reviews are splitted and cleaned from certain special charcters and stored in a list.
#It contains all the words.
all_words=[] # contains around 1 lakh words.
pos_words=re.findall(r'[a-z]*',pos_reviews)
neg_words=re.findall(r'[a-z]*',neg_reviews)

for word in pos_words:
    if word != '' and len(word)>=3:
        all_words.append(word.lower())

for word in neg_words:
    if word != '' and len(word)>=3:
        all_words.append(word.lower())


all_words=nltk.FreqDist(all_words) #Used to distribute the words which are more frequently occuring.
word_features=list(all_words.keys())[:5000] #including only 5000 word features.

#this method extracts the features of the Reviews
def find_features(document):
    document_words=[]
    regex=re.findall(r'[a-z]*',document)
    for r in regex:
        if r != '' and len(r)>=3:
            document_words.append(r.lower())

    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

feature_set=[(find_features(rev),category) for (rev,category) in documents]

#splitting the feature_set into Training_set(0 to 10000) and Testing_set(10000 to last)
training_set=feature_set[:10000]
testing_set=feature_set[10000:]

print("Training... Please wait !!")
classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Training Has Been Completed\n")
print("Accuracy Percentage:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)


#Here the Testing Set is used to predict the sentiments based on the algorithm accuracy. we can several other testing set to do the sentiment.
#Note: The extraction of features of testing should be done before classifying its sentiment.
neg_count=0
pos_count=0
for (test_data,category) in testing_set:
    result=classifier.classify(test_data)
    if result == 'pos':
        pos_count+=1
    else:
        neg_count+=1

total=pos_count+neg_count
print("\nTotal reviews:",total) #It outputs the total number of reviews.
print("\nNegative Reviews count:",neg_count) #it outputs the total negative reviews in the testing set.
print("\nPositive Reviews Count:",pos_count) #It outputs the total positive reviews in the testing set.

#Using the Python Library Matplotlib for data visualization.
#Plotting a pie chart for the testing data. It shows the Postivity score and Negativity score in percent
def sentiment_analysis_pie_chart(neg,pos,total):
    neg=(neg_count/total)*100
    pos=(pos_count/total)*100
    Sentiment = ['Negativity','Positivity']
    per_data = [neg,pos]
    color=['#fb4642','#1f77c9']
    explode = (0.1, 0)
    plt.pie(per_data,labels=Sentiment,colors=color,explode=explode,shadow=True,autopct='%1.1f%%')
    plt.legend(Sentiment)
    plt.axis('equal')
    plt.title('Sentiment Analysis Pie Chart.')
    return plt.show()

pie_chart=sentiment_analysis_pie_chart(neg_count,pos_count,total)
