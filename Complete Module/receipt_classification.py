import pandas as pd
import csv
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

def receipt_classification():
    #np.random.seed(452)
    Stopwords = []
    all_corpus = []
    path = Path(__file__).parent.parent.absolute()
    print(path)
    stopword_file = str(path) + r"\Dataset\stopwords.csv"
    training_file = str(path) + r"\Dataset\training.csv"
    testing_file = str(path) + r"\Dataset\proceeded\output.csv"

    # Read Stopword file
    with open(stopword_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                Stopwords.append(row[0])
                line_count += 1

    Corpus = pd.read_csv(testing_file, encoding='latin-1')
    Corpus1 = pd.read_csv(training_file, encoding='latin-1')
    Corpus_all = Corpus

    print(Corpus)

    # Process Testing Data
    # Step - a : Remove blank rows if any.
    Corpus['Keyword'].dropna(inplace=True)
    # Step - b : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['Keyword']= [word_tokenize(entry) for entry in Corpus['Keyword']]
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc.
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index,entry in enumerate(Corpus['Keyword']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha() and word not in Stopwords:
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
                all_corpus.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)

    # Process Training Data
    # Step - a : Remove blank rows if any.
    Corpus1['name'].dropna(inplace=True)
    # Step - b : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus1['name']= [word_tokenize(entry) for entry in Corpus1['name']]
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index,entry in enumerate(Corpus1['name']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
            all_corpus.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus1.loc[index,'text_final'] = str(Final_words)


    Train_X, Train_Y = Corpus1['text_final'],Corpus1['label']
    Test_X, Test_Y = Corpus['text_final'],Corpus['label']
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(all_corpus)
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(all_corpus)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)


    from sklearn.metrics import classification_report, confusion_matrix

    # Classifier - Algorithm - Naive Bayes
    # fit the training dataset on the classifier
    from sklearn import naive_bayes
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
    print(classification_report(Test_Y, predictions_NB, zero_division=1))


    # Classifier - Algorithm - Maximum Entropy
    # fit the training dataset on the classifier
    from sklearn import linear_model
    from sklearn.metrics import accuracy_score
    maxent = linear_model.LogisticRegression(penalty='l2', C=1.0)
    maxent.fit(Train_X_Tfidf, Train_Y)
    predictions_ME = maxent.predict(Test_X_Tfidf)
    print("Maximum Entropy Accuracy Score -> ",accuracy_score(predictions_ME, Test_Y)*100)
    print(classification_report(Test_Y, predictions_ME, zero_division=1))


    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    from sklearn import svm
    SVM = svm.SVC(kernel='linear', C = 1.0)
    SVM.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
    print(classification_report(Test_Y, predictions_SVM, zero_division=1))


    # Classifier - Algorithm - Linear SVC
    # fit the training dataset on the classifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import classification_report
    lsvc = LinearSVC(dual=False)
    lsvc.fit(Train_X_Tfidf, Train_Y)
    predictions_LinearSVC = lsvc.predict(Test_X_Tfidf)
    print("Linear SVC Accuracy Score -> ",accuracy_score(predictions_LinearSVC, Test_Y)*100)
    print(classification_report(Test_Y, predictions_LinearSVC, zero_division=1))


    # Classifier - Algorithm - KNN
    # fit the training dataset on the classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    #Train the model using the training sets
    knn.fit(Train_X_Tfidf, Train_Y)
    #Predict the response for test dataset
    predictions_KNN = knn.predict(Test_X_Tfidf)
    # Model Accuracy, how often is the classifier correct?
    print("KNN Accuracy Score -> ",accuracy_score(predictions_KNN, Test_Y)*100)
    print(classification_report(Test_Y, predictions_KNN, zero_division=1))


    # Classifier - Algorithm - Decision Tree
    # fit the training dataset on the classifier
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(Train_X_Tfidf, Train_Y)
    predictions_DecisionTree = classifier.predict(Test_X_Tfidf)
    print("Decision Tree Accuracy Score -> ",accuracy_score(predictions_DecisionTree, Test_Y)*100)
    print(classification_report(Test_Y, predictions_DecisionTree, zero_division=1))


    # Classifier - Algorithm - Random Forest
    # fit the training dataset on the classifier
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(Train_X_Tfidf,Train_Y)
    predictions_RandomForestClassifier = clf.predict(Test_X_Tfidf)
    print("Random Forest Accuracy Score -> ",accuracy_score(predictions_RandomForestClassifier, Test_Y)*100)
    print(classification_report(Test_Y, predictions_RandomForestClassifier, zero_division=1))
