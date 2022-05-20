import pandas as pd
import numpy as np
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from unicodedata import normalize
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora.dictionary import Dictionary
from gensim.models import word2vec


def TextProcessing(df):

    # Remove all HTML entities
    pattern = r"\&\#[0-9]+\;"
    df["processedText"] = df["Text"].str.replace(pat=pattern, repl="", regex=True)

    # Create Lemmatizer object
    lemma = WordNetLemmatizer()

    def lemmatize_word(tagged_token): #Returns lemmatized word given its tag
        root = []
        for token in tagged_token:
            tag = token[1][0]
            word = token[0]
            if tag.startswith('J'):
                root.append(lemma.lemmatize(word, wordnet.ADJ))
            elif tag.startswith('V'):
                root.append(lemma.lemmatize(word, wordnet.VERB))
            elif tag.startswith('N'):
                root.append(lemma.lemmatize(word, wordnet.NOUN))
            elif tag.startswith('R'):
                root.append(lemma.lemmatize(word, wordnet.ADV))
            else:          
                root.append(word)
        return root

    def lemmatize_doc(document): #Tags words then returns sentence with lemmatized words"""
        lemmatized_list = []
        tokenized_sent = sent_tokenize(document)
        for sentence in tokenized_sent:
            no_punctuation = re.sub(r"[`'\",.!?()]", " ", sentence)
            tokenized_word = word_tokenize(no_punctuation)
            tagged_token = pos_tag(tokenized_word)
            lemmatized = lemmatize_word(tagged_token)
            lemmatized_list.extend(lemmatized)
        return " ".join(lemmatized_list)

    df["processedText"] = df["processedText"].apply(lambda row: lemmatize_doc(row))

    
    # Remove accents
    remove_accent = lambda text: normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    df["processedText"] = df["processedText"].apply(remove_accent)

    # Remove punctuation
    pattern = r"[^\w\s]"
    df["processedText"] = df["processedText"].str.replace(pat=pattern, repl=" ", regex=True)

    # Convert to lower case
    df["processedText"] = df["processedText"].str.lower()

    # Remove stop words
    stop_words = stopwords.words("english")
    stop_words = [word.replace("\'", "") for word in stop_words]
    remove_stop_words = lambda row: " ".join([token for token in row.split(" ") if token not in stop_words])
    df["processedText"] = df["processedText"].apply(remove_stop_words)

    # Remove extra phrases
    pattern = r"[\s]+"
    df["processedText"] = df["processedText"].str.replace(pat=pattern, repl=" ", regex=True) 

    # Tokenizatiom
    corpora = df["processedText"].values
    tokenized = [corpus.split(" ") for corpus in corpora]

    # Identify phrases - words that appear frequently together
    bi_gram = Phrases(tokenized, min_count=300, threshold=50)
    tri_gram = Phrases(bi_gram[tokenized], min_count=300, threshold=50) 
    tokenized = [Phraser(tri_gram)[Phraser(bi_gram)[i]] for i in tokenized]

    # Creating the vocabulary
    vocabulary = Dictionary(tokenized)
    vocabulary_keys = list(vocabulary.token2id)[0:10]
    for key in vocabulary_keys:
        print(f"ID: {vocabulary.token2id[key]}, Token: {key}")
    
    # Create Word2Vec model
    np.set_printoptions(suppress=True)
    feature_size = 100
    context_size = 20
    min_word = 1

    word_vec= word2vec.Word2Vec(tokenized, size=feature_size, window=context_size, min_count=min_word, iter=50, seed=42)
    word_vec_unpack = [(word, idx.index) for word, idx in word_vec.wv.vocab.items()]
    tokens, indexes = zip(*word_vec_unpack)
    word_vec_df = pd.DataFrame(word_vec.wv.syn0[indexes, :], index=tokens)
    tokenized_array = np.array(tokenized)
    model_array = np.array([word_vec_df.loc[doc].mean(axis=0) for doc in tokenized_array])
    word_vec_df = pd.DataFrame(model_array)

    return word_vec_df





def process(df):
    # This is where you can do all your processing

    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)

    # Since length of the review distinguishes between Classes 1,5 and 2,3,4 ver well, it is added as a column. 
    df['LengthOfReview'] = df['Text'].str.len()
    df['LengthOfReview'] = df['LengthOfReview'].fillna(0)

    # Vader sentiment analyzer is used to get sentiment scores for 'Summary' and 'Text'
    sentiment = SentimentIntensityAnalyzer()
    df['Summary'] = df['Summary'].fillna("")
    polarity = [round(sentiment.polarity_scores(i)['compound'], 2) for i in df['Summary']]
    df['SummaryScore'] = polarity

    df['Text'] = df['Text'].fillna("")
    polarity = [round(sentiment.polarity_scores(i)['compound'], 2) for i in df['Text']]
    df['TextScore'] = polarity

    # Process Text using Word2Vec technique
    word_vec_df = TextProcessing(df)
    df = pd.concat([df, word_vec_df], axis = 1)

    return df


# Load the dataset
trainingSet = pd.read_csv("./data/train.csv")

# Process the DataFrame
train_processed = process(trainingSet)

print(train_processed.head(10))

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

# Remove unwanted columns
trainX = trainX.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'processedText', 'Time'])
testX = testX.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary','HelpfulnessNumerator', 'HelpfulnessDenominator', 'processedText', 'Time'])

testX.to_csv("./data/X_test.csv", index=False)
trainX.to_csv("./data/X_train.csv", index=False)