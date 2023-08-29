import re

from nltk import ngrams
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def preprocess_dataframe(dataframe):
    # Pré-processar os dados existentes
    # Transformar todas as letras para minusculas
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: re.sub(r'\s', ' ', str(x).lower()))

    # Remover todas as palavras que contenham dígitos e todos os dígitos
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: ' '.join(word for word in x.split() if not any(c.isdigit() for c in word)))

    # Excluindo as linhas que não possuem valor para o teste:
    dataframe.dropna(subset=['setor_gerenciado'], inplace=True)

    # Criar um stemmer RSLP
    stemmer = RSLPStemmer()
    # Aplicar o stemmer RSLP à coluna 'nome_produto'
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: ' '.join(stemmer.stem(word) for word in x.split()))

    return dataframe


def train_test_setor_gerenciado(dataframe_existing):
    from nltk.corpus import stopwords

    custom_words = ['vej', 'nao', 'cif', 'outr', 'versa',
                    'promoca', 'fabric', 'unilev', 'bombril',
                    'ingl', 'ajax', 'co', 'mr', 'uau', 'sol',
                    'azulim', 'muscul', 'johnson', 'marc',
                    'individ', 'off', 'glade', 'sem', 'x']

    stopwords = stopwords.words('portuguese') + custom_words
    vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 2))

    preprocess_dataframe_existing = preprocess_dataframe(
        dataframe_existing)

    # NOME_PRODUTO PARA SETOR_GERENCIADO
    X = vectorizer.fit_transform(preprocess_dataframe_existing['nome_produto'])
    y = preprocess_dataframe_existing['setor_gerenciado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=50)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_setor_gerenciado(dataframe_new, vectorizer, random_forest):

    # Prever o 'setor_gerenciado'
    X_new = vectorizer.transform(dataframe_new['nome_produto'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'setor_gerenciado'
    dataframe_new['setor_gerenciado'] = predict

    return dataframe_new
