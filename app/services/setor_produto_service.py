import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def preprocess_dataframe(dataframe):
    # Converter a coluna 'nome_produto' para string
    dataframe['nome_produto'] = dataframe['nome_produto'].astype(str)

    # Excluindo as linhas que não possuem valor para o teste:
    dataframe.dropna(subset=['cod_categoria'], inplace=True)
    dataframe.dropna(subset=['setor_gerenciado'], inplace=True)

    # Pré-processar os dados existentes para a coluna 'nome_produto' do segmento
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: re.sub(r'\b(?:\d+\w*|\w*\d+)\b', '', x))

    # Retirar palavras com 2 letras ou menos
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: ' '.join(word for word in x.split() if len(word) > 1))

    return dataframe


def train_test_setor_produto(dataframe_existing):
    from nltk.corpus import stopwords
    custom_words = ['marca', 'fabricante', 'promocao', 'nao']

    stopwords = stopwords.words('portuguese') + custom_words
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    preprocess_dataframe_existing = preprocess_dataframe(
        dataframe_existing)

    # Excluindo as linhas que não possuem valor para o teste:
    preprocess_dataframe_existing.dropna(
        subset=['setor_produto'], inplace=True)

    # NOME_PRODUTO PARA SEGMENTO
    X = vectorizer.fit_transform(preprocess_dataframe_existing['nome_produto'])
    y = preprocess_dataframe_existing['setor_produto']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_setor_produto(dataframe_new, vectorizer, random_forest):

    # Prever os 'SEGMENTOS'
    X_new = vectorizer.transform(dataframe_new['nome_produto'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'segmento'
    dataframe_new['setor_produto'] = predict

    return dataframe_new
