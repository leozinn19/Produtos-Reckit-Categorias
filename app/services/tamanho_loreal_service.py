import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def preprocess_dataframe(dataframe):
    # Pré-processar os dados existentes
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: re.sub(r'\s', ' ', str(x).lower()))
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(lambda x: ' '.join(
        word for word in x.split() if any(c.isdigit() for c in word) and len(re.findall(r'\d', word)) <= 3))
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: re.sub(r'(\d+)', r' \1 ', x))
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: re.sub(r'\b(?![\d\s])\w{4,}\b', '', x))

    # Remover espaços em branco
    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
        lambda x: ''.join(x.split()))

    # Atualizar a coluna 'tamanho_medida' com valores vazios se 'nome_produto' estiver vazio
    dataframe.loc[dataframe['nome_produto'] == '', 'tamanho_unidade'] = ''

    # Remover amostras com valores NaN na coluna 'tamanho_unidade'
    dataframe = dataframe.dropna(subset=['tamanho_unidade'])

    return dataframe


def train_test_tamanho_loreal(dataframe_tamanho):

    vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)

    dataframe_tamanho = dataframe_tamanho[dataframe_tamanho['tamanho_unidade'] != 'unknown']

    dataframe_tamanho = dataframe_tamanho.dropna(subset=['tamanho_unidade'])

    X = vectorizer.fit_transform(
        dataframe_tamanho['nome_produto'])
    y = dataframe_tamanho['tamanho_unidade']

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Criar o classificador RandomForest
    random_forest = RandomForestClassifier(n_estimators=100)

    # Treinar o classificador com os dados de treinamento
    random_forest.fit(X_train, y_train)

    # Avaliar o desempenho do classificador nos dados de teste
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_tamanho_loreal(dataframe_new, vectorizer, random_forest):

    # Prever o tamanho dos novos produtos
    X_new = vectorizer.transform(
        dataframe_new['nome_produto'])
    predicted_sizes = random_forest.predict(X_new)

    # Atualizar a coluna 'tamanho_medida' com os tamanhos previstos
    dataframe_new['tamanho_unidade'] = predicted_sizes
    dataframe_new['tamanho_total'] = dataframe_new['tamanho_unidade'].apply(
        lambda x: re.sub(r'[a-zA-Z]', '', str(x)))
    dataframe_new['unid_medida'] = dataframe_new['tamanho_unidade'].apply(
        lambda x: re.sub(r'\d', '', str(x)))

    return dataframe_new
