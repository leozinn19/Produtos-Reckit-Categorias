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
    dataframe.loc[dataframe['nome_produto'] == '', 'tamanho_medida'] = ''

    # Remover amostras com valores NaN na coluna 'tamanho_medida'
    dataframe = dataframe.dropna(subset=['tamanho_medida'])

    return dataframe


def train_test_tamanho(dataframe_tamanho):

    from nltk.corpus import stopwords

    custom_words = [
        'aut', 'des', 'sqz', 'gts', 'cj', 'gat', 'pt', 'lav', 'c5', 'c3', 'un', 'lim', '2d', '3x1', '4house', 'ap', 'off',
        'spr', '45n', '2c', 'c35', 'pg2', 'cx12', 'liq', 'trad', '8072', 'tul', 'ªu', 'euc', 'ªun', 'sq', 'und', 'ed', 'mar',
        'ype', 'rf', 'ul', 'man', 'pet', 'pe', 'car', 'jd', 'rom', 'lv', 'inc', 'pc', 'int', 'san', 'inf', 'hig', 'nat', 'uso',
        'tab', 'og', 'per', 'cx', 'sc', 'of', 'coz', 'ae', 'pg', 'tot', 'rap', 'ref', 'fr', 'aer', 'bg', 'cl', 'del', 'uni', 'fl',
        'bar', 'gf', 'noi', 'ft', 'mr', 'tq', 'lp', 'ca', 'pes', 'ac', 'ar', 'lem', '%', 'c', 'x', '/', 'c/', '5x1'
    ]

    stopwords = stopwords.words('portuguese') + custom_words
    vectorizer = TfidfVectorizer(stop_words=stopwords)

    preprocess_dataframe_existing = preprocess_dataframe(
        dataframe_tamanho)

    X = vectorizer.fit_transform(
        preprocess_dataframe_existing['nome_produto'])
    y = preprocess_dataframe_existing['tamanho_medida']

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


def process_tamanho(dataframe_new, vectorizer, random_forest):

    # Prever o tamanho dos novos produtos
    X_new = vectorizer.transform(
        dataframe_new['nome_produto'])
    predicted_sizes = random_forest.predict(X_new)

    # Atualizar a coluna 'tamanho_medida' com os tamanhos previstos
    dataframe_new['tamanho_medida'] = predicted_sizes
    dataframe_new['tamanho_produto'] = dataframe_new['tamanho_medida'].apply(
        lambda x: re.sub(r'[a-zA-Z]', '', str(x)))
    dataframe_new['unid_medida'] = dataframe_new['tamanho_medida'].apply(
        lambda x: re.sub(r'\d', '', str(x)))

    return dataframe_new
