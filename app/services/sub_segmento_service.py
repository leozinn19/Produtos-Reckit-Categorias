import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# def preprocess_sub_segmento(dataframe):
#    # Pré-processar os dados existentes
#    # Separar conjuntos de números por espaços em branco
#    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
#        lambda x: re.sub(r'(\d+)', r' \1 ', x))
#
#    # Excluir todos os números do atributo 'nome_produto'
#    dataframe['nome_produto'] = dataframe['nome_produto'].apply(
#        lambda x: re.sub(r'\d', '', x))
#
#    # Remover amostras com valores NaN na coluna 'sub_segmento'
#    dataframe = dataframe.dropna(subset=['sub_segmento'])
#
#    return dataframe


def train_test_sub_segmento(dataframe_existing):
    custom_words = ['promocao', 'versao',  'nao', 'marca', 'cif',
                    'fabricante', 'musculo', 'total', 'brilhante',
                    'bufalo', 'rb', 'ingleza', 'veja', 'cp', 'unilever',
                    'johnson', 'bombril', 'promocional', 'azul', 'zupp',
                    'uau', 'ype', 'brasil', 'glade', 'triex', 'ajax',
                    'perfume', 'vidro', 'vanish']

    stopwords = nltk.corpus.stopwords.words('portuguese') + custom_words
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    
    # Preencher valores NaN com string vazia
    dataframe_existing['sub_segmento'].fillna('', inplace=True)

    # preprocess_dataframe_existing = preprocess_sub_segmento(dataframe_existing)

    # NOME PARA SUB_SEGMENTO
    X = vectorizer.fit_transform(dataframe_existing['nome_produto'])
    y = dataframe_existing['sub_segmento']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_sub_segmento(dataframe_new, vectorizer, random_forest):
    # Preencher valores NaN com string vazia
    dataframe_new['sub_segmento'].fillna('', inplace=True)

    # Prever os 'sub_segmentos'
    X_new = vectorizer.transform(dataframe_new['nome_produto'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'sub_segmentos'
    dataframe_new['sub_segmento'] = predict

    return dataframe_new
