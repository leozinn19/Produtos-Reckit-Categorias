from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_test_fabricante(dataframe_existing):
    vectorizer = TfidfVectorizer()

    # Remover amostras com valores NaN na coluna 'tamanho_medida'
    dataframe_existing = dataframe_existing.dropna(subset=['fabricante'])

    # MARCA PARA FABRICANTE
    X = vectorizer.fit_transform(dataframe_existing['marca'])
    y = dataframe_existing['fabricante']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_fabricante(dataframe_new, vectorizer, random_forest):

    # Prever os 'fabricantes'
    X_new = vectorizer.transform(dataframe_new['marca'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'fabricante'
    dataframe_new['fabricante'] = predict

    return dataframe_new
