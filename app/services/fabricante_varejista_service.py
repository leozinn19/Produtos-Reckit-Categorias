from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_test_fabricante_varejista(dataframe_existing):
    vectorizer = TfidfVectorizer()

    # Converter valores da coluna 'fabricante_varejista' em strings
    dataframe_existing['fabricante'] = dataframe_existing['fabricante'].astype(
        str)

    # Excluindo as linhas que não possuem valor para o teste:
    dataframe_existing.dropna(subset=['fabricante_varejista'], inplace=True)

    # FABRICANTE PARA FABRICANTE_VAREJISTA
    X = vectorizer.fit_transform(dataframe_existing['fabricante'])
    y = dataframe_existing['fabricante_varejista']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_fabricante_varejista(dataframe_new, vectorizer, random_forest):

    # Prever os 'fabricantes verejistas'
    X_new = vectorizer.transform(dataframe_new['fabricante'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'fabricante_varejista'
    dataframe_new['fabricante_varejista'] = predict

    return dataframe_new
