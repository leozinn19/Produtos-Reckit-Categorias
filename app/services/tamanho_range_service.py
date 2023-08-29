from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_test_tamanho_range(dataframe_existing):
    vectorizer = TfidfVectorizer()

    # Preencher valores NaN com string vazia
    dataframe_existing['tamanho_range'].fillna('', inplace=True)
    dataframe_existing['tamanho_medida'].fillna('', inplace=True)

    # MARCA PARA tamanho_range
    X = vectorizer.fit_transform(dataframe_existing['tamanho_unidade'])
    y = dataframe_existing['tamanho_range']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_tamanho_range(dataframe_new, vectorizer, random_forest):

    # Prever os 'tamanho_ranges'
    X_new = vectorizer.transform(dataframe_new['tamanho_unidade'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'tamanho_range'
    dataframe_new['tamanho_range'] = predict
    dataframe_new['desc_b'] = dataframe_new['tamanho_range']

    return dataframe_new
