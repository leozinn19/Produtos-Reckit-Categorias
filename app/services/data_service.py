import pandas as pd
from termcolor import colored

def xlsx_to_csv(xlsx_file, csv_file):
    dataframe = pd.read_excel(xlsx_file)
    dataframe.to_csv(csv_file, index=False)


def acurracy(column, accuracy):
    print('Acurácia do classificador ' + column + ' :',
          colored(accuracy, attrs=['underline']))


def save_updated_dataframe(final_dataframe):
    # Salvar o novo DataFrame em um arquivo XLSX atualizado, incluindo a coluna original 'nome_produto'
    final_dataframe.to_excel(
        '../bases/BASE_NOVA.xlsx', index=False)
    # Contar o número de linhas do novo DataFrame
    num_rows = final_dataframe.shape[0]
    print(colored('Produtos novos cadastrados', 'blue'), colored(
        num_rows, 'green', attrs=['underline']))
