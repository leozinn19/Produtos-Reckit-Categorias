import pandas as pd
# import matplotlib.pyplot as plt

# from wordcloud import WordCloud
# from termcolor import colored
from services.data_service import save_updated_dataframe, acurracy, xlsx_to_csv
# from services.tamanho_service import train_test_tamanho, process_tamanho
from services.tamanho_range_service import train_test_tamanho_range, process_tamanho_range
from services.marca_service import train_test_marca, process_marca
from services.segmento_service import train_test_segmento, process_segmento
from services.cod_categoria_service import train_test_cod_categoria, process_cod_categoria
from services.sub_segmento_service import train_test_sub_segmento, process_sub_segmento
from services.setor_gerenciado_service import train_test_setor_gerenciado, process_setor_gerenciado
from services.setor_produto_service import train_test_setor_produto, process_setor_produto
from services.sub_categoria_service import train_test_sub_categoria, process_sub_categoria
from services.fabricante_service import train_test_fabricante, process_fabricante
from services.fabricante_varejista_service import train_test_fabricante_varejista, process_fabricante_varejista
from services.dcr_oferta_service import train_test_dcr_oferta, process_dcr_oferta
from services.desc_e_service import train_test_desc_e, process_desc_e
from services.tamanho_service import train_test_tamanho, process_tamanho

# Converter a base xlsx para csv:
xlsx_to_csv('../bases/PURIFICADORES.xlsx', '../bases/BASE_CONVERTIDA.csv')

# Ler os dados existentes do arquivo CSV
dataframe_existing = pd.read_csv('../bases/BASE_CONVERTIDA.csv')

dataframe_tamanho = dataframe_existing.copy()
dataframe_cod_categoria = dataframe_existing.copy()
dataframe_setor_gerenciado = dataframe_existing.copy()
dataframe_setor_produto = dataframe_existing.copy()
dataframe_sub_categoria = dataframe_existing.copy()

# Pré-processar os dados existentes e treinar os classificadores
vectorizer_cod_categoria, random_forest_cod_categoria, accuracy_cod_categoria = train_test_cod_categoria(
    dataframe_cod_categoria)
print('Processo 1 feito...')
vectorizer_setor_gerenciado, random_forest_setor_gerenciado, accuracy_setor_gerenciado = train_test_setor_gerenciado(
    dataframe_setor_gerenciado)
print('Processo 2 feito...')
vectorizer_setor_produto, random_forest_setor_produto, accuracy_setor_produto = train_test_setor_produto(
    dataframe_setor_produto)
print('Processo 3 feito...')
vectorizer_segmento, random_forest_segmento, accuracy_segmento = train_test_segmento(
    dataframe_existing)
print('Processo 4 feito...')
vectorizer_sub_segmento, random_forest_sub_segmento, accuracy_sub_segmento = train_test_sub_segmento(
    dataframe_existing)
print('Processo 5 feito...')
vectorizer_marca, random_forest_marca, accuracy_marca = train_test_marca(
    dataframe_existing)
print('Processo 6 feito...')
vectorizer_sub_categoria, random_forest_sub_categoria, accuracy_sub_categoria = train_test_sub_categoria(
    dataframe_sub_categoria)
print('Processo 7 feito...')
vectorizer_fabricante, random_forest_fabricante, accuracy_fabricante = train_test_fabricante(
    dataframe_existing)
print('Processo 8 feito...')
vectorizer_fabricante_varejista, random_forest_fabricante_varejista, accuracy_fabricante_varejista = train_test_fabricante_varejista(
    dataframe_existing)
print('Processo 9 feito...')
vectorizer_dcr_oferta, random_forest_dcr_oferta, accuracy_dcr_oferta = train_test_dcr_oferta(
    dataframe_existing)
print('Processo 10 feito...')
vectorizer_desc_e, random_forest_desc_e, accuracy_desc_e = train_test_desc_e(
    dataframe_existing)
print('Processo 11 feito...')

print('Processo 12 feito...')
vectorizer_tamanho, random_forest_tamanho, accuracy_tamanho = train_test_tamanho(
   dataframe_tamanho)
print('Processo 1 feito...')
vectorizer_tamanho_range, random_forest_tamanho_range, accuracy_tamanho_range = train_test_tamanho_range(
   dataframe_existing)
print('Processo 12 feito...')

# Mostrar a acuracia dos classificadores
acurracy('COD_CATEGORIA', accuracy_cod_categoria)
acurracy('SETOR_GERENCIADO', accuracy_setor_gerenciado)
acurracy('SETOR_PRODUTO', accuracy_setor_produto)
acurracy('SEGMENTO', accuracy_segmento)
acurracy('SUB_SEGMENTO', accuracy_sub_segmento)
acurracy('MARCA_VAREJISTA', accuracy_marca)
acurracy('SUB_CATEGORIA', accuracy_sub_categoria)
acurracy('FABRICANTE', accuracy_fabricante)
acurracy('FABRICANTE_VAREJISTA', accuracy_fabricante_varejista)
acurracy('DCR_OFERTA', accuracy_dcr_oferta)
acurracy('DESC_E', accuracy_desc_e)

acurracy('TAMANHO', accuracy_tamanho)
acurracy('TAMANHO_RANGE', accuracy_tamanho_range)

# Calcular a frequência das palavras
# frequencies = X.sum(axis=0)
# word_frequencies = {word: frequency for word, frequency in zip(
#    vectorizer_setor_gerenciado.get_feature_names_out(), frequencies.tolist()[0])}
# Criar a nuvem de palavras com max_words=500
# wordcloud = WordCloud(width=800, height=800, background_color='white',
#                      max_font_size=150,  max_words=800).generate_from_frequencies(word_frequencies)
# Exibir a nuvem de palavras
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# Converter os novos produtos de xlsx para csv:
# xlsx_to_csv('../bases/NOVOS_PRODUTOS.xlsx',
#            '../bases/NOVOS_PRODUTOS_CONVERTIDOS.csv')
# Ler os novos produtos a serem adicionados
# dataframe_new = pd.read_csv('../bases/NOVOS_PRODUTOS_CONVERTIDOS.csv')


# Processar os novos produtos
# processed_dataframe_new = process_setor_gerenciado(
#    dataframe_new, vectorizer_setor_gerenciado, random_forest_setor_gerenciado)
# processed_dataframe_new = process_segmento(
#    dataframe_new, vectorizer_segmento, random_forest_segmento)
# processed_dataframe_new = process_sub_segmento(
#    dataframe_new, vectorizer_sub_segmento, random_forest_sub_segmento)
# processed_dataframe_new = process_cod_categoria(
#    dataframe_new, vectorizer_cod_categoria, random_forest_cod_categoria)
# processed_dataframe_new = process_marca(
#    dataframe_new, vectorizer_marca, random_forest_marca)
# processed_dataframe_new = process_setor_produto(
#    dataframe_new, vectorizer_setor_produto, random_forest_setor_produto)
# processed_dataframe_new = process_sub_categoria(
#    dataframe_new, vectorizer_sub_categoria, random_forest_sub_categoria)
# processed_dataframe_new = process_fabricante(
#    dataframe_new, vectorizer_fabricante, random_forest_fabricante)
# processed_dataframe_new = process_fabricante_varejista(
#    dataframe_new, vectorizer_fabricante_varejista, random_forest_fabricante_varejista)
# processed_dataframe_new = process_dcr_oferta(
#    dataframe_new, vectorizer_dcr_oferta, random_forest_dcr_oferta)
# processed_dataframe_new = process_desc_e(
#    dataframe_new, vectorizer_desc_e, random_forest_desc_e)

# processed_dataframe_new = process_tamanho_loreal(
#    dataframe_new, vectorizer_tamanho, random_forest_tamanho)
# processed_dataframe_new = process_tamanho_range(
#    dataframe_new, vectorizer_tamanho_range, random_forest_tamanho_range)
# Salvar o DataFrame atualizado
# save_updated_dataframe(processed_dataframe_new)
