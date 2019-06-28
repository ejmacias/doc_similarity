# TRABAJO FIN DE MASTER
#
# Máster en Big Data y Data Science
#
# Título: Descubriendo similitud entre documentos a partir de entidades nombradas
#
# Tutor: Pablo A. Haya Coll
#
# Estudiante: Emilio J. Macias Macias
#
# Fase 3: Script para crear un grafo de noticias a partir de las entidades indexadas en ES.

# =================================================================================================

import sys
import os
from pathlib import Path
from elasticsearch import Elasticsearch
import datetime
import visualization
import spacy
import pandas as pd

# =================================================================================================

if len(sys.argv) == 2 and sys.argv[1].isdecimal():

    # Establecer comunicación con ES
    es_client = Elasticsearch(hosts = [{'host':'localhost', 'port':9200}])

    # Directorio raiz del proyecto.
    tfm_dir = str(Path(os.getcwd()).parent)

    # Variables temporales
    minutes = int(sys.argv[1])
    now = datetime.datetime.now()
    timevar = (now - datetime.timedelta(minutes=minutes)).replace(microsecond=0)

    # Consultar ES para obtener las noticias de los últimos X minutos
    res = es_client.search(index="news-index", size=10000, body={"query": {"range": {"timestamp":{"gte":str(timevar)}}}})

    # Guardar lista de entidades de cada noticia para después construir la matriz de distancias.
    entities_by_news = {}
    for hit in res['hits']['hits']:
        entities_by_news[hit['_id']] = hit['_source']['entities']

    # Comprobar que al menos una noticia cumple el requisito temporal.
    if (len(entities_by_news) > 0):

        print(str(res['hits']['total']), 'noticias desde', timevar)

        # Modelo de vectores (inglés). Usado para detectar similitud entre listas de entidades nombradas.
        nlp_vectors = spacy.load('en_vectors_web_lg')

        # Cargar matriz de distancias conocidas.
        matrixpath = str(Path(os.getcwd())) + '/distmatrix.csv'
        old_dist_matrix = pd.DataFrame()

        try:
            old_dist_matrix = pd.read_csv(matrixpath, index_col=0)

        except pd.errors.EmptyDataError:
            print ('Distance matrix file not found.')

        # Calcular matriz de distancias entre documentos.
        dist_matrix, edges = visualization.find_doc_distances(entities_by_news, nlp_vectors, old_dist_matrix)

        # Guardar matriz de distancias conocidas.
        dist_matrix.to_csv(matrixpath)

        # Filtrar por tamaño de comunidades, para representar solamente las que tengan un tamaño por encima
        # del umbral (% mínimo de noticias sobre el total).
        threshold = 0.05

        # Crear grafo de noticias y comunidades.
        visualization.plot_graph(dist_matrix, edges, threshold, entities_by_news, tfm_dir)

    else:
        print('Error: no existen noticias en ese intervalo.')

else:
    print('Error: introduce el número de minutos.')
