#!/usr/bin/env python
# coding: utf-8

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
# Fase 4: Persistencia distribuida 1

# =================================================================================================

from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
import json, logging

# =================================================================================================

# Construir consumer para la cola "enriched-news"
consumer = KafkaConsumer(
    'enriched-news',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Índice de documentos para ElasticSearch
newsIndex = "news-index"

# Establecer comunicación con ES
es_client = Elasticsearch(hosts = [{'host':'localhost', 'port':9200}])

# Deshabilitar logs de ES
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)

# Iterar por los mensajes recibidos en la cola
for message in consumer:

    # Indexar noticia en ES junto con su lista de entidades
    es_client.index(
        index=newsIndex,
        doc_type=newsIndex.rstrip('-index'),
        body=message.value)
