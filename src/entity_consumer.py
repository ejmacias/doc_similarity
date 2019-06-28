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
# Fase 4: Persistencia distribuida 2

# =================================================================================================

from kafka import KafkaConsumer
from elasticsearch import Elasticsearch, helpers
import json, logging

# =================================================================================================

# Construir consumer para la cola "entities"
consumer = KafkaConsumer(
    'entities',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Duplicar información para poder agregar entidades en ElasticSearch
# (Kibana no soporta objetos anidados)
entityIndex = "entity-index"

# Establecer comunicación con ES
es_client = Elasticsearch(hosts = [{'host':'localhost', 'port':9200}])

# Deshabilitar logs de ES
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)

# Iterar por los mensajes recibidos en la cola
for message in consumer:

    # Lista documentos JSON asociados a una noticia
    actions = []

    # Iterar por las entidades de la noticia
    for entity in message.value:

        # Crear documento con entidad nombrada, solamente para poder agregar en Kibana
        actions.append({
            "_index":entityIndex,
            "_type":entityIndex.rstrip('-index'),
            "_source":entity})

    # Indexar todas las entidades de la noticia en ES
    helpers.bulk(es_client, actions)
