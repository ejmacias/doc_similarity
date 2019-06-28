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
# Fase 4: Ingesta

# =================================================================================================

from kafka import KafkaProducer
import json
from time import sleep
from pathlib import Path
import os
import math
import random

# =================================================================================================

# Ejecuta una espera en base al proceso de Poisson.
def waitForNews(rateParameter):
    sleep(-math.log(1.0 - random.random()) / rateParameter)
    return

# =================================================================================================

# Construir producer para Kafka
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x:json.dumps(x).encode('utf-8'))

# Directorio raiz del proyecto
tfm_dir = str(Path(os.getcwd()))

# Añadir directorio de noticias
data_dir = tfm_dir + '/data_rep'

# Para modelar un proceso de Poisson (cuánto debemos esperar hasta la próxima noticia),
# usamos una función de distribución exponencial
# De media, una noticia llega cada 5 segundos.
rateParameter = 1/5

# Recorrer ficheros txt de noticias planas.
for file in os.listdir(data_dir):

    filename = os.fsdecode(file)

    if filename.endswith(".txt"):

        filepath = data_dir + '/' + filename

        with open(filepath, 'r') as f:

            print("producing news:" + filename + "...")

            # Leer noticia junto con el nombre del fichero, añadiendo un separador
            data = filename.rstrip('.txt') + '#' + f.read()

            # Enviar mensaje a la cola "raw-news"
            producer.send('raw-news', value=data)

            # Esperar hasta la próxima noticia mediante distribución de Poisson.
            waitForNews(rateParameter)

        # borrar fichero para no volver a procesarlo de nuevo (ya estará anotado en Brat e indexado en ES)
        #os.remove(filepath) ###########################################################################################
