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
# Fase 4: Procesamiento distribuido

# =================================================================================================

import sys
import argparse
import uuid
import os
from pathlib import Path
import json
import datetime

import spacy
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer

# =================================================================================================

# Construir producer para Kafka
producer = KafkaProducer(bootstrap_servers=['localhost:9092'], \
    value_serializer=lambda x:json.dumps(x).encode('utf-8'))

# Servidor Kafka
DEFAULT_KAFKA_SERVER = "localhost:9092"

# =================================================================================================

# Lee los argumentos pasados por línea de comandos al spark-submit
def read_args():
    parser = argparse.ArgumentParser(description='Read messages from a Kafka topic')

    s1 = parser.add_argument_group('source')
    s1.add_argument('--server', metavar='<kafka address>',
                    default=DEFAULT_KAFKA_SERVER,
                    help='Kafka address(es) to contact (default: %(default)s')
    s1.add_argument('topic', help='Kafka topic to read')

    s2 = parser.add_argument_group('Modifiers')
    s2.add_argument('--rewind', action='store_true',
                    help='read all messages from the beginning')
    s2.add_argument('--wait', type=float, default=1.0,
                    help='micro-batch interval between RDDs in the DStream')
    s2.add_argument('--id', default=str(uuid.uuid4()), help='consumer id')

    args = parser.parse_args()
    return args

# =================================================================================================

# Carga el modelo entrenado de Spacy
def load_model():
    # Especificar parámetros para cargar el modelo actualizado:

    # Cargar modelo pre-entrenado o uno actualizado en local.
    model_name = 'es_core_news_sm'

    # Cargar modelo.
    model = spacy.load(model_name)

    return model

# =================================================================================================

# Decodifica los mensajes de la cola Kafka
def msg_decoder(s):
    if s is None:
        return None
    return s.decode('unicode_escape')

# =================================================================================================

# Tranforma etiquetas Spacy en el formato BRAT
def label_to_brat(label):
    if (label == "ORG"):
        return "Organization"
    elif (label == "LOC"):
        return "GPE"
    else:
        return "Person"

# =================================================================================================

# Convierte el objeto JSON de entidades a formato BRAT
def spacy_to_brat(entities):

    brat = ''

    # Recorre la lista de entidades de entrada.
    for idx, ent in enumerate(entities):

        brat += 'T' + str(idx) + '\t' + label_to_brat(ent['label']) + ' ' + \
            str(ent['startChar']) + ' ' + str(ent['endChar']) + '\t' + ent['name'] + '\n'

    # Devuelve un cadena de texto que contiene todas las anotaciones en formato BRAT.
    return brat

# =================================================================================================

# Almacena las entidades en un fichero de anotaciones específico para la herramienta BRAT
def store_annotations(filename, news, entities):

    # Directorio raiz del proyecto
    tfm_dir = str(Path(os.getcwd()))

    # Directorio de BRAT para visualizar anotaciones
    brat_dir = tfm_dir + "/brat-v1.3_Crunchy_Frog/data/ner_output/"

    # Guardar fichero de anotaciones en formato BRAT
    with open(brat_dir + filename + '.ann', 'w') as outfile:
        outfile.write(spacy_to_brat(entities))

    # Guardar noticia plana
    with open(brat_dir + filename + '.txt', 'w') as outfile:
        outfile.write(news)

# =================================================================================================

# Procesa la noticia mediante el modelo de Spacy e indexa las entidades nombradas
def process_news(rdd):

    received = rdd.collect()

    # Recorrer noticias recibidas
    for key, msg in received:

        # Separar nombre del fichero y contenido de la noticia, y eliminar comillas de los extremos
        contents = msg[1:-1].split('#',1)

        # Lista de entidades de la noticia: nombre, posición y etiqueta
        entities_full = []

        # Lista de entidades de la noticia: únicamente el nombre
        entities_basic = []

        try:

            # Aplicar modelo de Spacy al contenido de la noticia
            nlp_doc = nlp(contents[1])

        except UnicodeEncodeError:

            # Error de formato. Esta noticia se ignorará.
            print("UnicodeEncodeError: news might contain emojis.")
            continue

        # Recorrer las entidades detectadas por Spacy dentro de la noticia actual
        for ent in nlp_doc.ents:

            # Ignorar las etiquetas Misc
            if (ent.label_ != "MISC"):

                # Añadir a lista de entidades para el objeto Brat del documento
                entities_full.append({"name":ent.text, "startChar":ent.start_char, \
                    "endChar":ent.end_char, "label":ent.label_})

                # Añadir a lista sencilla de entidades
                entities_basic.append(ent.text)

        # Guardar ficheros para visualización en BRAT
        store_annotations(contents[0], contents[1], entities_full)

        # Producir noticia y lista sencilla de entidades para la cola "enriched-news" de Kafka
        producer.send('enriched-news', value={"fullText":contents[1], "entities":entities_basic,\
            "timestamp": str(datetime.datetime.now().replace(microsecond=0))})

        # Producir lista completa de entidades para la cola "entities" de Kafka
        producer.send('entities', value=entities_full)

        producer.flush()

# =================================================================================================

if __name__ == "__main__":

    # Leer opciones de línea de comandos
    args = read_args()
    print("\n***Connecting Spark Streaming to {}, topic {}\n".format(args.server, args.topic))

    # Crear contextos de Spark y Streaming
    sc = SparkContext("local[2]", appName="streaming: kafka")
    ssc = StreamingContext(sc, args.wait)

    # Cargar modelo de Spacy
    nlp = load_model()

    # Construir el stream de Kafka
    opts = {"metadata.broker.list": args.server,
            "group.id": args.id,
            "auto.offset.reset": 'smallest' if args.rewind else 'largest'}
    kstream = KafkaUtils.createDirectStream(ssc, [args.topic], opts, valueDecoder=msg_decoder)

    # Por cada RDD recibido (noticia), buscar sus entidades nombradas
    kstream.foreachRDD(process_news)

    # Lanzar contexto de Spark Streaming
    try:
        ssc.start()
        ssc.awaitTermination()
    except (KeyboardInterrupt, Exception) as e:
        print(".. stopping streaming context", str(e))
        ssc.stop(stopSparkContext=False, stopGraceFully=True)
        sys.exit(0 if isinstance(e, KeyboardInterrupt) else 1)
