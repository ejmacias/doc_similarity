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
# Fase 3: Persistencia

# =================================================================================================

import os
import json

# =================================================================================================

# Tranforma etiquetas Spacy en el formato BRAT.
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

# Detecta entidades y características lingüísticas de las noticias, genera ficheros JSON
# y devuelve la lista de acciones para indexar en Elastic Search.
def process_news(data_dir, nlp, newsIndex, entityIndex, lingIndex):

    # Lista de acciones a realizar para ElasticSearch
    actions = []

    # Guardar lista de entidades de cada noticia para después construir la matriz de distancias.
    entities_by_news = {}

    # Recorrer ficheros txt de noticias planas.
    for file in os.listdir(data_dir):

        # Listas de entidades de la noticia: una con nombre, posición y etiqueta; otra solo con nombres.
        ent_list_full = []
        ent_list_basic = []
        pos_list = []
        lemma_list = []

        filename = os.fsdecode(file)
        if filename.endswith(".txt"):

            # Cargar noticia plana en Spacy.
            file = data_dir + "/" + filename
            nlp_doc = nlp(open(file, "r").read())


            # Recorrer las entidades detectadas por Spacy dentro de la noticia actual.
            for ent in nlp_doc.ents:

                # Ignorar las etiquetas Misc.
                if (ent.label_ != "MISC"):

                    # Añadir a lista completa de entidades para el objeto Brat del documento.
                    ent_list_full.append({"name":ent.text, "startChar":ent.start_char, "endChar":ent.end_char,
                        "label":ent.label_})

                    # Añadir a lista básica de entidades.
                    ent_list_basic.append(ent.text)

                    # Crear documento con entidad nombrada solamente para poder agregar en Kibana
                    action = {
                        "_index":entityIndex,
                        "_type":entityIndex.rstrip('-index'),
                        "_source": {
                            "name":ent.text,
                            "label":ent.label_ } }

                    # Añadir entidad nombrada a la lista de acciones para ES
                    actions.append(action)

            # Guardar lista de entidades para después construir la matriz de distancias entre noticias.
            entities_by_news[filename.rstrip('.txt')] = ent_list_basic

            # Recorrer los tokens detectados por Spacy
            for token in nlp_doc:

                # Añadir a la lista el part-of-speech del token
                pos_list.append(token.pos_)

                # Añadir a la lista el lema del token
                lemma_list.append(token.lemma_)

                # Crear documento con POS y lema solamente para poder agregar en Kibana
                action = {
                    "_index":lingIndex,
                    "_type":lingIndex.rstrip('-index'),
                    "_source": {
                        "word":token.text,
                        "pos":token.pos_,
                        "lemma":token.lemma_ } }

                # Añadir POS y lema a la lista de acciones para ES
                actions.append(action)


            # Construir objeto JSON
            data = {
                "brat":spacy_to_brat(ent_list_full),
                "entities":ent_list_basic,
                "numSent":len([sentence for sentence in nlp_doc.sents]),
                "pos":pos_list,
                "lemmas":lemma_list,
                "fullText":str(nlp_doc) }

            # Crear documento general para ES
            action = {
                "_index":newsIndex,
                "_type":newsIndex.rstrip('-index'),
                "_source":data }

            # Añadir documento general de esta noticia a la lista de acciones para ES
            actions.append(action)

            # Guardar fichero JSON
            with open(file.rstrip('.txt') + '.json', 'w') as outfile:
                json.dump(data, outfile, ensure_ascii=False)

    return actions, entities_by_news

# =================================================================================================
