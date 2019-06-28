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
# Fase 2: Mejora del reconocimiento. Training

# =================================================================================================

import spacy
from spacy.matcher import PhraseMatcher
from spacy.gold import GoldParse
from spacy.util import minibatch, compounding
import csv, os
from pathlib import Path
import random

# =================================================================================================

# Repasar entidades ya conocidas por el modelo para que no las olvide.
def revise_data(nlp, filename, revision_data):

    result = False

    # Abrir fichero de la noticia.
    file = open(filename, "r")

    # Dividir noticias anotadas de entrenamiento y revisión en líneas para evitar
    # un uso excesivo de la memoria a la hora de entrenar el modelo.
    for sentence in file:

        # Para evitar errores de memoria, evitar entrenar con frases demasiado largas.
        if (len(sentence) < 3000):

            # Cargar frase en Spacy.
            doc = nlp(sentence)

            # Para evitar errores de memoria, evitar entrenar con listas de entidades demasiado largas.
            if (len(doc.ents) < 40):

                # Una lista de anotaciones para cada frase de cada noticia.
                annotations = []

                # Iterar por las entidades detectas por el modelo pre-entrenado.
                for nlp_ent in doc.ents:

                    # Añadir índices de caracter junto con tipo de entidad a la lista de anotaciones.
                    annotations.append((nlp_ent.start_char, nlp_ent.end_char, nlp_ent.label_))

                # Añadir frase de la noticia plana junto con su lista de anotaciones a los datos de revisión.
                revision_data.append((str(doc), annotations))

                result = True

    return revision_data, result

# =================================================================================================

# Generar ejemplos de entrenamiento a partir del PhraseMather y la lista CSV de entidades.
def generate_data(nlp, ent_matcher, entities, filename, matching_data):

    result = False

    # Abrir fichero de la noticia.
    file = open(filename, "r")

    # Dividir noticias anotadas de entrenamiento y revisión en líneas para evitar
    # un uso excesivo de la memoria a la hora de entrenar el modelo.
    for sentence in file:

        # Para evitar errores de memoria, evitar entrenar con frases demasiado largas.
        if (len(sentence) < 3000):

            # Cargar frase en Spacy.
            doc = nlp(sentence)

            # Una lista de anotaciones para cada frase de cada noticia.
            annotations = []

            # Buscar matches en la noticia actual.
            matches = ent_matcher(doc)

            # Para evitar errores de memoria, evitar entrenar con listas de entidades demasiado largas.
            if (len(matches) < 40):

                # Iterar por los matches encontrados en esta noticia.
                for match_id, start, end in matches:

                    # Índices de palabra de la entidad encontrada dentro de la frase actual de la noticia.
                    span = doc[start:end]

                    # Comprobar que la entidad existe en el diccionario => es necesario ????????????????????????????
                    if (span.text in entities):

                        # Añadir índices de caracter junto con tipo de entidad a la lista de anotaciones.
                        annotations.append((span.start_char, span.end_char, entities[span.text]))

                # Añadir frase de la noticia plana junto con su lista de anotaciones a los datos de entrenamiento.
                matching_data.append((str(doc), annotations))

                result = True

    return matching_data, result

# =================================================================================================

# Usar reglas lingüísticas de Spacy para detectar entidades nombradas a partir de una lista de entidades.
def match_entities(nlp, train_dir, ne_filename):

    # Lista de entidades nombradas
    filename = str(Path(train_dir).parent) + '/' + ne_filename

    # Leer entidades nombradas del fichero.
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        entities = {rows[1]:rows[0] for rows in reader}

    # Convertir entidades a mayúsculas.
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        uppercase_entities = {rows[1].upper():rows[0] for rows in reader}
        entities.update(uppercase_entities)

    # Spacy Linguistic Features: Rule-based Matching
    # Construir matcher
    ent_matcher = PhraseMatcher(nlp.vocab)

    # Spacy únicamente permite patrones de hasta 9 tokens en el matcher.
    # Por tanto, descartamos las entidades de más de 9 palabras.
    patterns = [nlp.make_doc(entity) for entity in entities if len(nlp.make_doc(entity)) < 10]

    ent_matcher.add('EntityList', None, *patterns)

    return ent_matcher, entities

# =================================================================================================

def build_training_data(nlp, train_dir, revision_rate, ent_matcher, entities):

    # Datos de entrenamiento generados: noticias con anotaciones detectadas por reglas lingüísticas.
    matching_data = []

    # Datos de revision: noticias con entidades ya conocidas por el modelo pre-entrenado.
    revision_data = []

    num_rev_examples = round(len([x for x in os.listdir(train_dir) if x.endswith('.txt')]) * revision_rate)
    count_rev = 0

    # Recorrer ficheros txt de noticias planas.
    for file in os.listdir(train_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):

            # Cargar noticia plana de entrenamiento.
            file = train_dir + "/" + file

            if (count_rev < num_rev_examples):

                revision_data, result = revise_data(nlp, file, revision_data)
                count_rev += 1

            else:

                matching_data, result = generate_data(nlp, ent_matcher, entities, file, matching_data)

    # Unir los datos de entrenamiento y revisión.
    train_data = matching_data + revision_data

    print('Num matching examples:' + str(len(matching_data)))

    print('Num revision examples:' + str(len(revision_data)))

    print('Num total examples:' + str(len(train_data)))

    return train_data

# =================================================================================================

def update_model(nlp, train_dir, train_data, epoch, dropout, revision_rate):

        # Obtener optimizador para actualizar el modelo
    nlp.vocab.vectors.name = 'spacy_pretrained_vectors' # evita warning: unnamed vectors *******************************
    optimizer = nlp.begin_training()

    #batch_size = compounding(1, 10, 1.5)

    # Entrenar durante las épocas especificadas
    for itn in range(epoch):

        print('Epoch:', str(itn))

        # Hacer un shuffle para que el modelo no generalice basándose en el orden de las noticias.
        random.shuffle(train_data)

        # Dividir ejemplos en batches
        batches = minibatch(train_data, size=1)
        for batch in batches:

            raw_text, entity_offsets = zip(*batch)

            print('epoch:', str(itn), ' text_len:', str(len(raw_text[0])))

            # Convertir noticia de texto plano en tipo Spacy Doc.
            doc = nlp.make_doc(raw_text[0])

            print('epoch:', str(itn), ' ent_len:', str(len(entity_offsets[0])))

            # Codificar las anotaciones en formato de entrenamiento de Spacy.
            gold = GoldParse(doc, entities=entity_offsets[0])

            print('epoch:', str(itn), ' updating Model...')

            # Actualizar modelo con una tasa de dropout especificado para evitar que el modelo memorice ejemplos.
            nlp.update([doc], [gold], drop=dropout, sgd=optimizer)

            print('epoch:', str(itn), ' updated.')


    print('saving Model...')
    # Guardar modelo entrenado en disco.
    data_size = train_dir.split('/')[-1].rstrip('-data')
    model_path = str(Path(train_dir).parent.parent) + '/models/model' + \
                '_' + data_size + '_e' + str(epoch) + '_d' + str(dropout) + '_r' + str(revision_rate)
    nlp.to_disk(model_path)

    # Comprobar si estamos ejecutando en la nube para descargar el modelo a local.
    try:
        import google.colab
        in_colab = True
    except:
        in_colab = False

    if (in_colab == True):

        # Zipear modelo, descargar y borrar del cloud.
        import subprocess
        subprocess.call(['zip', '-r', model_path + '.zip', model_path])
        files.download(model_path + '.zip')
        subprocess.call(['rm', model_path + '.zip'])
        subprocess.call(['rm', '-rf', model_path])

    return nlp

# =================================================================================================

# Entrenar modelo de Spacy según los parámetros de entrada.
def train_model(nlp, train_dir, ne_filename, epoch, dropout, revision_rate):

    # Construir Matcher de entidades nombradas a partir de una lista de entrada.
    ent_matcher, entities = match_entities(nlp, train_dir, ne_filename)

    # Construir datos de entrenamiento (por revisión y por generación mediante reglas).
    train_data = build_training_data(nlp, train_dir, revision_rate, ent_matcher, entities)

    # Actualizar modelo de Spacy.
    trained_model = update_model(nlp, train_dir, train_data, epoch, dropout, revision_rate)

    # Devolver modelo entrenado.
    return trained_model

# =================================================================================================
