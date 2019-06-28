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
# Fase 1: Evaluación del modelo

# =================================================================================================

import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os

# =================================================================================================

# Cuenta el número de entidades anotadas en Brat.
# Nota: la librería BratReader genera tantas anotaciones como palabras componen una entidad.
# Es por ello que existen repeticiones y hay que ignorarlas en esta función.
def count_brat_annotations(gs_annotations):
    counter = 0
    for idx, ann in enumerate(gs_annotations):
        if (idx == 0) or (gs_annotations[idx].realspan[0] != gs_annotations[idx-1].realspan[0]):
            counter += 1
    return counter

# =================================================================================================

# Tranforma etiquetas Brat en el formato de Spacy.
def brat_to_spacy(label):
    if (label == "Organization"):
        return "ORG"
    elif (label == "GPE"):
        return "LOC"
    elif (label == "Person"):
        return "PER"
    else:
        return label

# =================================================================================================

# Tranforma etiquetas Brat en numérico 0..3.
def brat_to_num(label):
    if (label == "Organization"):
        return 0
    elif (label == "GPE"):
        return 1
    elif (label == "Person"):
        return 2
    else:
        return 3

# =================================================================================================

# Tranforma etiquetas Spacy en numérico 0..3.
def spacy_to_num(label):
    if (label == "ORG"):
        return 0
    elif (label == "LOC"):
        return 1
    elif (label == "PER"):
        return 2
    else:
        return 3

# =================================================================================================

# Añade las entidades anotadas en Brat y no detectadas por Spacy entre 2 posiciones.
def process_skipped_entities(brat, doc_key, start_offs, end_offs, idx_gs, num_fn, ann_test, ann_pred):

    # Cargar anotaciones en Brat para este documento
    gs_doc = brat.documents[doc_key]

    # Verificar que la lista de anotaciones de esta noticia no esté vacía.
    if (len(gs_doc.annotations) > 0) and (idx_gs < len(gs_doc.annotations)):

        # Buscar entidades manualmente anotadas entre start_offs y end_offs
        ann = gs_doc.annotations[idx_gs]

        while ann.realspan[0] < end_offs:

            # Ignorar repeticiones de anotaciones en BratReader
            if (idx_gs == 0) or (ann.realspan[0] != gs_doc.annotations[idx_gs-1].realspan[0]):

                # Actualizar vectores
                ann_test.append(brat_to_num(list(ann.labels.items())[0][0])) # Gold Standard vector
                ann_pred.append(spacy_to_num("NONE")) # Spacy vector

                # False Negative
                num_fn += 1

            # Incrementar índice de anotaciones en el Gold Standard para procesar la siguiente
            idx_gs += 1

            try:

                # Obtener siguiente anotación.
                ann = gs_doc.annotations[idx_gs]

            except:

                # Salir del bucle ya que se ha alcanzado el final de la lista de anotaciones.
                break

    return idx_gs, num_fn, ann_test, ann_pred

# =================================================================================================

# Función que calcula el F1-Score, Precision y Recall.
def calc_score(tp, fp, fn):

    # Calcular precisión: TP / (TP + FP)
    # TP + FP = nº entidades detectadas por Spacy
    precision = tp / (tp + fp)

    # Calcular exhaustividad (recall): TP / (TP + FN)
    # TP + FN = nº entidades anotadas manualmente en el Gold Standard
    recall = tp / (tp + fn)

    # Calcular F1-Score
    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score, precision, recall

# =================================================================================================

# Función que construye los vectores de TP, FP y FN
def eval_ner(directory, nlp, brat):

    # Contador de TruePositive, FalseNegative y FalsePositive
    num_tp = 0
    num_fp = 0
    num_fn = 0

    # Vector de entidades anotadas en el Gold Standard.
    ann_test = []

    # Vector de entidades detectadas por Spacy.
    ann_pred = []

    # Recorrer ficheros txt de noticias planas.
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):

            # Indice para recorrer las entidades anotadas en Brat. Inicializar en cada noticia.
            idx_gs = 0

            # Cargar anotaciones de Brat de la noticia actual.
            filename = filename.rstrip('.txt')
            gs_doc = brat.documents[filename]

            # Cargar noticia plana en Spacy.
            file = directory + "/" + gs_doc.key + ".txt"
            nlp_doc = nlp(open(file, "r").read())

            idx_nlp = 0

            # Recorrer todas las entidades detectadas por Spacy dentro de la noticia actual.
            for idx_nlp, nlp_ent in enumerate(nlp_doc.ents):

                # Ignorar las etiquetas Misc.
                if (nlp_ent.label_ != "MISC"):

                    # Ver si existen entidades anotadas en GS no detectadas por Spacy desde la última posición.
                    if (idx_nlp > 0):

                        idx_gs, num_fn, ann_test, ann_pred = process_skipped_entities(brat, filename, \
                                                             nlp_doc.ents[idx_nlp-1].end_char, \
                                                             nlp_ent.start_char, idx_gs, num_fn, \
                                                             ann_test, ann_pred)

                    else:

                        # Ver si no se ha detectado alguna entidad antes de la 1ª entidad detectada por Spacy.
                        idx_gs, num_fn, ann_test, ann_pred = process_skipped_entities(brat, filename, \
                                                             0, nlp_ent.start_char, idx_gs, num_fn, \
                                                             ann_test, ann_pred)

                    # Flag para marcar si una entidad detectada por Spacy está anotada en el Gold Standard.
                    ann_found = False

                    # Buscar entidad detectada por Spacy dentro de todas las anotaciones manuales de esta noticia.
                    for ann in gs_doc.annotations:

                        # Usamos "menor o igual" y "mayor o igual" para dar por válido los casos en que se
                        # detecta entidad correcta pero más larga a la anotada manualmente p.e.
                        # "el Real Madrid" vs "Real Madrid".
                        if ((nlp_ent.start_char <= ann.realspan[0]) and (nlp_ent.end_char >= ann.realspan[1])):

                            # Actualizar vectores
                            ann_test.append(brat_to_num(list(ann.labels.items())[0][0])) # GS vector
                            ann_pred.append(spacy_to_num(nlp_ent.label_)) # Spacy vector

                            # Incrementar índice de anotaciones en el Gold Standard.
                            idx_gs += 1

                            # Anotación encontrada en la misma posición.
                            ann_found = True

                            # Chequear si las etiquetas de entidad coinciden (teniendo en cuenta el formato).
                            if (nlp_ent.label_ == brat_to_spacy(list(ann.labels.items())[0][0])):

                                # True Positive
                                num_tp += 1

                            else:

                                # La entidad nombrada se ha detectado pero con distinta etiqueta.
                                # False Positive y False Negative
                                num_fp += 1
                                num_fn += 1

                            # Saltar a la siguiente entidad detectada por Spacy.
                            break

                    # Chequear si la entidad detectada por Spacy NO está anotada en el Gold Standard.
                    if (ann_found == False):

                        # Actualizar vectores
                        ann_test.append(brat_to_num("None")) # GS vector
                        ann_pred.append(spacy_to_num(nlp_ent.label_)) # Spacy vector

                        # False Positive
                        num_fp += 1

            # Ver si existen entidades anotadas en GS no detectadas por Spacy hasta el final de la noticia.
            if (idx_nlp > 0):

                idx_gs, num_fn, ann_test, ann_pred = process_skipped_entities(brat, filename, nlp_ent.end_char, \
                                                                              math.inf, idx_gs, num_fn, \
                                                                              ann_test, ann_pred)

            else:

                # Ver si existe alguna entidad anotada aunque Spacy no haya detectado ninguna.
                idx_gs, num_fn, ann_test, ann_pred = process_skipped_entities(brat, filename, 0, math.inf, \
                                                                              idx_gs, num_fn, \
                                                                              ann_test, ann_pred)

    # Devolver F1-Score, Precision, Recall y dos vectores (del GS y detecciones de Spacy)
    f1_score, precision, recall = calc_score(num_tp, num_fp, num_fn)
    return ann_test, ann_pred, f1_score, precision, recall

# =================================================================================================

# Muestra una matriz de confusión
def plot_confusion_matrix(cm, classes, ylabel, xlabel,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    # Calcular normalización 0..1
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Fijar número de decimales para la matriz normalizada
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Título de ejes
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()

# =================================================================================================

def show_conf_matrix(cnf_matrix, class_names, tfm_dir, model_name):

    # Fijar precisión
    np.set_printoptions(precision=2)

    # Mostrar matriz sin normalizar
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          ylabel='Gold Standard', xlabel='Predicciones',
                          title='Matriz de confusión sin normalizar')

    # Mostrar matriz normalizada
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          ylabel='Gold Standard', xlabel='Predicciones',
                          normalize=True, title='Matriz de confusión normalizada')

    plt.savefig(tfm_dir + '/img/cnf_matrix_' + model_name + '.png')
    #plt.show()

# =================================================================================================
