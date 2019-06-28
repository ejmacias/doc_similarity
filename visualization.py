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
# Fase 3: Visualización

# =================================================================================================

from collections import Counter
import pandas as pd

from sklearn.manifold import MDS
import networkx as nx
import matplotlib.pyplot as plt
import community as community_detection

# =================================================================================================

# Calcula similitud "coseno" entre los vectores de entidades de dos documentos.
def get_similarity(news1, news2, dist_matrix, nlp, old_dist_matrix):

    # Si es la misma noticia, la similitud es máxima.
    if (news1[0] == news2[0]):

        return 1.0

    # Si una de las noticias no tiene entidades, devolver similitud 0.
    elif (len(news1[1]) == 0 or len(news2[1]) == 0):

        return 0.0

    # Si la similitud entre estas 2 noticias se ha calculado ya, reusar.
    elif (dist_matrix.index.isin([news2[0]]).any() == True):

        return dist_matrix.at[news2[0], news1[0]]

    # Si la distancia ya era conocida en la anterior matriz de distancias, reusar.
    elif (news1[0] in old_dist_matrix.columns and news2[0] in old_dist_matrix.columns):

        return old_dist_matrix[news1[0]][news2[0]]

    # Calcular la similitud entre los 2 vectores de entidades.
    else:

        doc1 = nlp(' '.join(news1[1]))
        doc2 = nlp(' '.join(news2[1]))
        sim = doc1.similarity(doc2)

        # doc.similarity devuelve la similitud. La distancia se calculará después.
        return sim

# =================================================================================================

# Calcula la matriz de distancias entre pares de documentos.
def find_doc_distances(entities_by_news, nlp, old_dist_matrix):

    # Matriz de distancias entre pares de documentos
    dist_matrix = pd.DataFrame()

    # Lista de aristas entre documentos (solo si la similitud es > 0.5).
    # Las aristas (aunque no se pinten) son necesarias para el algoritmo de clustering.
    edges = []

    for idx, news1 in enumerate(entities_by_news.items()):

        for news2 in entities_by_news.items():

            sim = get_similarity(news1, news2, dist_matrix, nlp, old_dist_matrix)
            dist_matrix.at[news1[0], news2[0]] = sim

            # No crear arista si el nodo es el mismo o la similitud es nula
            if (sim > 0 and news1[0] != news2[0]):
                edges.append((news1[0], news2[0], sim*100))

    # La matriz tiene hasta ahora una medida de similitud. Para calcular la distancia entre
    # pares es necesario restar la similitud a 1. Documentos idénticos tendrán distancia cero.
    return (1 - dist_matrix), edges

# =================================================================================================

def make_graph(dist_matrix, edges, threshold):

    # Convertir únicamente 2 componentes para pintar un grafo de 2 dimensiones. 'Precomputed' porque
    # pasamos una matriz de distancias. Especificar semilla de aleatoriedad para reproducir el grafo.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    # Calcular la posición de los nodos (noticias) dentro del grafo, según las distancias entre pares.
    pos = mds.fit_transform(dist_matrix)

    node_pos = dict(zip(list(dist_matrix), pos))

    # Crear el grafo con los nodos (noticias) y aristas entre los más cercanos.
    G = nx.Graph()
    G.add_nodes_from(node_pos)
    G.add_weighted_edges_from(edges)

    # Buscar comunidades de noticias según el peso de las aristas entre ellas.
    partition = community_detection.best_partition(G, weight='weight')

    # Filtrar por tamaño de comunidades, para representar solamente las que tengan un tamaño por encima
    # del umbral (% mínimo de noticias sobre el total).
    nodes_by_community = Counter(partition.values()).most_common()

    large_communities = [community[0] for community in nodes_by_community if
                       community[1] >= (len(G.nodes) * threshold)]

    return G, node_pos, partition, large_communities

# =================================================================================================

# Devuelve las noticias por comunidad.
def find_news_by_community(partition):

    # Invierte el diccionario partition para que la clave sea la comunidad,
    # y el valor, la lista de noticias asociadas.
    news_by_community = {}

    # Iterar por las comunidades.
    for key, value in partition.items():

        # Añadir la noticia a la lista de la comunidad.
        if value in news_by_community:
            news_by_community[value].append(key)
        else:
            news_by_community[value]=[key]

    return news_by_community

# =================================================================================================

# Calcula la frecuencia de entidades dentro de cada comunidad.
# Nota: para evitar los casos donde una entidad se repita con mucha frecuencia en una noticia y con
# poca o ninguna frecuencia en el resto de noticias de la comunidad, se contabilizará sólamente una
# vez por noticia i.e. la frecuencia indica el nº de noticias de la comunidad donde aparece cada entidad.
def find_entity_frequency(news_by_community, entities_by_news):

    # Diccionario de entidad y frecuencia de cada comunidad.
    entities_by_community = {}

    # Iterar por las comunidades.
    for key, value in news_by_community.items():

        # Un diccionario de entidades por cada comunidad.
        entity_freq = {}

        # Iterar por las noticias de la comunidad.
        for news in value:

            # Iterar por el conjunto único de entidades de la noticia.
            for entity in set(entities_by_news[news]):

                # Incrementar frecuencia de esta entidad.
                entity_freq[entity] = entity_freq.get(entity, 0) + 1

        entities_by_community[key] = [x[0] for x in Counter(entity_freq).most_common(5)]

    # Nos quedamos solo con las entidades diferenciadoras que no existen en el resto de comunidades.
    unique_entities_by_community = {x[0]:x[1] for x in entities_by_community.items()}

    for community1 in entities_by_community.keys():
        for community2 in entities_by_community.keys():
            if (community1 != community2):
                unique_entities_by_community[community1] = list(set(unique_entities_by_community[community1]) - set(entities_by_community[community2]))

    print('entities_by_community:', entities_by_community)
    return unique_entities_by_community

# =================================================================================================

# Crear y pintar grafo de noticias con las comunidades y sus entidades más significativas.
def plot_graph(dist_matrix, edges, threshold, entities_by_news, tfm_dir):

    # Crear grafo y detectar comunidades.
    G, node_pos, partition, large_communities = make_graph(dist_matrix, edges, threshold)

    # Obtener las noticias por comunidad.
    news_by_community = find_news_by_community(partition)

    # Obtener las frecuencias de entidades por comunidad.
    entities_by_community = find_entity_frequency(news_by_community, entities_by_news)

    # Crear canvas y eje.
    fig, ax = plt.subplots(1, figsize=(12,12))

    # Mostrar únicamente las comunidades grandes.
    for idx, cluster in enumerate(large_communities):

        # Al pintar grafo, incluir una lista de las entidades más significativas de cada comunidad.
        nx.draw_networkx_nodes(G, node_pos, nodelist=news_by_community[cluster],
                               node_size=100, ax=ax, node_color=[plt.cm.Set1(idx)],
                               label=str([x for x in entities_by_community[cluster]])[1:-1].replace("'",""))

    # Fijar parámetros del gráfico.
    ax.legend(scatterpoints=1, loc=2, bbox_to_anchor=(1,1))
    plt.title('Comunidades de Noticias')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(tfm_dir + '/img/news_graph.png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    print('Nodes:', len(G.nodes))
    print('Num Communities:', len(set(partition.values())))
    print('Nodes per community:', Counter(partition.values()).most_common())
    #print('Modularity:', community_detection.modularity(partition, G, weight='weight'))

# =================================================================================================
