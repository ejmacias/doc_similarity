# tfm
Título: Descubriendo similitud entre documentos a partir de compartición de entidades nombradas.

Este Trabajo Fin de Master (TFM) es parte del Máster en Big Data y Data Science de la Universidad Autónoma de Madrid.

El objetivo del TFM es la extracción de información relevante de un conjunto de noticias que permite una adecuada visualización de las mismas.

Estructura de archivos:
- /scripts: 
- /src
  * \prototype1: primera fase - evaluación de los modelos de lenguaje de Spacy.
  * \prototype2: segunda fase - entrenamiento de los modelos para mejora de precisión.
  * \prototype3: tercera fase - persistencia en Elastic y visualización.
  * enriched_news_consumer.py: Consumer de la cola Kafka para nuevas noticias e indexación en Elastic Search.
  * entity_consumer.py: Consumer de la cola Kafka para entidades nombradas.
  * news_graph.py: Lanza el grafo de noticias a partir de las entidades indexadas en ES.
  * news_producer.py: Producer de la cola Kafka para nuevas noticias.
  * spark_streaming_ner.py: Procesamiento distribuido de nuevas noticias y producer de la cola Kafka para entidades nombradas.
  * visualization.py: Dibuja el grafo de noticias en distintas comunidades.
- TFM_memoria.pdf: memoria descriptiva del proyecto.
- environment.yml: Conda environment file. Lanzar los siguientes comandos para instalar los paquetes de Python:
  * conda env create -f environment.yml
  * conda activate doc-similarity
- requisitos_previos: lista de herramientas necesarias para esta aplicación.
