English article in Medium [here](https://ejmacias.medium.com/clustering-news-articles-based-on-named-entities-306a23d368e1).


Título: Descubriendo similitud entre documentos a partir de compartición de entidades nombradas.

Este Trabajo Fin de Master (TFM) es parte del Máster en Big Data y Data Science de la Universidad Autónoma de Madrid.

El objetivo del TFM es la extracción de información relevante de un conjunto de noticias que permita una adecuada visualización y categorización de las mismas.

<img width="1161" alt="Screenshot 2021-05-01 at 23 58 37" src="https://user-images.githubusercontent.com/40733745/116795986-288ebb80-aad9-11eb-82bd-948079b6bd95.png">


Estructura de archivos:
- /scripts
  * json2txt.sh: Extract the text from a JSON article and convert it into txt format.
  * launch_ner.sh: Launch all the required processes for the system.
  * random_news.sh: Pick a subset of random news articles from a directory.
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
