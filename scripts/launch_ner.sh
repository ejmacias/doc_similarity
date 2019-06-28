#!/bin/bash

# Raiz del proyecto
cd ~/Documents/master/tfm

# Zookeeper
./kafka_2.12-2.2.1/bin/zookeeper-server-start.sh kafka_2.12-2.2.1/config/zookeeper.properties &
sleep 10

# Kafka-Server
./kafka_2.12-2.2.1/bin/kafka-server-start.sh kafka_2.12-2.2.1/config/server.properties &
sleep 10

# ElasticSearch
./elasticsearch-6.5.1/bin/elasticsearch & PID_LIST=$!
sleep 5

# Kibana
./kibana-6.5.1-darwin-x86_64/bin/kibana & PID_LIST+=" "$!
sleep 5

# Brat Annotation Tool
cd brat-v1.3_Crunchy_Frog
/anaconda3/envs/brat-env/bin/python standalone.py & PID_LIST+=" "$!
sleep 5
cd ..

# Kafka Consumer: enriched-news
/anaconda3/envs/doc-similarity/bin/python src/enriched_news_consumer.py & PID_LIST+=" "$!
sleep 2

# Kafka Consumer: entities
/anaconda3/envs/doc-similarity/bin/python src/entity_consumer.py & PID_LIST+=" "$!
sleep 2

# Spark NER
export PYSPARK_PYTHON=/anaconda3/envs/doc-similarity/bin/python
spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.2.1 --archives scripts/doc-similarity-env.tar.gz#environment src/spark_streaming_ner.py raw-news & PID_LIST+=" "$!
sleep 25

# Kafka Producer: raw-news
/anaconda3/envs/doc-similarity/bin/python src/news_producer.py & PID_LIST+=" "$!

echo
echo "Streaming-NER started:$PID_LIST";

# Parar todos los procesos con Ctrl+C
trap "kill -9 $PID_LIST; sleep 2; ./kafka_2.12-2.2.1/bin/kafka-server-stop.sh; sleep 5; ./kafka_2.12-2.2.1/bin/zookeeper-server-stop.sh" SIGINT

wait $PID_LIST

echo
echo "Streaming-NER stopped: $PID_LIST";
