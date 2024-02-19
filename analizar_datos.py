from pyspark.sql import SparkSession

# Crear una sesión de Spark
spark = SparkSession.builder.appName("AnalizarDatos").getOrCreate()

#Cargo el archivo de datos
df = spark.read.csv("datos_fut.csv", header=True, inferSchema=True)

#limpieza de datos
df = df.dropna()

#eliminamos datos duplicados 
df = df.dropDuplicates()


#Cerramos la sesión de Spark
spark.stop()