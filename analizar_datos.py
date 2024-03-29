from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Crear una sesión de Spark
spark = SparkSession.builder.appName("AnalizarDatos").getOrCreate()

# Configuramos el nivel de log
spark.sparkContext.setLogLevel('WARN')

#Cargo el archivo de datos
df = spark.read.csv("datos_fut.csv", sep = ";",header=True, inferSchema=True)

#limpieza de datos
df = df.dropna()

#eliminamos datos duplicados 
df = df.dropDuplicates()

#Analizamos los datos
print("Número de filas: ", df.count())
print("Número de columnas: ", len(df.columns))

#datos estadísticos 
datos_pandas = df.describe().toPandas()
print("Datos estadísticos: ")
print(datos_pandas)

#matriz de correlación de las variables numéricas
#seleccionamos las variables numéricas
columnas_num = [c for c, tipo in df.dtypes if tipo != "string"]

assembler = VectorAssembler(inputCols=columnas_num, outputCol="features")
datos_transformados = assembler.transform(df).select("features")
matriz_correlacion = Correlation.corr(datos_transformados, "features").head()
array_correlacion = matriz_correlacion[0].toArray()

#imprimimos la matriz de correlación
print("Matriz de correlación: ")
df_correlacion = pd.DataFrame(array_correlacion, columns=columnas_num, index=columnas_num)
print(df_correlacion)

#Cerramos la sesión de Spark
spark.stop()