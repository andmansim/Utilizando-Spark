from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import seaborn as sns
import matplotlib.pyplot as plt

# Crear una sesión de Spark
spark = SparkSession.builder.appName("AnalizarDatos").getOrCreate()

#Cargo el archivo de datos
df = spark.read.csv("datos_fut.csv", header=True, inferSchema=True)

#limpieza de datos
df = df.dropna()

#eliminamos datos duplicados 
df = df.dropDuplicates()

#Analizamos los datos
print("Número de filas: ", df.count())
print("Número de columnas: ", len(df.columns))
df.describe().show()  # Resumen estadístico de los datos

#matriz de correlación de las variables numéricas
assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
datos_transformados = assembler.transform(df).select("features")
matriz_correlacion = Correlation.corr(datos_transformados, "features").head()
array_correlacion = matriz_correlacion[0].toArray()

#imprimimos la matriz de correlación
print("Matriz de correlación: ")
sns.heatmap(array_correlacion, annot=True, fmt=".2f")
plt.title('Matriz de correlación de las variables numéricas')
plt.show()

#Cerramos la sesión de Spark
spark.stop()