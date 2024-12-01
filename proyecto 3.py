import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importación de datos
data = pd.read_csv("C:/Users/Nicolay Barrera/OneDrive - Universidad de los andes/Documentos/Nicolay/Universidad/IIND/analitica computacional/proyectos/proyecto 3/db final saber11.csv")
print(data.head())

# Limpieza de datos

# Tamaño de la base de datos
print(data.shape)
# Información de la base de datos
print(data.info())

# Eliminación de variables que no son relevantes
data.drop(["estu_tipodocumento","estu_consecutivo","cole_cod_dane_establecimiento",
           "cole_nombre_sede","cole_sede_principal","estu_fechanacimiento"],
            axis=1, inplace=True)

# Verificación de la información de la base de datos
print(data.info())
# Valores vacíos
print(data.isna().sum())

# Se eliminan las filas donde el puntaje de inglés es Nan
delete_rows = data[data["punt_ingles"].isna()].index
data.drop(delete_rows, axis=0, inplace=True)
data.reset_index(drop=True, inplace=True)

# Tamaño de la base de datos
print(data.shape)

# Verificación de valores vacíos
print(data.isna().sum())

# Se unifican los periodos por año
data["periodo"] = data["periodo"].astype(str).str[:4]
data["periodo"] = data["periodo"].astype(int)
print(data.head())