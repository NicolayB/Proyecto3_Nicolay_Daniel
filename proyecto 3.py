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

# Análisis de datos

# Estadísticas descriptivas
ed = data.drop("periodo", axis=1).describe()
print(ed)

# Creación de matriz de variables explicativas y variable de interés
X = data.drop("punt_global", axis=1)
Y = data["punt_global"]

# Lista de variables
cols = data.columns

# Box plots de puntaje global promedio de distintas variables
cols_interes = [1,2,3,4,6,7,9]
nom_cols = ["área","bilingüismo","calendario","caracter","genero del colegio","jornada","naturaleza"]
titulo_cols = ["Área","Bilingüe","Calendario","Caracter","Genero colegio","Jornada","Naturaleza"]
cont = 0
for i in cols_interes:
    variable = data.groupby([cols[i],"periodo"])["punt_global"].mean().reset_index()
    plt.figure(figsize=(15,8))
    sns.boxplot(variable, x=cols[i], y="punt_global", palette="Blues_d", width=0.9)
    plt.title(f"Puntaje global promedio {nom_cols[cont]}")
    plt.xlabel(titulo_cols[cont])
    plt.ylabel("Puntaje global promedio")
    plt.show()
    cont += 1

# Box plots de puntaje global promedio de distintas variables por periodo
cols_interes = [1,2,3,4,6,7,9]
nom_cols = ["área","bilingüismo","calendario","caracter","genero del colegio","jornada","naturaleza"]
titulo_cols = ["Área","Bilingüe","Calendario","Caracter","Genero colegio","Jornada","Naturaleza"]
cont = 0
for i in cols_interes:
    if i == 3:
        orden = ["A","B","OTRO"]
    else:
        orden = None
    plt.figure(figsize=(15,8))
    sns.boxplot(data, x="periodo", y="punt_global", hue=cols[i], palette="Blues_d", width=0.9, hue_order=orden)
    plt.title(f"Puntaje global promedio por periodo y {nom_cols[cont]}")
    plt.xlabel("Periodo")
    plt.ylabel("Puntaje global promedio")
    plt.legend(title=titulo_cols[cont])
    plt.show()
    cont += 1

# Box plots de puntaje global promedio por departamentos y municipio de presentación y residencia
cols_interes = ["estu_depto_presentacion","estu_depto_reside","estu_mcpio_presentacion","estu_mcpio_reside"]
nom_cols = ["departamento de presentación","departamento de residencia","municipio de presentación","municipio de residencia"]
titulo_cols = ["Departamento de presentación","Departamento de residencia","Municipio de presentación","Municipio de residencia"]
cont = 0
for i in cols_interes:
    if i == "estu_mcpio_presentacion" or i == "estu_mcpio_reside":
        mcpios = data[i].value_counts()
        mcpios = mcpios[mcpios>100].index
        mcpios_df = data[data[i].isin(mcpios)]
        plt.figure(figsize=(15,8))
        sns.boxplot(mcpios_df, x=i, y="punt_global", width=0.9)
    else:
        plt.figure(figsize=(15,8))
        sns.boxplot(data, x=i, y="punt_global", width=0.9)
    plt.title(f"Puntaje global promedio por {nom_cols[cont]}")
    plt.xlabel(titulo_cols[cont])
    plt.ylabel("Puntaje global promedio")
    plt.xticks(rotation=90)
    plt.show()
    cont += 1

# Box plots de información estudiante
cols_interes = [14,19]
nom_cols = ["genero del estudiante","privación de libertad"]
titulo_cols = ["Genero del estudiante","Privación de libertad"]
cont = 0
for i in cols_interes:
    plt.figure(figsize=(15,8))
    sns.boxplot(data, x="periodo", y="punt_global", hue=cols[i], palette="Blues_d", width=0.9)
    plt.title(f"Puntaje global promedio por perdiodo y {nom_cols[cont]}")
    plt.xlabel("Periodo")
    plt.ylabel("Puntaje global promedio")
    plt.legend(title=titulo_cols[cont])
    plt.show()
    cont += 1

# Box plot de puntaje global promedio por estratos
orden = ["Estrato 1","Estrato 2","Estrato 3","Estrato 4","Estrato 5","Estrato 6","Sin Estrato"]
plt.figure(figsize=(15,8))
sns.boxplot(data, x="fami_estratovivienda", y="punt_global", palette="Blues_d", width=0.9, order=orden)
plt.title(f"Puntaje global promedio por estrato")
plt.xlabel("Estrato")
plt.ylabel("Puntaje global promedio")
plt.show()

# Box plots de información familiar
cols_interes = [25,26,27,28]
nom_cols = ["automovil","computador","internet","lavadora"]
titulo_cols = ["Automovil","Computador","Internet","Lavadora"]
cont = 0
for i in cols_interes:
    plt.figure(figsize=(15,8))
    sns.boxplot(data, x="periodo", y="punt_global", hue=cols[i], palette="Blues_d", width=0.9, hue_order=["No","Si"])
    plt.title(f"Puntaje global promedio por periodo y tiene {nom_cols[cont]}")
    plt.xlabel("Periodo")
    plt.ylabel("Puntaje global promedio")
    plt.legend(title=titulo_cols[cont])
    plt.show()
    cont += 1

# Codificación de algunas variables categóricas
data["cole_area_ubicacion"] = data["cole_area_ubicacion"].map({"URBANO":1,"RURAL":0})
data["cole_bilingue"] = data["cole_bilingue"].map({"N":0,"S":1})
data["cole_calendario"] = data["cole_calendario"].map({"OTRO":0,"A":1,"B":2})
data["cole_caracter"] = data["cole_caracter"].map({"NO APLICA":0,"ACADÉMICO":1,"TÉCNICO/ACADÉMICO":2,"TÉCNICO":3})
data["cole_genero"] = data["cole_genero"].map({"FEMENINO":0,"MASCULINO":1,"MIXTO":2})
data["cole_jornada"] = data["cole_jornada"].map({"UNICA":0,"MAÑANA":1,"TARDE":2,"NOCHE":3,"SABATINA":4,"COMPLETA":5})
data["cole_naturaleza"] = data["cole_naturaleza"].map({"OFICIAL":0,"NO OFICIAL":1})
data["estu_estadoinvestigacion"] = data["estu_estadoinvestigacion"].map({"PUBLICAR":0,"VALIDEZ OFICINA JURÍDICA":1,"NO SE COMPROBO IDENTIDAD DEL EXAMINADO":2,"PRESENTE CON LECTURA TARDIA":3})
data["estu_genero"] = data["estu_genero"].map({"F":0,"M":1})
data["estu_privado_libertad"] = data["estu_privado_libertad"].map({"N":0,"S":1})
data["fami_estratovivienda"] = data["fami_estratovivienda"].map({"Sin Estrato":0,"Estrato 1":1,"Estrato 2":2,"Estrato 3":3,"Estrato 4":4,"Estrato 5":5,"Estrato 6":6})
data["fami_tieneautomovil"] = data["fami_tieneautomovil"].map({"No":0,"Si":1})
data["fami_tienecomputador"] = data["fami_tienecomputador"].map({"No":0,"Si":1})
data["fami_tieneinternet"] = data["fami_tieneinternet"].map({"No":0,"Si":1})
data["fami_tienelavadora"] = data["fami_tienelavadora"].map({"No":0,"Si":1})

# DataFrame con únicamente variables continuas luego de la codificación
df_cols = data.select_dtypes(exclude = "object").columns 
df = data[df_cols]
print(df.head())

# Correlaciones entre variables
corr = df.corr()
corr = corr.round(2)
plt.figure(figsize=(15,8))
sns.heatmap(corr, cmap="Blues", annot=True)
plt.show()

# Mapa de calor de la correlación entre algunas variables y la variable de interés
X_df = df.drop("punt_global", axis=1)
correlacion = pd.DataFrame(X_df.corrwith(Y))
plt.figure(figsize=(15,8))
sns.heatmap(correlacion, cmap="Blues", annot=True)
plt.title("Correlación de variables con variable de interés")
plt.ylabel("Variable")
plt.show()

# Mapa de calor de la correlación entre departamento de residencia y la variable de interés
df2 = data[["estu_depto_reside","punt_global"]]
cat = df2.select_dtypes(include=["object"]).columns
df2 = pd.get_dummies(df2, columns=cat)
X_df2 = df2.drop("punt_global", axis=1)
corr_dep = pd.DataFrame(X_df2.corrwith(Y))
plt.figure(figsize=(15,8))
sns.heatmap(corr_dep, cmap="Blues", annot=True)
plt.title("Correlación de variables con variable de interés")
plt.ylabel("Variable")
plt.show()

# Gráficos de correlación con regresión
"""cols_reg = correlacion[(correlacion[0]>0.1) & (correlacion[0]<0.8) | (correlacion[0]<-0.1)].index
df2 = data[cols_reg]
df2["punt_global"] = data["punt_global"]
for i in range(0,9,2):
    sns.pairplot(df2, x_vars=df2[df2.columns[[i,i+1]]], y_vars=["punt_global"], height=7, kind="reg", plot_kws={"line_kws": {"color": "red"}})
plt.show()"""