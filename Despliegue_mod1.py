import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import load
from tensorflow.keras.losses import MeanSquaredError
import requests

# Cargar el modelo entrenado
url_model = "https://raw.githubusercontent.com/NicolayB/Proyecto3_Nicolay_Daniel/main/modelo_colegios.h5"

response_model = requests.get(url_model)
with open("modelo_colegios.h5", "wb") as file:
    file.write(response_model.content)

model = load_model('modelo_colegios.h5', compile=False)
model.compile(loss=MeanSquaredError()) 

# URL del preprocesador en GitHub
url_preprocessor = "https://raw.githubusercontent.com/NicolayB/Proyecto3_Nicolay_Daniel/main/pipeline_preprocessor.joblib"

# Descargar el archivo
response = requests.get(url_preprocessor)
if response.status_code == 200:
    with open("pipeline_preprocessor.joblib", "wb") as file:
        file.write(response.content)
    print("Preprocesador descargado exitosamente.")
else:
    raise Exception("Error al descargar el preprocesador. Verifica la URL.")

# Cargar el preprocesador descargado
preprocessor = load("pipeline_preprocessor.joblib")

# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Opciones para los campos categóricos
options = {
    'cole_area_ubicacion': ['URBANO', 'RURAL'],
    'cole_bilingue': ['S', 'N'],
    'cole_calendario': ['A', 'B', 'OTRO'],
    'cole_caracter': ['ACADÉMICO', 'TÉCNICO' ,'TÉCNICO/ACADÉMICO'],
    'cole_genero': ['MIXTO', 'FEMENINO', 'MASCULINO'],
    'cole_jornada': ['MAÑANA', 'TARDE', 'NOCHE', 'COMPLETA', 'UNICA', 'SABATINA'],
    'cole_naturaleza': ['OFICIAL', 'NO OFICIAL']
}

# Layout de la aplicación
app.layout = dbc.Container([
    html.H1("Predicción del Puntaje Global de Pruebas Saber", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Área de Ubicación"),
            dcc.Dropdown(
                id='cole_area_ubicacion',
                options=[{'label': opt, 'value': opt} for opt in options['cole_area_ubicacion']],
                value='URBANO'
            ),
        ], md=4),
        dbc.Col([
            html.Label("¿Es bilingüe?"),
            dcc.Dropdown(
                id='cole_bilingue',
                options=[{'label': opt, 'value': opt} for opt in options['cole_bilingue']],
                value='N'
            ),
        ], md=4),
        dbc.Col([
            html.Label("Calendario"),
            dcc.Dropdown(
                id='cole_calendario',
                options=[{'label': opt, 'value': opt} for opt in options['cole_calendario']],
                value='A'
            ),
        ], md=4),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Carácter del Colegio"),
            dcc.Dropdown(
                id='cole_caracter',
                options=[{'label': opt, 'value': opt} for opt in options['cole_caracter']],
                value='ACADÉMICO'
            ),
        ], md=4),
        dbc.Col([
            html.Label("Género"),
            dcc.Dropdown(
                id='cole_genero',
                options=[{'label': opt, 'value': opt} for opt in options['cole_genero']],
                value='MIXTO'
            ),
        ], md=4),
        dbc.Col([
            html.Label("Jornada"),
            dcc.Dropdown(
                id='cole_jornada',
                options=[{'label': opt, 'value': opt} for opt in options['cole_jornada']],
                value='COMPLETA'
            ),
        ], md=4),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Naturaleza"),
            dcc.Dropdown(
                id='cole_naturaleza',
                options=[{'label': opt, 'value': opt} for opt in options['cole_naturaleza']],
                value='OFICIAL'
            ),
        ], md=4),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Button("Predecir Puntaje", id="predict-button", color="primary", className="mt-4"),
        ], className="text-center"),
    ]),

    html.Div(id="prediction-result", className="mt-4 text-center")
])

# Callback para predecir el puntaje
@app.callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    [
        State("cole_area_ubicacion", "value"),
        State("cole_bilingue", "value"),
        State("cole_calendario", "value"),
        State("cole_caracter", "value"),
        State("cole_genero", "value"),
        State("cole_jornada", "value"),
        State("cole_naturaleza", "value")
    ]
)
def predict_puntaje(n_clicks, area, bilingue, calendario, caracter, genero, jornada, naturaleza):
    if n_clicks is None:
        return "Ingrese los valores y haga clic en Predecir."

    # Crear un dataframe con los datos de entrada
    input_data = pd.DataFrame([{
        'cole_area_ubicacion': area,
        'cole_bilingue': bilingue,
        'cole_calendario': calendario,
        'cole_caracter': caracter,
        'cole_genero': genero,
        'cole_jornada': jornada,
        'cole_naturaleza': naturaleza
    }])

    # Preprocesar los datos
    input_transformed = preprocessor.transform(input_data)

    # Predecir el puntaje
    prediction = model.predict(input_transformed)
    predicted_score = np.round(prediction[0][0], 2)

    # Comparar con el promedio nacional
    average_score = 274.896362
    if predicted_score > average_score:
        comparison_message = "El puntaje está por encima del promedio nacional."
    elif predicted_score < average_score:
        comparison_message = "El puntaje está por debajo del promedio nacional."
    else:
        comparison_message = "El puntaje es igual al promedio nacional."

    return f"El puntaje global estimado es: {predicted_score}. {comparison_message}"


# Ejecutar la aplicación
if __name__ == "__main__":
    app.run_server(debug=False, host='0.0.0.0', port=8080)

