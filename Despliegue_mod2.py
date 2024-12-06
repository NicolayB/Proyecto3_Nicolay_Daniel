import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.losses import MeanSquaredError
import requests
from joblib import load
import pandas as pd

url_model = "https://raw.githubusercontent.com/NicolayB/Proyecto3_Nicolay_Daniel/main/modelo_estudiantes.h5"

response_model = requests.get(url_model)
with open("modelo_estudiantes.h5", "wb") as file:
    file.write(response_model.content)

model = load_model('modelo_estudiantes.h5', compile=False)
model.compile(loss=MeanSquaredError()) 

# URL del preprocesador en GitHub
url_preprocessor = "https://raw.githubusercontent.com/NicolayB/Proyecto3_Nicolay_Daniel/main/preprocessor.pkl"

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

# Opciones de selección para cada variable
options = {
    'estu_genero': ['F', 'M'],
    'fami_cuartoshogar': ['Tres', 'Dos', 'Cuatro', 'Cinco', 'Uno', 'Seis o mas', 
                          'Seis', 'Siete', 'Ocho', 'Diez o más', 'Nueve'],
    'fami_educacionmadre': [
        'Secundaria (Bachillerato) completa', 'Secundaria (Bachillerato) incompleta',
        'Educación profesional completa', 'Técnica o tecnológica completa',
        'Primaria completa', 'Primaria incompleta', 'Postgrado', 
        'Técnica o tecnológica incompleta', 'Educación profesional incompleta',
        'No sabe', 'Ninguno', 'No Aplica'
    ],
    'fami_educacionpadre': [
        'Secundaria (Bachillerato) completa', 'Secundaria (Bachillerato) incompleta',
        'Educación profesional completa', 'Primaria incompleta', 
        'Primaria completa', 'Técnica o tecnológica completa', 'No sabe', 'Postgrado',
        'Técnica o tecnológica incompleta', 'Ninguno', 'Educación profesional incompleta',
        'No Aplica'
    ],
    'fami_estratovivienda': ['Estrato 2', 'Estrato 3', 'Estrato 1', 'Estrato 4',
                             'Estrato 5', 'Estrato 6', 'Sin Estrato'],
    'fami_personashogar': [
        '3 a 4', 'Cuatro', '5 a 6', 'Cinco', 'Tres', 'Seis', '1 a 2', 'Dos', 
        '7 a 8', 'Siete', '9 o más', 'Ocho', 'Nueve', 'Doce o más', 'Diez', 
        'Una', 'Once'
    ],
    'fami_tieneautomovil': ['Si', 'No'],
    'fami_tienecomputador': ['Si', 'No'],
    'fami_tieneinternet': ['Si', 'No'],
    'fami_tienelavadora': ['Si', 'No']
}

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Diseño del tablero
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Predicción del Puntaje Global", className="text-center mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Género:"),
            dcc.Dropdown(id='estu_genero', options=[{'label': i, 'value': i} for i in options['estu_genero']],
                         placeholder="Seleccione Género")
        ], width=4),
        dbc.Col([
            html.Label("Cuartos en el hogar:"),
            dcc.Dropdown(id='fami_cuartoshogar', 
                         options=[{'label': i, 'value': i} for i in options['fami_cuartoshogar']],
                         placeholder="Seleccione Cuartos")
        ], width=4),
        dbc.Col([
            html.Label("Educación de la madre:"),
            dcc.Dropdown(id='fami_educacionmadre', 
                         options=[{'label': i, 'value': i} for i in options['fami_educacionmadre']],
                         placeholder="Seleccione Educación Madre")
        ], width=4)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Educación del padre:"),
            dcc.Dropdown(id='fami_educacionpadre', 
                         options=[{'label': i, 'value': i} for i in options['fami_educacionpadre']],
                         placeholder="Seleccione Educación Padre")
        ], width=4),
        dbc.Col([
            html.Label("Estrato de vivienda:"),
            dcc.Dropdown(id='fami_estratovivienda', 
                         options=[{'label': i, 'value': i} for i in options['fami_estratovivienda']],
                         placeholder="Seleccione Estrato")
        ], width=4),
        dbc.Col([
            html.Label("Personas en el hogar:"),
            dcc.Dropdown(id='fami_personashogar', 
                         options=[{'label': i, 'value': i} for i in options['fami_personashogar']],
                         placeholder="Seleccione Personas")
        ], width=4)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("¿Tiene automóvil?:"),
            dcc.Dropdown(id='fami_tieneautomovil', 
                         options=[{'label': i, 'value': i} for i in options['fami_tieneautomovil']],
                         placeholder="Seleccione")
        ], width=3),
        dbc.Col([
            html.Label("¿Tiene computador?:"),
            dcc.Dropdown(id='fami_tienecomputador', 
                         options=[{'label': i, 'value': i} for i in options['fami_tienecomputador']],
                         placeholder="Seleccione")
        ], width=3),
        dbc.Col([
            html.Label("¿Tiene internet?:"),
            dcc.Dropdown(id='fami_tieneinternet', 
                         options=[{'label': i, 'value': i} for i in options['fami_tieneinternet']],
                         placeholder="Seleccione")
        ], width=3),
        dbc.Col([
            html.Label("¿Tiene lavadora?:"),
            dcc.Dropdown(id='fami_tienelavadora', 
                         options=[{'label': i, 'value': i} for i in options['fami_tienelavadora']],
                         placeholder="Seleccione")
        ], width=3)
    ]),
    dbc.Row([
        dbc.Col(html.Button("Predecir", id='predict-button', n_clicks=0, className="btn btn-primary mt-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H2(id='prediction-output', className="text-center mt-4"), width=12)
    ])
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input(var, 'value') for var in options.keys()]
)
def predict_puntaje(n_clicks, *user_inputs):
    if n_clicks is None or n_clicks == 0:
        return "Ingrese los valores y haga clic en Predecir."

    # Crear un dataframe con los datos de entrada
    input_data = pd.DataFrame([{
        'estu_genero': user_inputs[0],
        'fami_cuartoshogar': user_inputs[1],
        'fami_educacionmadre': user_inputs[2],
        'fami_educacionpadre': user_inputs[3],
        'fami_estratovivienda': user_inputs[4],
        'fami_personashogar': user_inputs[5],
        'fami_tieneautomovil': user_inputs[6],
        'fami_tienecomputador': user_inputs[7],
        'fami_tieneinternet': user_inputs[8],
        'fami_tienelavadora': user_inputs[9]
    }])

    # Preprocesar los datos
    try:
        input_data_transformed = preprocessor.transform(input_data)
    except Exception as e:
        return f"Error al procesar los datos: {str(e)}"

    # Realizar la predicción
    try:
        prediction = model.predict(input_data_transformed)
        predicted_score = np.round(prediction[0][0], 2)

        # Definir el mensaje basado en el promedio nacional
        promedio_nacional = 274.896362
        if predicted_score > promedio_nacional:
            mensaje = "El puntaje predicho está por encima del promedio nacional."
        elif predicted_score < promedio_nacional:
            mensaje = "El puntaje predicho está por debajo del promedio nacional."
        else:
            mensaje = "El puntaje predicho es igual al promedio nacional."

        return f"El puntaje global estimado es: {predicted_score}. {mensaje}"
    except Exception as e:
        return f"Error al realizar la predicción: {str(e)}"



# Correr la aplicación
if __name__ == "__main__":
    app.run_server(debug=False, host='0.0.0.0', port=8080)
