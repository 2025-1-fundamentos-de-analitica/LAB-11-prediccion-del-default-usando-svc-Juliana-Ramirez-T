# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.


import pickle
import os
import gzip
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score 
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import f_classif,SelectKBest
from sklearn.svm import SVC
from sklearn import set_config


#Cargar los datos train y test
test = pd.read_csv("files/input/test_data.csv.zip", index_col=False, compression="zip",)
train = pd.read_csv("files/input/train_data.csv.zip", index_col=False, compression="zip",)

# Paso 1. Realice la limpieza de los datasets:

# - Renombre la columna "default payment next month" a "default".
test = test.rename(columns={'default payment next month': 'default'})
train = train.rename(columns={'default payment next month': 'default'})
# - Remueva la columna "ID".
test=test.drop(columns=['ID'])
train=train.drop(columns=['ID'])
# - Para la columna EDUCATION, valores > 4 indican niveles superiores de educación, agrupe estos valores en la categoría "others".
train = train.loc[train["MARRIAGE"] != 0]
test = test.loc[test["MARRIAGE"] != 0]
train = train.loc[train["EDUCATION"] != 0]
test = test.loc[test["EDUCATION"] != 0]
test['EDUCATION'] = test['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
train['EDUCATION'] = train['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
# - Elimine los registros con informacion no disponible.
train.dropna(inplace=True)
test.dropna(inplace=True)

# Paso 2.

# Divida los datasets en x_train, y_train, x_test, y_test.
#Train
x_train, y_train = train.drop(columns=["default"]), train["default"]
#Test
x_test, y_test = test.drop(columns=["default"]), test["default"]

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).

var_categ=["SEX","EDUCATION","MARRIAGE"]
def num_categ(df, categorical_vars):
    return [col for col in df.columns if col not in categorical_vars]
x = num_categ(x_train, var_categ)

transformador = ColumnTransformer(transformers=[('cat', OneHotEncoder(), var_categ), ('scaler',StandardScaler(with_mean=True, with_std=True),x),],
)
model_pipe=Pipeline(
    [("transf",transformador), ('pca',PCA()), ('selec_caracteristicas',SelectKBest(score_func=f_classif)),('clasificador', SVC(kernel="rbf",random_state=12345,max_iter=-1))]
)

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. 
# Use la función de precision balanceada para medir la precisión del modelo.
set_config(transform_output="pandas")
transformador.fit(x_train)
numdprep = transformador.transform(x_train).shape[1]

paramet = {
    'pca__n_components': [min(20, numdprep)],
    'feature_selection__k': [min(12, numdprep)],
    'classifier__kernel': ['rbf'],
    'classifier__gamma': [0.1],
}
estima=GridSearchCV(
    model_pipe,
    paramet,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True
    )
estima.fit(x_train, y_train)

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
models_dir = os.path.join("files", "models")
os.makedirs(models_dir, exist_ok=True)
comprimido = os.path.join(models_dir, "model.pkl.gz")
with gzip.open(comprimido, "wb") as f:
    pickle.dump(estima, f)

# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:

# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
def cmetricas(model, X_train, X_test, y_train, y_test):
    py_train = model.predict(X_train)
    py_test = model.predict(X_test)

    entrenar_metri = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, py_train, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, py_train),
        'recall': recall_score(y_train, py_train, zero_division=0),
        'f1_score': f1_score(y_train, py_train, zero_division=0)
    }
    test_metri = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, py_test, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, py_test),
        'recall': recall_score(y_test, py_test, zero_division=0),
        'f1_score': f1_score(y_test,py_test, zero_division=0)
    }
    #Guardarrr
    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, 'metrics.json')
    with open(output_dir, 'w') as f: 
        f.write(json.dumps(entrenar_metri) + '\n')
        f.write(json.dumps(test_metri) + '\n')

# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

def matrizconfu(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Convertir las matrices de confusión en formato JSON
    def nose(cm, dataset_type):
        return {
            'type': 'cm_matrix',
            'dataset': dataset_type,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }

    metricas = []
    metricas.append(nose(cm_train, 'train'))
    metricas.append(nose(cm_test, 'test'))

    output_dir = os.path.join('files/output', 'metrics.json')
    with open(output_dir, 'a', encoding='utf-8') as archivo:
        for metrica in metricas:
            archivo.write(json.dumps(metrica) + '\n')




