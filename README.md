# Actividad 5: clasificación, regresión y seguimiento con MLflow
Proyecto reproducible con el dataset [Employee Dataset de Kaggle](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset).
## Estructura


```text
Actividad5/
├── datos/
│   ├── datos_ini/       # CSV descargado de Kaggle (generado al ejecutar)
│   └── datos_limp/      # CSV limpio (generado al ejecutar)
├── fuentes/
│   ├── entrena.ipynb    # flujo para Google Colab
│   ├── datos_prep.py    # descarga, limpieza y preprocesamiento
│   └── train.py         # entrenamiento, validación y logging MLflow
├── README.md
├── CHANGELOG.md
└── requirements.txt
```


## Decisiones de modelado

- **Clasificación:** `LeaveOrNot` (0/1), con regresión logística y selección de `C`/`class_weight` mediante `GridSearchCV`, optimizando F1. Se reportan accuracy, precision, recall, F1 y ROC-AUC.
- **Regresión:** `ExperienceInCurrentDomain`, comparando regresión lineal y XGBoost. Es una tarea proxy: el conjunto no contiene una variable continua de salario o desempeño. Se excluye la variable objetivo de los predictores de ambos regresores. La selección por `GridSearchCV` optimiza MAE; se reportan MAE, RMSE y R².
- Se conserva un conjunto de prueba del 20 % (estratificado para clasificación); toda la codificación y escalado ocurre dentro del pipeline de cada fold para evitar fuga de información.
- Se fija semilla 42, se eliminan duplicados y filas con valores ausentes o no numéricos en las variables requeridas. Categorías se limpian y se codifican con one-hot; variables numéricas se estandarizan.
- Para clasificación se usa `StratifiedKFold(5)` y para regresión `KFold(5)`, ambos con barajado y semilla fija. Las métricas CV y de prueba se registran por separado.


## Requisitos y ejecución local


```bash
python fuentes/train.py
# o
python fuentes/train.py --data datos/datos_ini/employee_dataset.csv
```


Se crea `mlflow.db`, el CSV limpio en `datos/datos_limp/`, y `resultados.json`. Cada entrenamiento genera una corrida en el experimento `Actividad5-Employee-Dataset`. Para abrir la interfaz:


```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```


Abre `http://127.0.0.1:5000`. Para registrar los modelos en Model Registry:


```bash
python fuentes/train.py --register
```


El backend SQLite local soporta seguimiento y registro de modelos en versiones actuales de MLflow. Los modelos serializados quedan como artefactos de cada corrida.


## Google Colab


Abre `fuentes/entrena.ipynb` en Colab. El notebook instala dependencias, solicita subir el proyecto (o montar Drive), y lanza el mismo script para mantener la ejecución equivalente al modo local. Para descargar desde Kaggle en Colab se requieren credenciales Kaggle configuradas en el entorno.



