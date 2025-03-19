# Still Lost in ML - Team Challenge: Pipelines

## Descripción del Proyecto

Este repositorio contiene el desarrollo del **Team Challenge: Pipelines**. Implementamos pipelines de Scikit-learn para predecir el precio de automóviles (`Price`) utilizando un dataset de Kaggle.

### Integrantes del Grupo

-   Carlos Noya
-   Lucas Perez
-   Joaquín Ballesteros
-   Aitor Pérez
-   José Yaya

### Caso de Uso

El objetivo es predecir el precio de automóviles basado en características como:

-   Marca, modelo, año, tamaño del motor, tipo de combustible, transmisión, kilometraje, número de puertas y propietarios previos.

### Estructura del Repositorio

```
Team-Challenge---Still_Lost_in_ML
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ Team_Challenge_Pipelines.ipynb
└─ src
   ├─ data
   │  ├─ car_price_dataset_test.csv
   │  ├─ car_price_dataset_train.csv
   │  ├─ car_price_dataset.csv
   │  ├─ car_price_test_target.csv
   │  ├─ car_price_test.csv
   │  └─ car_price_train_target.csv
   │  ├─ car_price_train.csv
   ├─ models
   │  ├─ best_model.pkl
   │  └─ best_model_net.pkl
   ├─ notebooks
   │  └─ Still_Lost_in_ML - Notebook.ipynb
   ├─ result_notebooks
   │  ├─ Still_Lost_in_ML_Pipelines_I.ipynb
   │  └─ Still_Lost_in_ML_Pipelines_II.ipynb
   └─ utils
      ├─ bootcampviztools.py
      └─ toolbox_ML.py
```

### Instrucciones de Ejecución

1. Clona este repositorio:
    ```bash
    git clone https://github.com/neural-insights/Team-Challenge---Still_Lost_in_ML.git
    ```
2. Instala las depedencias
    ```bash
    pip install -r requirements.txt
    ```
3. Ejecuta los notebooks ubicados en `./src/result_notebooks/` en el siguiente orden:
    - `Still_Lost_in_ML_Pipelines_I.ipynb`: Construye el pipeline, entrena el modelo y lo guarda.
    - `Still_Lost_in_ML_Pipelines_II.ipynb`: Carga el modelo entrenado, realiza predicciones sobre el conjunto de test y evalúa las métricas.

### Dataset

El dataset utilizado es **"Car Price Dataset"**, descargado de Kaggle. Puedes acceder al dataset original desde el siguiente enlace:  
[Car Price Dataset](https://www.kaggle.com/datasets/asinow/car-price-dataset)

### Resultados

| Métrica | Valor  |
| ------- | ------ |
| RMSE    | 64.60  |
| MAPE    | 0.45%  |
| R²      | 99.95% |

#### Interpretación de las Métricas

-   **RMSE:** Error promedio de 64.60 unidades. Este valor es bajo en comparación con la escala del target (`Price`), lo que sugiere que el modelo tiene un buen desempeño.
-   **MAPE:** Error porcentual absoluto de 0.45%, lo que significa que las predicciones tienen un margen de error muy pequeño en términos relativos.
-   **R²:** Explica el 99.95% de la varianza del target, indicando un ajuste excelente del modelo a los datos.

### Modelos Evaluados

-   **ElasticNet**: Modelo final seleccionado debido a su excelente rendimiento y simplicidad.
-   **RandomForest Regressor**: Buen rendimiento en entrenamiento, pero presentó overfitting en el conjunto de prueba.
-   **XGBoost Regressor**: Buen rendimiento, aunque no superó al ElasticNet en términos de generalización.

### Ventajas de Usar Pipelines

-   Automatización del preprocesamiento (imputación, escalado, codificación one-hot).
-   Evita errores comunes como data leakage durante la validación cruzada.
-   Simplifica el flujo de trabajo y facilita la reproducibilidad.

### Autor

Team Leviathan A.K.A. **Still_Lost_in_ML**
