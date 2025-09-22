# IA-avanzada-para-la-ciencia-de-datos

Repositorio del proyecto de detección de fraudes en apertura de cuentas bancarias realizado para la asignatura “Inteligencia Artificial Avanzada para la Ciencia de Datos”.

---

## Dataset

El dataset principal utilizado es el **Bank Account Fraud (BAF) Dataset** disponible en Kaggle:

🔗 [Bank Account Fraud Dataset (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

Este dataset contiene aplicaciones bancarias simuladas, algunas legítimas y otras fraudulentas, con un fuerte **desbalance de clases**.

---
## Preparación de los datos

- Se cuenta con un **dataset base** y dos **variantes sintéticas (Variante 1 y Variante 2)**.  
- Para los modelos combinados se tomó un 80 % del dataset base + 20 % de la variante 1 + 20 % de la variante 2.  
- En los modelos temporales, se usaron datos de los **meses 0 al 5** para entrenamiento, reservando los meses posteriores para test.

---
## Modelos implementados

Se entrenaron y compararon los siguientes modelos:

| Modelo       | Temporalidad       | Balanceo aplicado correctamente |
|---------------|---------------------|-------------------------------|
| **XGBoost Atemporal** "XGBoostAtemporal.py" | Sin separación temporal | **NO**: se aplicó SMOTE-Tomek sobre todo el dataset → contaminación de datos |
| **XGBoost Temporal** "xgboost_model.py" | Meses 0-5 para entrenamiento, resto para test | Sí, balanceo solo al training |
| **LightGBM Temporal** "lightgbm_model.py" | Meses 0-5 para entrenamiento, resto para test | Sí, correcto |
| **CatBoost Temporal** "catboost_modelo.py" | Meses 0-5 para entrenamiento, resto para test | Sí, correcto |

## Cómo reproducir los resultados

1. Clonar el repositorio:

  ```bash
   git clone https://github.com/dinav2/IA-avanzada-para-la-ciencia-de-datos.git
   cd IA-avanzada-para-la-ciencia-de-datos
   ```
2. Crear un entorno virtual y activar:
   
  ```bash
  python -m venv venv
  venv\Scripts\activate  # en Windows

   ```
3. Instalar dependencias
  ```bash
  pip install -r requirements.txt

   ```
5. Preparar los datos: descomprimir los archivos dataset_final.zip, train_balanced_base.zip, etc., según lo que el modelo específico requiera.

6. Ejecutar los modelos:
 ```bash
  python XGBoostAtemporal.py

  python xgboost_model.py

  python lightgbm_model.py

  python catboost_model.py

   ```

