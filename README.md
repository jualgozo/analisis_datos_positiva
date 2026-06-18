# Sistema de Predicción de Riesgos Laborales - Colombia

## Descripción del Proyecto

Este proyecto implementa un **sistema de predicción de riesgos laborales** basado en Machine Learning para Colombia. Utiliza datos de la Administradora de Riesgos Laborales (ARL) Positiva Compañía de Seguros para predecir accidentes de trabajo, enfermedades laborales, y analizar patrones de incapacidades y mortalidad laboral.

## Contexto y Fuente de Datos

- **Fuente:** Positiva Compañía de Seguros S.A.
- **Cobertura Geográfica:** Nacional (Colombia)
- **Período:** Datos trimestrales desde 2017
- **Licencia:** CC BY-SA (Creative Commons - Atribución y Compartir Igual)
- **Enlace:** http://www.positiva.gov.co

## Objetivos del Proyecto

### Predicción de Riesgos
- Predecir la probabilidad de **accidentes de trabajo** y **enfermedades laborales** por región, actividad económica y tipo de trabajador.
- Identificar sectores y regiones con mayor riesgo.

### Análisis de Patrones
- Detectar patrones en incapacidades permanentes parciales.
- Agrupar regiones y actividades económicas con comportamientos similares.

### Optimización de Recursos
- Predecir pensiones e incapacidades futuras.
- Estimar costos asociados para las ARL.

### Alertas Tempranas
- Identificar anomalías en los datos.
- Proporcionar alertas sobre posibles riesgos laborales.

## Estructura del Repositorio

```
DatosArl_Mac/
├── data/
│   ├── processed/
│   │   ├── dataset_curado.csv          # Dataset limpio y procesado
│   │   └── dataset_with_clusters.csv   # Dataset con etiquetas de clusters
├── notebooks/
│   ├── 01_Exploracion_Datos.ipynb      # Análisis exploratorio de datos
│   ├── 02_Preprocesamiento.ipynb       # Limpieza y preparación de datos
│   ├── 03_Modelos_Prediccion.ipynb     # Modelos de clasificación y regresión
│   ├── 04_Analisis_Patrones.ipynb      # Clustering y análisis de patrones
│   └── 05_Explicabilidad.ipynb         # Análisis SHAP y explicabilidad
├── reports/
│   ├── paper/                          # Documentos académicos
│   ├── shap_analysis/                  # Resultados de análisis SHAP
│   ├── *.csv                           # Métricas y estadísticas
│   └── *.png                           # Visualizaciones
├── models/                             # Modelos entrenados guardados
├── main_dataset.csv                    # Dataset original
├── DiccionarioDatos.csv                # Diccionario de variables
├── ContextoData.md                     # Contexto de la fuente de datos
├── Project_Brief.md                    # Especificaciones del proyecto
└── README.md                           # Este archivo
```

## Variables Principales

| Variable | Descripción |
|----------|-------------|
| `DPTO` | Departamento de Colombia |
| `MPIO` | Municipio de Colombia |
| `CODIGO_DE_LA_ARL` | Código de la ARL |
| `AÑO_DE_INFORME` | Año del informe |
| `MES_DE_INFORME` | Mes del informe |
| `ACTIVEC` | Actividad económica del afiliado |
| `RELA_DEP` | Trabajadores dependientes |
| `RELA_INDEP` | Trabajadores independientes |
| `PRESUACCIDETRASUCE` | Presuntos accidentes de trabajo |
| `MUERTES_REPOR_AT` | Muertes por accidentes de trabajo |
| `NUEVAPENSIONIVA_R_AT` | Nuevas pensiones por accidentes |
| `NUEVAPENSIONIVA_R_EL` | Nuevas pensiones por enfermedad laboral |
| `INCAPERMAPARCIAR_AT` | Incapacidades permanentes por accidentes |
| `INCAPERMAPARCIAR_EL` | Incapacidades permanentes por enfermedad |

## Modelos Implementados

### Clasificación y Regresión
- **Random Forest Classifier/Regressor**
- **XGBoost Classifier/Regressor**
- **Logistic Regression**
- **Gradient Boosting**

### Clustering
- **K-Means**
- **DBSCAN**

### Detección de Anomalías
- **Isolation Forest**
- **PCA-based Anomaly Detection**

### Explicabilidad
- **SHAP (SHapley Additive exPlanations)**
- **Feature Importance Analysis**

## Resultados Generados

### Análisis Exploratorio
- Distribución de variables numéricas y categóricas
- Matriz de correlación
- Análisis por departamento y actividad económica

### Modelos de Predicción
- Comparación de modelos de clasificación
- Comparación de modelos de regresión
- Validación cruzada
- Matrices de confusión

### Análisis de Patrones
- Clustering con K-Means y DBSCAN
- Características de clusters
- Visualizaciones PCA

### Análisis Temporal
- Tendencias temporales
- Análisis de estacionalidad

### Explicabilidad
- Importancia de características
- Análisis SHAP detallado

## Requisitos

```python
# Python 3.8+
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
jupyter
```

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd DatosArl_Mac
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar Notebooks

Los notebooks están numerados en orden lógico de ejecución:

1. **Exploración de Datos**: `01_Exploracion_Datos.ipynb`
   - Análisis inicial de datos
   - Visualizaciones exploratorias

2. **Preprocesamiento**: `02_Preprocesamiento.ipynb`
   - Limpieza de datos
   - Feature engineering

3. **Modelos de Predicción**: `03_Modelos_Prediccion.ipynb`
   - Entrenamiento de modelos
   - Evaluación de rendimiento

4. **Análisis de Patrones**: `04_Analisis_Patrones.ipynb`
   - Clustering
   - Detección de anomalías

5. **Explicabilidad**: `05_Explicabilidad.ipynb`
   - Análisis SHAP
   - Interpretación de modelos

### Datos de Entrada

El dataset principal se encuentra en `main_dataset.csv`. Los datos procesados están en `data/processed/`.

## Impacto Esperado

- **Reducción de riesgos laborales** mediante alertas tempranas
- **Optimización de recursos** en las ARL
- **Mejora en políticas públicas** de salud ocupacional
- **Prevención proactiva** de accidentes y enfermedades laborales

## Licencia

Este proyecto utiliza datos bajo licencia **CC BY-SA** de Positiva Compañía de Seguros S.A.

## Contacto

Para más información sobre los datos originales, visitar: http://www.positiva.gov.co

---

**Última actualización:** Junio 2026
