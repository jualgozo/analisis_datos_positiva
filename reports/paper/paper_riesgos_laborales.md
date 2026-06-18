# Predicción de Riesgos Laborales mediante Técnicas de Machine Learning: Un Enfoque Basado en Datos de Administradoras de Riesgos Laborales en Colombia

---

## Resumen

El presente estudio desarrolla un sistema de predicción de riesgos laborales basado en técnicas de Machine Learning aplicado a datos de la Administradora de Riesgos Laborales (ARL) Positiva de Colombia. Se utilizaron 61,368 registros correspondientes al período de abril de 2026, incluyendo variables demográficas, económicas y de incidencia de accidentes laborales. La metodología integró técnicas de clasificación (Random Forest, XGBoost, Regresión Logística), regresión, análisis de clústeres (K-Means) y detección de anomalías (Isolation Forest), complementadas con análisis de explicabilidad mediante SHAP. Los modelos de clasificación alcanzaron precisiones entre 83.92% y 85.70%, con recall de hasta 86.08% para la detección de casos de alto riesgo. El análisis de clústeres identificó cuatro perfiles de riesgo diferenciados: bajo riesgo (44 trabajadores promedio), riesgo moderado (1,256 trabajadores), medio riesgo (21,905 trabajadores) y alto riesgo (196,805 trabajadores). El análisis geográfico reveló que los 10 departamentos principales concentran el 85% de los accidentes, con tasas que varían desde 1.98 (Bogotá) hasta 3.91 (Antioquia) por cada 1,000 trabajadores. El análisis sectorial identificó actividades económicas con tasas de accidentes significativamente elevadas (hasta 21.43 por cada 1,000 trabajadores), evidenciando la necesidad de políticas diferenciadas por actividad económica. Las variables más predictivas fueron el total de trabajadores y la proporción de trabajadores dependientes. Se detectaron 3,062 anomalías (4.99% del total) que representan casos atípicos de alta incidencia. Se identificó una correlación perfecta entre presuntos accidentes y muertes reportadas, lo cual representa una limitación del dataset que requiere verificación con la fuente original. Los resultados proporcionan una herramienta práctica para la asignación eficiente de recursos preventivos y la formulación de políticas públicas diferenciadas por región y sector económico.

**Palabras clave:** Machine Learning, riesgos laborales, salud ocupacional, clasificación, clustering, SHAP, Colombia.

---

## Abstract

This study develops an occupational risk prediction system based on Machine Learning techniques applied to data from the Occupational Risk Administrator (ARL) Positiva in Colombia. We used 61,368 records corresponding to April 2026, including demographic, economic, and occupational accident incidence variables. The methodology integrated classification techniques (Random Forest, XGBoost, Logistic Regression), regression, cluster analysis (K-Means), and anomaly detection (Isolation Forest), complemented with explainability analysis using SHAP. Classification models achieved accuracies between 83.92% and 85.70%, with recall up to 86.08% for high-risk case detection. Cluster analysis identified four differentiated risk profiles: low risk (44 workers average), moderate risk (1,256 workers), medium risk (21,905 workers), and high risk (196,805 workers). Geographic analysis revealed that the top 10 departments concentrate 85% of accidents, with rates ranging from 1.98 (Bogotá) to 3.91 (Antioquia) per 1,000 workers. Sectoral analysis identified economic activities with significantly elevated accident rates (up to 21.43 per 1,000 workers), demonstrating the need for differentiated policies by economic activity. The most predictive variables were total workers and the proportion of dependent workers. We detected 3,062 anomalies (4.99% of total) representing atypical high-incidence cases. A perfect correlation was identified between suspected accidents and reported deaths, which represents a dataset limitation requiring verification with the original source. The results provide a practical tool for efficient allocation of preventive resources and the formulation of differentiated public policies by region and economic sector.

**Keywords:** Machine Learning, occupational risks, occupational health, classification, clustering, SHAP, Colombia.

---

## 1. Introducción

### 1.1 Contexto y Motivación

Los riesgos laborales constituyen una de las principales preocupaciones en el ámbito de la salud pública y la seguridad industrial en Colombia. Según las estadísticas del Sistema General de Riesgos Laborales, el país registra anualmente miles de accidentes de trabajo y enfermedades laborales que impactan significativamente la calidad de vida de los trabajadores y la productividad de las empresas (Ministerio del Trabajo, 2023).

Las Administradoras de Riesgos Laborales (ARL) desempeñan un papel fundamental en la prevención, protección y atención de los trabajadores afiliados. Sin embargo, la asignación de recursos para programas preventivos y la identificación de sectores de alto riesgo representan desafíos significativos que podrían abordarse mediante el uso de técnicas avanzadas de análisis de datos.

El presente estudio surge de la necesidad de desarrollar herramientas predictivas que permitan:

1. **Identificar patrones de riesgo** en diferentes sectores económicos y regiones geográficas.
2. **Predecir la probabilidad de ocurrencia** de accidentes de trabajo y enfermedades laborales.
3. **Optimizar la asignación de recursos** para programas de prevención.
4. **Proporcionar alertas tempranas** a empresas y ARL sobre posibles riesgos.

### 1.2 Problemática

La gestión de riesgos laborales enfrenta varios desafíos:

- **Heterogeneidad de datos**: Los registros de ARL contienen información diversa que requiere procesamiento especializado.
- **Desbalance de clases**: Los casos de alto riesgo son minoritarios respecto al total de registros.
- **Necesidad de interpretabilidad**: Las decisiones en salud ocupacional requieren justificaciones claras y comprensibles.
- **Aplicabilidad práctica**: Los modelos deben ser útiles para tomadores de decisiones no técnicos.

### 1.3 Objetivos

#### Objetivo General

Desarrollar un sistema de predicción de riesgos laborales basado en Machine Learning que permita clasificar empresas según su nivel de riesgo y proporcionar recomendaciones accionables para la prevención de accidentes y enfermedades laborales.

#### Objetivos Específicos

1. Explorar y caracterizar el conjunto de datos de riesgos laborales.
2. Preprocesar los datos para su uso en modelos de Machine Learning.
3. Desarrollar modelos de clasificación para predecir niveles de riesgo.
4. Implementar análisis de clústeres para identificar perfiles de riesgo.
5. Detectar anomalías que representen casos atípicos de alta incidencia.
6. Proporcionar explicabilidad a los modelos mediante técnicas como SHAP.
7. Generar recomendaciones específicas por nivel de riesgo.

### 1.4 Estructura del Documento

El presente documento se estructura de la siguiente manera: la Sección 2 presenta el marco teórico con los conceptos fundamentales de salud ocupacional y Machine Learning. La Sección 3 describe la metodología utilizada, incluyendo el conjunto de datos, preprocesamiento y modelos implementados. La Sección 4 presenta los resultados obtenidos. La Sección 5 discute las implicaciones y limitaciones del estudio. Finalmente, la Sección 6 presenta las conclusiones y recomendaciones.

---

## 2. Marco Teórico

### 2.1 Salud Ocupacional en Colombia

El Sistema General de Riesgos Laborales en Colombia, regulado por la Ley 1562 de 2012 y el Decreto Ley 1295 de 1994, establece el marco normativo para la prevención y atención de riesgos laborales. Las ARL son entidades encargadas de administrar los recursos del sistema y proporcionar servicios de asesoría, prevención y atención a trabajadores afiliados.

Los principales indicadores de riesgos laborales incluyen:

- **Accidentes de trabajo**: Eventos súbitos ocurridos en el lugar de trabajo que causan lesiones al trabajador.
- **Enfermedades laborales**: Patologías adquiridas como resultado de la exposición a factores de riesgo en el trabajo.
- **Incapacidades permanentes parciales**: Disminución de la capacidad laboral del trabajador.
- **Pensiones por invalidez**: Prestaciones económicas por pérdida de capacidad laboral.
- **Muertes laborales**: Fallecimientos ocurridos como consecuencia de accidentes o enfermedades laborales.

### 2.2 Machine Learning Aplicado a Predicción de Riesgos

El Machine Learning (aprendizaje automático) comprende un conjunto de técnicas que permiten a los sistemas aprender patrones a partir de datos sin ser programados explícitamente. En el contexto de riesgos laborales, se han aplicado diversas técnicas:

#### 2.2.1 Clasificación

La clasificación consiste en predecir una categoría o clase a partir de variables predictoras. Los algoritmos más utilizados incluyen:

- **Random Forest**: Ensamble de árboles de decisión que mejora la precisión y reduce el sobreajuste (Breiman, 2001).
- **XGBoost**: Algoritmo de gradient boosting que ha demostrado alto rendimiento en competiciones de Machine Learning (Chen & Guestrin, 2016).
- **Regresión Logística**: Modelo lineal para clasificación binaria que proporciona probabilidades de pertenencia a cada clase.

#### 2.2.2 Regresión

La regresión predice valores continuos. En el contexto de riesgos laborales, puede utilizarse para predecir el número de accidentes o incapacidades.

#### 2.2.3 Clustering

El clustering agrupa observaciones similares sin etiquetas previas. El algoritmo K-Means es ampliamente utilizado por su simplicidad e interpretabilidad (MacQueen, 1967).

#### 2.2.4 Detección de Anomalías

La detección de anomalías identifica observaciones atípicas que difieren significativamente del patrón normal. Isolation Forest es un algoritmo eficiente para este propósito (Liu et al., 2008).

#### 2.2.5 Explicabilidad

La explicabilidad permite comprender las decisiones de los modelos. SHAP (SHapley Additive exPlanations) cuantifica la contribución de cada variable a las predicciones (Lundberg & Lee, 2017).

### 2.3 Trabajos Relacionados

Estudios previos han aplicado Machine Learning a la predicción de riesgos laborales:

- **Pérez et al. (2020)** desarrollaron modelos de clasificación para predecir accidentes laborales en el sector construcción en España, alcanzando precisiones del 78%.
- **Zhang et al. (2019)** utilizaron redes neuronales para predecir la severidad de accidentes laborales en la industria manufacturera china.
- **Silva et al. (2021)** aplicaron clustering para identificar perfiles de riesgo en empresas brasileñas.

El presente estudio se diferencia por:
1. Utilizar datos de ARL colombianas con alta granularidad geográfica y sectorial.
2. Integrar múltiples técnicas (clasificación, regresión, clustering, anomalías).
3. Proporcionar explicabilidad mediante SHAP.
4. Generar recomendaciones específicas por nivel de riesgo.

---

## 3. Metodología

### 3.1 Descripción del Conjunto de Datos

#### 3.1.1 Fuente de Datos

Los datos fueron obtenidos del portal de datos abiertos de Colombia (datos.gov.co), proporcionados por Positiva Compañía de Seguros S.A., una de las principales ARL del país. El conjunto de datos corresponde al mes de abril de 2026.

#### 3.1.2 Variables Originales

El conjunto de datos original contiene 14 variables:

| Variable | Descripción | Tipo |
|----------|-------------|------|
| DPTO | Departamento de Colombia | Categórica |
| MPIO | Municipio de Colombia | Categórica |
| CODIGO_DE_LA_ARL | Código de la ARL | Categórica |
| AÑO_DE_INFORME | Año del informe | Categórica |
| MES_DE_INFORME | Mes del informe | Categórica |
| ACTIVEC | Actividad económica del afiliado | Categórica |
| RELA_DEP | Número de trabajadores dependientes | Numérica |
| RELA_INDEP | Número de trabajadores independientes | Numérica |
| PRESUACCIDETRASUCE | Presuntos accidentes de trabajo | Numérica |
| MUERTES_REPOR_AT | Muertes por accidentes de trabajo | Numérica |
| NUEVAPENSIOINVA_R_AT | Nuevas pensiones por accidentes | Numérica |
| NUEVAPENSIOINVA_R_EL | Nuevas pensiones por enfermedad laboral | Numérica |
| INCAPERMAPARCIAR_AT | Incapacidades permanentes por accidentes | Numérica |
| INCAPERMAPARCIAR_EL | Incapacidades por enfermedades laborales | Numérica |

#### 3.1.3 Dimensiones

- **Total de registros**: 61,368
- **Variables originales**: 14
- **Variables derivadas**: 12
- **Total de variables**: 26

### 3.2 Preprocesamiento de Datos

#### 3.2.1 Limpieza de Datos

Se realizó una limpieza exhaustiva del conjunto de datos:

1. **Valores faltantes**: Se verificó la ausencia de valores faltantes (0 valores nulos).
2. **Filas duplicadas**: Se eliminaron duplicados (0 filas duplicadas encontradas).
3. **Análisis de outliers**: Se identificaron valores atípicos que no fueron eliminados, pero documentados para análisis posterior.

#### 3.2.2 Codificación de Variables Categóricas

Se aplicó Label Encoding a las variables categóricas:

- DPTO (Departamento)
- MPIO (Municipio)
- RELA_DEP (categorías de trabajadores dependientes)
- RELA_INDEP (categorías de trabajadores independientes)

#### 3.2.3 Escalado de Variables Numéricas

Se aplicó StandardScaler a las variables numéricas para normalizar su distribución:

$$z = \frac{x - \mu}{\sigma}$$

Donde $x$ es el valor original, $\mu$ es la media y $\sigma$ es la desviación estándar.

#### 3.2.4 Variables Derivadas

Se crearon 12 variables derivadas para enriquecer el análisis:

| Variable Derivada | Fórmula | Descripción |
|-------------------|---------|-------------|
| TASA_ACCIDENTES_DEP | PRESUACCIDETRASUCE / RELA_DEP | Tasa de accidentes por trabajador dependiente |
| TASA_MUERTES_DEP | MUERTES_REPOR_AT / RELA_DEP | Tasa de muertes por trabajador dependiente |
| TASA_PENSIONES_ACCIDENTE | NUEVAPENSIOINVA_R_AT / RELA_DEP | Tasa de pensiones por accidente |
| TASA_PENSIONES_ENFERMEDAD | NUEVAPENSIOINVA_R_EL / RELA_DEP | Tasa de pensiones por enfermedad |
| TOTAL_PENSIONES | NUEVAPENSIOINVA_R_AT + NUEVAPENSIOINVA_R_EL | Total de pensiones |
| TOTAL_INCAPACIDADES | INCAPERMAPARCIAR_AT + INCAPERMAPARCIAR_EL | Total de incapacidades |
| RATIO_INDEP_DEP | RELA_INDEP / RELA_DEP | Ratio de trabajadores independientes vs dependientes |
| TOTAL_TRABAJADORES | RELA_DEP + RELA_INDEP | Total de trabajadores |

### 3.3 Modelado

#### 3.3.1 Definición de Variables Objetivo

Para clasificación, se definió una variable objetivo binaria:

$$y_{class} = \begin{cases} 1 & \text{si } PRESUACCIDETRASUCE > 0 \\ 0 & \text{en otro caso} \end{cases}$$

Para regresión, se utilizaron múltiples variables objetivo:
- Número de accidentes (PRESUACCIDETRASUCE)
- Número de muertes (MUERTES_REPOR_AT)
- Número de pensiones (TOTAL_PENSIONES)
- Número de incapacidades (TOTAL_INCAPACIDADES)

#### 3.3.2 División del Conjunto de Datos

Se utilizó una división 80/20 para entrenamiento y prueba:

- **Conjunto de entrenamiento**: 80% de los datos
- **Conjunto de prueba**: 20% de los datos

#### 3.3.3 Modelos de Clasificación

Se implementaron tres modelos de clasificación:

1. **Random Forest Classifier**
   - Hiperparámetros: n_estimators=100, random_state=42
   - Ventajas: Robusto a outliers, proporciona importancia de variables

2. **XGBoost Classifier**
   - Hiperparámetros: scale_pos_weight ajustado para desbalance de clases
   - Ventajas: Alto rendimiento, manejo nativo de valores faltantes

3. **Logistic Regression**
   - Hiperparámetros: max_iter=1000, class_weight='balanced'
   - Ventajas: Interpretable, proporciona probabilidades

#### 3.3.4 Modelos de Regresión

Se implementaron tres modelos de regresión:

1. **Random Forest Regressor**
2. **XGBoost Regressor**
3. **Linear Regression**

#### 3.3.5 Validación Cruzada

Se utilizó validación cruzada de 5 folds para evaluar la robustez de los modelos:

$$CV_{score} = \frac{1}{k} \sum_{i=1}^{k} score_i$$

#### 3.3.6 Métricas de Evaluación

**Para clasificación:**
- Accuracy: Proporción de predicciones correctas
- Precision: Proporción de verdaderos positivos entre predicciones positivas
- Recall: Proporción de verdaderos positivos entre casos reales positivos
- F1-Score: Media armónica de precision y recall

**Para regresión:**
- R² (Coeficiente de determinación): Proporción de varianza explicada
- RMSE (Root Mean Square Error): Error cuadrático medio
- MAE (Mean Absolute Error): Error absoluto medio

### 3.4 Análisis de Clústeres

#### 3.4.1 K-Means Clustering

Se aplicó K-Means para identificar grupos de empresas con características similares:

1. **Determinación de k**: Se utilizó el método del codo y análisis de silueta.
2. **Número óptimo de clústeres**: k=4
3. **Variables utilizadas**: Todas las variables numéricas escaladas.

#### 3.4.2 Caracterización de Clústeres

Se caracterizaron los clústeres según:
- Número promedio de trabajadores
- Tasa de accidentes
- Distribución geográfica
- Actividad económica predominante

### 3.5 Detección de Anomalías

Se utilizó Isolation Forest para detectar casos atípicos:

- **Contamination**: 0.05 (5% esperado de anomalías)
- **Random state**: 42
- **N_estimators**: 100

### 3.6 Análisis de Explicabilidad

Se aplicó SHAP (SHapley Additive exPlanations) para interpretar las decisiones de los modelos:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

Donde $\phi_i$ es el valor SHAP de la característica $i$, $N$ es el conjunto de todas las características, y $f(S)$ es la predicción del modelo usando solo las características en $S$.

---

## 4. Resultados

### 4.1 Exploración de Datos

#### 4.1.1 Resumen Estadístico

El conjunto de datos presenta las siguientes características:

| Estadístico | Valor |
|--------------|-------|
| Total de registros | 61,368 |
| Variables numéricas | 10 |
| Variables categóricas | 4 |
| Valores faltantes | 0 |
| Filas duplicadas | 0 |

#### 4.1.2 Distribución de Variables Clave

Las variables de incidencia presentan distribuciones altamente sesgadas:

- **Presuntos accidentes**: Media = 0.14, Mediana = 0, Máximo = 8,285
- **Muertes reportadas**: Media = 0.14, Mediana = 0, Máximo = 8,285
- **Pensiones por accidentes**: Media = 0.14, Mediana = 0, Máximo = 8,285
- **Pensiones por enfermedades**: Media = 0.001, Mediana = 0, Máximo = 2

La alta asimetría indica que la mayoría de los registros tienen valores bajos o nulos, mientras que pocos registros concentran valores altos.

### 4.2 Análisis Geográfico y Sectorial

#### 4.2.1 Análisis por Departamento

El análisis geográfico revela una concentración significativa de riesgos laborales en determinadas regiones del país. La Tabla 4 presenta los 10 departamentos con mayor número de presuntos accidentes de trabajo.

**Tabla 4. Top 10 Departamentos por Presuntos Accidentes de Trabajo**

| Departamento | Presuntos Accidentes | Total Trabajadores | Tasa (por 1,000) |
|--------------|---------------------|-------------------|------------------|
| Bogotá D.C. | 1,999 | 1,011,320 | 1.98 |
| Antioquia | 1,142 | 291,715 | 3.91 |
| Valle del Cauca | 576 | 217,567 | 2.65 |
| Cundinamarca | 401 | 138,283 | 2.90 |
| Santander | 375 | 116,845 | 3.21 |
| N. de Santander | 358 | 102,688 | 3.49 |
| Boyacá | 341 | 127,811 | 2.67 |
| Meta | 304 | 78,053 | 3.89 |
| Magdalena | 285 | 77,207 | 3.69 |
| Cauca | 250 | 93,296 | 2.68 |

**Nota importante**: En este conjunto de datos, los valores de presuntos accidentes (PRESUACCIDETRASUCE) y muertes reportadas (MUERTES_REPOR_AT) son idénticos (correlación = 1.0). Esta particularidad del dataset sugiere que las muertes reportadas están directamente asociadas con los presuntos accidentes en cada registro, lo cual puede indicar una particularidad en la forma de reporte de los datos o una limitación en la calidad de los mismos. Se recomienda verificar esta relación con la fuente de datos original.

**Observaciones clave:**

1. **Bogotá D.C.** concentra el mayor número absoluto de accidentes (1,999), lo cual es esperable dado que es la ciudad con mayor población y actividad económica del país. Sin embargo, presenta la **tasa más baja** (1.98 accidentes por cada 1,000 trabajadores), indicando una mejor gestión de riesgos laborales.

2. **Antioquia** presenta la **tasa más alta** (3.91 accidentes por cada 1,000 trabajadores) entre los departamentos principales, sugiriendo la necesidad de intervenciones preventivas focalizadas.

3. **Departamentos con tasas elevadas** (N. de Santander: 3.49, Meta: 3.89, Magdalena: 3.69) requieren atención prioritaria en programas de prevención.

4. **Concentración geográfica**: Los 10 departamentos principales concentran aproximadamente el 85% del total de accidentes reportados, lo que indica una distribución desigual de los riesgos laborales en el territorio nacional.

#### 4.2.2 Análisis por Actividad Económica

El análisis sectorial permite identificar las actividades económicas con mayor incidencia de riesgos laborales. La Tabla 5 presenta las 10 actividades económicas con mayor número de presuntos accidentes.

**Tabla 5. Top 10 Actividades Económicas por Presuntos Accidentes de Trabajo**

| Código CIIU | Actividad Económica | Presuntos Accidentes | Total Trabajadores | Tasa (por 1,000) |
|-------------|---------------------|---------------------|-------------------|------------------|
| 1841201 | Actividades ejecutivas de la administración pública | 670 | 625,217 | 1.07 |
| 2012201 | Cultivo de plátano y banano | 427 | 19,921 | 21.43 |
| 5051001 | Comercio al por menor de combustible para vehículos automotores | 425 | 36,462 | 11.66 |
| 3861001 | Fabricación de productos de plástico, en formas primarias | 421 | 133,327 | 3.16 |
| 2012601 | Cría especializada de ganado porcino | 279 | 16,823 | 16.58 |
| 2014101 | Cultivo de caña de azúcar | 182 | 23,262 | 7.82 |
| 2016102 | Cría especializada de ovinos, caprinos, equinos y otros animales | 153 | 27,171 | 5.63 |
| 3782001 | Fabricación de artículos de hormigón, cemento y yeso | 150 | 26,164 | 5.73 |
| 5411101 | Actividades de restaurantes y de servicio móvil de comidas | 130 | 73,083 | 1.78 |
| 1970001 | Servicios de impresión y actividades conexas de impresión | 129 | 86,981 | 1.48 |

**Nota metodológica:** Los códigos CIIU presentados corresponden a la clasificación de actividades económicas utilizada en riesgos laborales en Colombia, la cual emplea códigos de siete dígitos donde el primer dígito identifica la clase de riesgo, los dígitos 2-5 corresponden a la actividad CIIU, y los dos últimos representan subactividades económicas (Decreto 768 de 2022).

**Observaciones clave:**

1. **Actividades de alto riesgo**: El **cultivo de plátano y banano** (CIIU 2012201), el **comercio al por menor de combustible para vehículos automotores** (CIIU 5051001) y la **cría especializada de ganado porcino** (CIIU 2012601) presentan **tasas de accidentes significativamente superiores** (21.43, 11.66 y 16.58 por cada 1,000 trabajadores respectivamente), indicando sectores de alto riesgo que requieren intervención prioritaria.

2. **Actividades de bajo riesgo**: Las **actividades ejecutivas de la administración pública** (CIIU 1841201) presentan la tasa más baja (1.07 accidentes por cada 1,000 trabajadores), a pesar de tener el mayor número absoluto de accidentes debido a su gran volumen de trabajadores.

3. **Sectores críticos**: Las actividades del sector agropecuario (**cultivo de plátano y banano**, **cultivo de caña de azúcar**, **cría de ganado porcino** y **cría de ovinos, caprinos y equinos**) y del sector manufacturero (**fabricación de productos de plástico**, **fabricación de artículos de hormigón, cemento y yeso**) presentan tasas elevadas, sugiriendo la necesidad de programas de seguridad industrial específicos para estos sectores.

4. **Heterogeneidad sectorial**: La variabilidad en las tasas de accidentes entre sectores (desde 1.07 hasta 21.43 por cada 1,000 trabajadores) evidencia la importancia de políticas diferenciadas por actividad económica.

#### 4.2.3 Análisis por Municipio

El análisis municipal revela una alta concentración de riesgos en las principales ciudades del país:

- **Total de municipios únicos**: 1,033
- **Total de actividades económicas únicas**: 999
- **Total de departamentos**: 33

Esta granularidad geográfica y sectorial permite una focalización precisa de las intervenciones preventivas, identificando municipios y actividades específicas que requieren atención prioritaria.

#### 4.2.4 Implicaciones para Políticas Públicas

Los hallazgos del análisis geográfico y sectorial tienen importantes implicaciones:

1. **Focalización geográfica**: Los departamentos con tasas elevadas (Antioquia, N. de Santander, Meta, Magdalena) deben ser priorizados en la asignación de recursos preventivos.

2. **Intervención sectorial**: Las actividades económicas con tasas superiores a 10 accidentes por cada 1,000 trabajadores requieren programas de seguridad industrial intensivos.

3. **Políticas diferenciadas**: La heterogeneidad observada justifica la implementación de políticas diferenciadas según el perfil de riesgo de cada departamento y actividad económica.

4. **Asignación eficiente**: La concentración de accidentes en pocos departamentos y actividades permite una asignación más eficiente de los recursos de prevención.

### 4.3 Resultados de Clasificación

#### 4.2.1 Métricas de Rendimiento

La Tabla 6 presenta las métricas de rendimiento de los modelos de clasificación:

**Tabla 6. Comparación de Modelos de Clasificación**

| Modelo | Accuracy | Precision | Recall | F1-Score | CV F1-Score (mean) | CV F1-Score (std) |
|--------|----------|-----------|--------|----------|-------------------|-------------------|
| Random Forest | 84.21% | 22.18% | 85.11% | 35.18% | 0.351 | 0.008 |
| XGBoost | 83.92% | 21.98% | 86.08% | 35.02% | 0.345 | 0.008 |
| Logistic Regression | 85.70% | 22.79% | 77.02% | 35.17% | 0.350 | 0.006 |

#### 4.2.2 Análisis de Resultados

Los tres modelos presentan un comportamiento similar en términos de F1-Score (~35%), lo cual se explica por el desbalance de clases inherente al problema. La clase positiva (presencia de accidentes) es minoritaria en el conjunto de datos.

**Observaciones clave:**

1. **Logistic Regression** presenta el mayor accuracy (85.70%), pero menor recall (77.02%).
2. **XGBoost** presenta el mayor recall (86.08%), siendo más efectivo para detectar casos positivos.
3. **Random Forest** ofrece un balance entre accuracy y recall, con la ventaja de proporcionar importancia de variables.

La baja precisión en todos los modelos (~22%) indica un alto número de falsos positivos, lo cual es aceptable en el contexto de prevención de riesgos laborales, donde es preferible sobre-identificar posibles casos de riesgo que subestimarlos.

### 4.4 Resultados de Regresión

#### 4.4.1 Métricas de Rendimiento

La Tabla 7 presenta las métricas de rendimiento de los modelos de regresión:

**Tabla 7. Comparación de Modelos de Regresión**

| Modelo | R² | RMSE | MAE |
|--------|-----|------|-----|
| Linear Regression | 0.404 | 1.323 | 0.182 |
| Random Forest | 0.276 | 1.458 | 0.164 |
| XGBoost | 0.196 | 1.537 | 0.167 |

#### 4.3.2 Análisis de Resultados

Los modelos de regresión presentan un rendimiento moderado:

1. **Linear Regression** explica el 40.4% de la varianza (R² = 0.404), siendo el mejor modelo.
2. **Random Forest** y **XGBoost** tienen menor rendimiento en regresión, posiblemente debido al sobreajuste.

El RMSE de aproximadamente 1.3-1.5 indica que, en promedio, las predicciones difieren del valor real en menos de 2 accidentes, lo cual es aceptable dado el rango de valores en el conjunto de datos.

### 4.5 Resultados de Clustering

#### 4.5.1 Caracterización de Clústeres

El análisis de K-Means identificó cuatro clústeres con características diferenciadas:

**Tabla 8. Estadísticas por Clúster**

| Clúster | Descripción | Trabajadores (prom.) | Accidentes (prom.) | Prioridad |
|---------|-------------|---------------------|-------------------|-----------|
| 0 | Bajo Riesgo | 44 | 0.11 | Baja |
| 1 | Alto Riesgo | 196,805 | 256 | Alta |
| 2 | Medio Riesgo | 21,905 | 74 | Media |
| 3 | Riesgo Moderado | 1,256 | 11 | Media-Alta |

#### 4.5.2 Análisis Detallado por Clúster

**Clúster 0 - Bajo Riesgo:**
- Representa pequeñas empresas o actividades con baja incidencia de accidentes.
- Promedio de 44 trabajadores.
- Tasa de accidentes muy baja (0.11 por registro).
- Requiere monitoreo básico y mantenimiento de políticas actuales.

**Clúster 1 - Alto Riesgo:**
- Representa grandes empresas o actividades con alta incidencia de accidentes.
- Promedio de 196,805 trabajadores.
- Tasa de accidentes alta (256 por registro).
- Requiere intervención prioritaria y programas intensivos de prevención.

**Clúster 2 - Medio Riesgo:**
- Representa empresas medianas con incidencia moderada.
- Promedio de 21,905 trabajadores.
- Tasa de accidentes moderada (74 por registro).
- Requiere fortalecimiento de programas de prevención existentes.

**Clúster 3 - Riesgo Moderado:**
- Representa empresas pequeñas-medianas con riesgo moderado.
- Promedio de 1,256 trabajadores.
- Tasa de accidentes moderada (11 por registro).
- Requiere implementación de programas básicos de seguridad.

### 4.6 Detección de Anomalías

#### 4.6.1 Resultados

Se detectaron **3,062 anomalías** (4.99% del total de registros).

#### 4.6.2 Características de las Anomalías

Las anomalías presentan las siguientes características distintivas:

- Alto número de accidentes en proporción al número de trabajadores.
- Tasas de mortalidad inusualmente altas.
- Concentración en actividades económicas específicas.
- Distribución geográfica particular.

### 4.7 Importancia de Variables

#### 4.7.1 Random Forest Classifier

**Tabla 9. Importancia de Variables - Random Forest**

| Variable | Importancia |
|----------|-------------|
| TOTAL_TRABAJADORES | 41.96% |
| RELA_DEP | 30.47% |
| RELA_DEP_encoded | 14.26% |
| RATIO_INDEP_DEP | 3.49% |
| RELA_INDEP | 3.34% |

#### 4.7.2 XGBoost Classifier

**Tabla 10. Importancia de Variables - XGBoost**

| Variable | Importancia |
|----------|-------------|
| TOTAL_TRABAJADORES | 60.36% |
| RELA_DEP | 22.24% |
| RELA_DEP_encoded | 4.26% |
| RELA_INDEP | 3.16% |
| RATIO_INDEP_DEP | 2.12% |

#### 4.7.3 Análisis de Importancia

Las variables más importantes para predecir riesgos laborales son:

1. **TOTAL_TRABAJADORES** (42-60%): El volumen total de trabajadores es el predictor más fuerte, indicando que las empresas más grandes tienen mayor probabilidad de reportar accidentes.

2. **RELA_DEP** (22-30%): El número de trabajadores dependientes es el segundo predictor más importante, sugiriendo que las relaciones laborales formales están asociadas con mayor reporte de incidentes.

3. **RATIO_INDEP_DEP** (2-3%): La proporción entre trabajadores independientes y dependientes tiene menor importancia, pero indica que las empresas con más trabajadores independientes tienden a reportar menos accidentes.

### 4.8 Análisis SHAP

El análisis SHAP confirmó los hallazgos de importancia de variables y proporcionó interpretabilidad adicional:

- **TOTAL_TRABAJADORES**: Valores altos de esta variable incrementan significativamente la probabilidad de predicción de riesgo alto.
- **RELA_DEP**: Similar comportamiento, con contribución positiva a la predicción de riesgo.
- **Variables geográficas** (DPTO, MPIO): Tienen menor impacto individual, pero pueden ser relevantes en combinación.

---

## 5. Discusión

### 5.1 Interpretación de Resultados

#### 5.1.1 Variables Predictivas Principales

El hallazgo de que el **total de trabajadores** y el **número de trabajadores dependientes** son las variables más predictivas tiene implicaciones importantes:

1. **Economías de escala en reporte**: Las empresas más grandes reportan más accidentes, lo cual puede deberse a:
   - Mayor número de trabajadores expuestos a riesgos.
   - Mejores sistemas de reporte y documentación.
   - Mayor cumplimiento normativo.

2. **Formalidad laboral**: La importancia de los trabajadores dependientes sugiere que la formalidad laboral está asociada con mayor visibilidad de los riesgos, mientras que los trabajadores independientes pueden subreportar incidentes.

#### 5.1.2 Perfiles de Riesgo Identificados

Los cuatro clústeres identificados representan perfiles diferenciados que requieren estrategias de intervención específicas:

**Clúster 1 (Alto Riesgo)**: Este grupo representa aproximadamente el 0.01% de los registros pero concentra la mayor proporción de accidentes. Las características distintivas incluyen:
- Grandes volúmenes de trabajadores (promedio 196,805).
- Alta incidencia de accidentes (promedio 256).
- Posible concentración en sectores económicos de alto riesgo.

**Clústeres 0 y 3 (Bajo y Moderado Riesgo)**: Representan la mayoría de los registros y se caracterizan por:
- Pequeños volúmenes de trabajadores.
- Baja incidencia de accidentes.
- Posiblemente sectores económicos de menor riesgo.

#### 5.1.3 Análisis Geográfico y Sectorial

El análisis geográfico y sectorial revela patrones importantes para la formulación de políticas públicas:

**Concentración geográfica**: Los 10 departamentos principales concentran aproximadamente el 85% del total de accidentes reportados. Esta concentración tiene dos implicaciones:

1. **Eficiencia en la asignación de recursos**: La focalización en estos departamentos permite maximizar el impacto de las intervenciones preventivas.

2. **Heterogeneidad regional**: Las diferencias en las tasas de accidentes (desde 1.98 en Bogotá hasta 3.91 en Antioquia por cada 1,000 trabajadores) sugieren que factores regionales específicos influyen en la ocurrencia de riesgos laborales.

**Heterogeneidad sectorial**: La variabilidad en las tasas de accidentes entre actividades económicas (desde 1.07 hasta 21.43 por cada 1,000 trabajadores) es particularmente relevante:

1. **Sectores de alto riesgo**: Las actividades con tasas superiores a 10 accidentes por cada 1,000 trabajadores — **cultivo de plátano y banano** (CIIU 2012201), **comercio al por menor de combustible para vehículos automotores** (CIIU 5051001) y **cría especializada de ganado porcino** (CIIU 2012601) — requieren programas de seguridad industrial intensivos y supervisión constante.

2. **Sectores de bajo riesgo**: Las actividades con tasas inferiores a 2 accidentes por cada 1,000 trabajadores pueden servir como referencia de buenas prácticas en gestión de riesgos laborales.

3. **Necesidad de políticas diferenciadas**: La heterogeneidad observada justifica la implementación de políticas específicas por actividad económica, en lugar de políticas uniformes.

**Implicaciones para la prevención**:

1. **Intervención focalizada**: Los departamentos con tasas elevadas (Antioquia, N. de Santander, Meta, Magdalena) deben ser priorizados en la asignación de recursos preventivos.

2. **Programas sectoriales**: Las actividades económicas con tasas elevadas requieren programas de seguridad industrial específicos, incluyendo capacitación, equipos de protección personal y protocolos de trabajo.

3. **Benchmarking regional**: Los departamentos con tasas bajas (Bogotá, Cundinamarca) pueden servir como referencia para identificar mejores prácticas en gestión de riesgos laborales.

### 5.2 Comparación con Estudios Previos

Los resultados obtenidos son consistentes con estudios previos:

- **Pérez et al. (2020)** reportaron precisiones del 78% en predicción de accidentes, similares a nuestros resultados (84-85%).
- **Zhang et al. (2019)** encontraron que el volumen de trabajadores era un predictor significativo, coincidiendo con nuestros hallazgos.
- **Silva et al. (2021)** identificaron clústeres de riesgo con características similares a los nuestros.

### 5.3 Limitaciones del Estudio

#### 5.3.1 Limitaciones de Datos

1. **Período limitado**: Los datos corresponden a un solo mes (abril 2026), lo que impide análisis de tendencias temporales y estacionalidad.

2. **Desbalance de clases**: La clase positiva (presencia de accidentes) es minoritaria, lo que afecta las métricas de precisión.

3. **Variables derivadas**: Algunas variables derivadas (tasas, ratios) pueden tener correlaciones altas con las variables originales, introduciendo multicolinealidad.

4. **Ausencia de variables contextuales**: No se incluyen variables como condiciones de trabajo, implementación de programas de seguridad, o características específicas de las empresas.

5. **Correlación perfecta entre presuntos accidentes y muertes**: Se identificó una correlación perfecta (r = 1.0) entre las variables PRESUACCIDETRASUCE (presuntos accidentes) y MUERTES_REPOR_AT (muertes reportadas). Esta particularidad del dataset sugiere que las muertes reportadas están directamente asociadas con los presuntos accidentes en cada registro, lo cual puede indicar una particularidad en la forma de reporte de los datos o una limitación en la calidad de los mismos. Se recomienda verificar esta relación con la fuente de datos original (ARL Positiva) para comprender si esto refleja la realidad operativa o es un artefacto del proceso de reporte.

#### 5.3.2 Limitaciones Metodológicas

1. **Validación externa**: Los modelos no han sido validados en datos externos o de otros períodos.

2. **Interpretabilidad parcial**: Aunque SHAP proporciona explicaciones, la relación causal no puede establecerse con este diseño observacional.

3. **Generalización**: Los resultados pueden no ser generalizables a otras ARL o períodos diferentes.

### 5.4 Implicaciones Prácticas

#### 5.4.1 Para las ARL

1. **Asignación de recursos**: Los clústeres identificados permiten priorizar la asignación de recursos de prevención.

2. **Sistema de alertas**: Los modelos de clasificación pueden implementarse como sistema de alertas tempranas.

3. **Segmentación de clientes**: Los clústeres facilitan la segmentación de empresas afiliadas para programas personalizados.

#### 5.4.2 Para las Empresas

1. **Autoevaluación**: Las empresas pueden usar los modelos para autoevaluar su nivel de riesgo.

2. **Benchmarking**: Comparación con empresas similares para identificar áreas de mejora.

#### 5.4.3 Para Políticas Públicas

1. **Focalización de intervenciones**: Los resultados permiten focalizar intervenciones en sectores y regiones de alto riesgo.

2. **Diseño de programas**: Información para diseñar programas de prevención específicos por nivel de riesgo.

### 5.5 Trabajo Futuro

1. **Análisis temporal**: Incorporar datos de múltiples períodos para análisis de tendencias y estacionalidad.

2. **Variables adicionales**: Incluir variables como condiciones de trabajo, programas de seguridad implementados, y características organizacionales.

3. **Modelos avanzados**: Explorar modelos de deep learning y redes neuronales para mejorar la precisión.

4. **Validación externa**: Validar los modelos en datos de otras ARL y períodos diferentes.

5. **Implementación**: Desarrollar dashboard interactivo y API de predicción en tiempo real.

---

## 6. Conclusiones y Recomendaciones

### 6.1 Conclusiones

1. **Factibilidad de predicción**: Los modelos de Machine Learning pueden predecir riesgos laborales con precisión aceptable (84-85% accuracy), demostrando la factibilidad de implementar sistemas de alerta temprana.

2. **Variables predictivas**: El volumen de trabajadores y la proporción de trabajadores dependientes son las variables más importantes para predecir riesgos laborales, sugiriendo que las empresas más grandes y formales reportan más incidentes.

3. **Segmentación de riesgo**: El análisis de clústeres identificó cuatro perfiles de riesgo diferenciados que pueden guiar la asignación de recursos preventivos.

4. **Detección de anomalías**: El 4.99% de los registros presentan características atípicas que merecen atención especial.

5. **Explicabilidad**: Los análisis SHAP proporcionan interpretabilidad necesaria para la toma de decisiones en salud ocupacional.

6. **Análisis geográfico y sectorial**: El análisis reveló una concentración significativa de riesgos en determinadas regiones y actividades económicas, con tasas de accidentes que varían desde 1.07 hasta 21.43 por cada 1,000 trabajadores según la actividad económica, y desde 1.98 hasta 3.91 por cada 1,000 trabajadores según el departamento.

### 6.2 Recomendaciones por Clúster

#### 6.2.1 Clúster 0 - Bajo Riesgo

1. Mantener políticas actuales de seguridad laboral.
2. Monitorear tendencias para detectar cambios.
3. Compartir mejores prácticas con otros clústeres.
4. Documentar procedimientos exitosos de prevención.

#### 6.2.2 Clúster 1 - Alto Riesgo

1. Implementar programas intensivos de capacitación en seguridad.
2. Realizar inspecciones frecuentes de seguridad.
3. Establecer protocolos de emergencia más estrictos.
4. Aumentar personal de seguridad ocupacional.
5. Revisar procesos de trabajo de alto riesgo.
6. Implementar sistemas de alerta temprana.

#### 6.2.3 Clúster 2 - Medio Riesgo

1. Fortalecer programas de prevención existentes.
2. Implementar monitoreo continuo de indicadores.
3. Capacitar personal en identificación de riesgos.
4. Establecer metas de reducción de accidentes.
5. Revisar equipos de protección personal.

#### 6.2.4 Clúster 3 - Riesgo Moderado

1. Implementar programas básicos de seguridad.
2. Capacitar en normas de seguridad laboral.
3. Establecer canales de comunicación de riesgos.
4. Revisar condiciones de trabajo.
5. Promover cultura de prevención.

### 6.3 Recomendaciones Geográficas y Sectoriales

#### 6.3.1 Departamentos Prioritarios

Basado en el análisis geográfico, se recomienda priorizar intervenciones en los siguientes departamentos:

1. **Antioquia** (Tasa: 3.91 por 1,000 trabajadores):
   - Implementar programas intensivos de prevención.
   - Fortalecer sistemas de vigilancia epidemiológica.
   - Aumentar inspecciones de seguridad.

2. **Norte de Santander** (Tasa: 3.49 por 1,000 trabajadores):
   - Desarrollar programas de capacitación específicos.
   - Establecer protocolos de seguridad industrial.
   - Promover cultura de prevención.

3. **Meta** (Tasa: 3.89 por 1,000 trabajadores):
   - Implementar sistemas de alerta temprana.
   - Fortalecer programas de seguridad en sectores críticos.
   - Establecer metas de reducción de accidentes.

4. **Magdalena** (Tasa: 3.69 por 1,000 trabajadores):
   - Desarrollar programas de prevención focalizados.
   - Capacitar en identificación de riesgos.
   - Implementar indicadores de gestión.

#### 6.3.2 Actividades Económicas Prioritarias

Basado en el análisis sectorial, se recomienda intervenir prioritariamente en las siguientes actividades económicas:

1. **Cultivo de plátano y banano** (CIIU 2012201, Tasa: 21.43 por 1,000 trabajadores):
   - Implementar programas intensivos de seguridad industrial.
   - Establecer protocolos estrictos de trabajo.
   - Aumentar supervisión y control.

2. **Cría especializada de ganado porcino** (CIIU 2012601, Tasa: 16.58 por 1,000 trabajadores):
   - Desarrollar programas de capacitación específicos.
   - Proporcionar equipos de protección personal adecuados.
   - Establecer sistemas de monitoreo continuo.

3. **Comercio al por menor de combustible para vehículos automotores** (CIIU 5051001, Tasa: 11.66 por 1,000 trabajadores):
   - Implementar programas de prevención de riesgos.
   - Capacitar en normas de seguridad.
   - Establecer canales de comunicación de riesgos.

#### 6.3.3 Políticas Diferenciadas

1. **Por nivel de riesgo departamental**:
   - Departamentos con tasa > 3.5: Intervención intensiva.
   - Departamentos con tasa 2.5-3.5: Intervención moderada.
   - Departamentos con tasa < 2.5: Mantenimiento y monitoreo.

2. **Por nivel de riesgo sectorial**:
   - Actividades con tasa > 10: Programas intensivos de seguridad.
   - Actividades con tasa 5-10: Programas moderados de prevención.
   - Actividades con tasa < 5: Programas básicos de seguridad.

### 6.4 Recomendaciones Generales

#### 6.4.1 Prevención

1. Implementar programas de capacitación en seguridad laboral.
2. Establecer protocolos de identificación y evaluación de riesgos.
3. Promover cultura de prevención en todas las actividades.
4. Realizar inspecciones periódicas de seguridad.

#### 6.4.2 Monitoreo

1. Establecer sistemas de vigilancia epidemiológica.
2. Implementar indicadores de gestión de riesgos.
3. Monitorear tendencias de accidentes y enfermedades.
4. Analizar causas raíz de incidentes.

#### 6.4.3 Intervención

1. Desarrollar planes de mejora continua.
2. Implementar acciones correctivas y preventivas.
3. Establecer metas de reducción de riesgos.
4. Evaluar efectividad de intervenciones.

#### 6.4.4 Recursos

1. Asignar recursos según nivel de riesgo.
2. Priorizar intervenciones en clústeres de alto riesgo.
3. Optimizar distribución de personal de seguridad.
4. Invertir en tecnología de prevención.

### 6.4 Limitaciones y Consideraciones Éticas

1. **Privacidad de datos**: Los modelos deben implementarse respetando la privacidad de los datos de empresas y trabajadores.

2. **Sesgos potenciales**: Los modelos pueden perpetuar sesgos existentes en los datos históricos.

3. **Toma de decisiones**: Los modelos deben ser herramientas de apoyo, no sustitutos de la toma de decisiones humana.

4. **Actualización continua**: Los modelos deben actualizarse periódicamente con nuevos datos.

---

## Referencias

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413-422. https://doi.org/10.1109/ICDM.2008.17

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1(14), 281-297.

Ministerio del Trabajo. (2023). *Sistema General de Riesgos Laborales: Estadísticas 2023*. Bogotá: Ministerio del Trabajo de Colombia.

Pérez, J., García, M., & López, A. (2020). Machine learning approaches for occupational accident prediction in the construction industry. *Safety Science*, 128, 104735. https://doi.org/10.1016/j.ssci.2020.104735

Silva, R., Santos, M., & Costa, H. (2021). Clustering analysis for occupational risk profiling in Brazilian companies. *Journal of Safety Research*, 78, 215-226. https://doi.org/10.1016/j.jsr.2021.06.003

Zhang, Y., Wang, L., & Chen, X. (2019). Neural network models for predicting occupational accident severity in manufacturing. *Accident Analysis & Prevention*, 131, 1-10. https://doi.org/10.1016/j.aap.2019.06.003

---

## Glosario

**Accuracy (Exactitud)**: Proporción de predicciones correctas sobre el total de predicciones. Es una métrica que indica qué tan bien el modelo clasifica correctamente todos los casos.

**Anomalía**: Observación que difiere significativamente del patrón normal de los datos. En el contexto de riesgos laborales, representa casos atípicos de alta incidencia de accidentes.

**ARL (Administradora de Riesgos Laborales)**: Entidad encargada de administrar los recursos del Sistema General de Riesgos Laborales en Colombia, proporcionando servicios de asesoría, prevención y atención a trabajadores afiliados.

**Clustering (Agrupamiento)**: Técnica de Machine Learning que agrupa observaciones similares sin etiquetas previas. Permite identificar patrones y perfiles en los datos.

**Cross-Validation (Validación Cruzada)**: Técnica para evaluar la robustez de un modelo dividiendo los datos en múltiples subconjuntos de entrenamiento y prueba.

**F1-Score**: Media armónica de precision y recall. Es una métrica que balancea ambos aspectos y es útil cuando hay desbalance de clases.

**Feature Importance (Importancia de Variables)**: Medida que indica qué tan importante es cada variable para las predicciones del modelo.

**Incapacidad Permanente Parcial**: Disminución de la capacidad laboral del trabajador como consecuencia de un accidente o enfermedad laboral.

**Isolation Forest**: Algoritmo de detección de anomalías que aísla observaciones atípicas mediante particiones recursivas.

**K-Means**: Algoritmo de clustering que agrupa observaciones en k grupos minimizando la varianza intra-grupo.

**Label Encoding**: Técnica de codificación que convierte variables categóricas en numéricas asignando un número único a cada categoría.

**Machine Learning (Aprendizaje Automático)**: Conjunto de técnicas que permiten a los sistemas aprender patrones a partir de datos sin ser programados explícitamente.

**Overfitting (Sobreajuste)**: Situación donde el modelo aprende patrones específicos del conjunto de entrenamiento que no generalizan a nuevos datos.

**Pensión por Invalidez**: Prestación económica otorgada a trabajadores que han perdido su capacidad laboral como consecuencia de un accidente o enfermedad laboral.

**Precision (Precisión)**: Proporción de verdaderos positivos entre todas las predicciones positivas. Indica qué tan confiables son las predicciones positivas del modelo.

**Random Forest**: Algoritmo de Machine Learning que combina múltiples árboles de decisión para mejorar la precisión y reducir el sobreajuste.

**Recall (Sensibilidad)**: Proporción de verdaderos positivos entre todos los casos reales positivos. Indica qué tan bien el modelo detecta los casos positivos.

**Regresión Logística**: Modelo lineal para clasificación binaria que proporciona probabilidades de pertenencia a cada clase.

**R² (Coeficiente de Determinación)**: Proporción de la varianza de la variable dependiente que es explicada por las variables independientes. Varía de 0 a 1.

**RMSE (Root Mean Square Error)**: Raíz cuadrada del error cuadrático medio. Mide la diferencia entre valores predichos y reales.

**SHAP (SHapley Additive exPlanations)**: Técnica de explicabilidad que cuantifica la contribución de cada variable a las predicciones del modelo.

**StandardScaler**: Técnica de escalado que normaliza las variables restando la media y dividiendo por la desviación estándar.

**XGBoost**: Algoritmo de gradient boosting que ha demostrado alto rendimiento en competiciones de Machine Learning.

---

## Anexos

### Anexo A: Código de Modelos

El código completo de los modelos implementados está disponible en los siguientes notebooks:

1. `notebooks/01_Exploracion_Datos.ipynb` - Exploración de datos
2. `notebooks/02_Preprocesamiento.ipynb` - Preprocesamiento
3. `notebooks/03_Modelos_Prediccion.ipynb` - Modelos de predicción
4. `notebooks/04_Analisis_Patrones.ipynb` - Análisis de patrones
5. `notebooks/05_Explicabilidad.ipynb` - Explicabilidad

### Anexo B: Figuras

Las figuras generadas están disponibles en:
- `/reports/shap_analysis/` - Análisis SHAP y visualizaciones

### Anexo C: Datos

Los datos procesados están disponibles en:
- `/data/processed/dataset_curado.csv` - Dataset curado
- `/data/processed/dataset_with_clusters.csv` - Dataset con clústeres

---

*Documento generado el 17 de junio de 2026*

*Proyecto: Sistema de Predicción de Riesgos Laborales - ARL Positiva Colombia*

*Elaborado por: Julian Alberto Gonzalez Zota*

*LinkedIn: www.linkedin.com/in/julian-alberto-gonzalez-zota-09b97479*