
# Resumen de Preprocesamiento - Riesgos Laborales Colombia
Fecha: 2026-06-17 12:50:53

## Dataset Original
- Dimensiones: 61368 filas x 14 columnas
- Valores faltantes: 0
- Filas duplicadas: 0

## Dataset Curado
- Dimensiones: 61368 filas x 26 columnas
- Valores faltantes: 0
- Filas duplicadas: 0

## Transformaciones Realizadas
1. **Limpieza de datos**:
   - Imputación de valores faltantes (mediana para numéricas, moda para categóricas)
   - Eliminación de filas duplicadas
   - Análisis de outliers (no eliminados, documentados)

2. **Codificación de variables categóricas**:
   - Label Encoding para: ['DPTO', 'MPIO', 'RELA_DEP', 'RELA_INDEP']

3. **Escalado de variables numéricas**:
   - StandardScaler aplicado a 20 variables numéricas

4. **Variables derivadas creadas**:
   - TASA_ACCIDENTES_DEP: Tasa de accidentes por trabajador dependiente
   - TASA_MUERTES_DEP: Tasa de muertes por trabajador dependiente
   - TASA_PENSIONES_ACCIDENTE: Tasa de pensiones por accidente
   - TASA_PENSIONES_ENFERMEDAD: Tasa de pensiones por enfermedad
   - TOTAL_PENSIONES: Total de pensiones (accidentes + enfermedades)
   - TOTAL_INCAPACIDADES: Total de incapacidades (accidentes + enfermedades)
   - RATIO_INDEP_DEP: Ratio de trabajadores independientes vs dependientes
   - TOTAL_TRABAJADORES: Total de trabajadores (dependientes + independientes)

## Archivos Generados
- dataset_curado.csv: Dataset curado en formato CSV
- dataset_curado.parquet: Dataset curado en formato Parquet
- label_encoders.pkl: Diccionario de LabelEncoders para variables categóricas
- scaler.pkl: StandardScaler para variables numéricas

## Próximos Pasos
1. Implementar modelos de predicción (notebook 03_Modelos_Prediccion.ipynb)
2. Analizar patrones y anomalías (notebook 04_Analisis_Patrones.ipynb)
3. Generar explicabilidad de modelos (notebook 05_Explicabilidad_Resultados.ipynb)
