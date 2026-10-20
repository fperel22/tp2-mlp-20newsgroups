# TP2 - MLP en TNG: Clasificación de texto con 20 Newsgroups

Trabajo práctico de **Redes Neuronales en Bioingeniería** (ITBA). Continúa el
[TP1](https://github.com/fperel22/tp-softmax-20newsgroups): se reemplaza la regresión
softmax por un MLP en PyTorch (activaciones, profundidad, neuronas y optimizadores
configurables), con búsqueda de hiperparámetros en Optuna usando pruning.

## Cómo correrlo

1. Abrir `TP2_MLP_20Newsgroups.ipynb` en [Google Colab](https://colab.research.google.com/)
   con GPU (`Entorno de ejecución > Cambiar tipo de entorno de ejecución`).
2. Ejecutar las celdas en orden. La primera instala `mlflow` y `optuna`.
3. La búsqueda de 100 trials tardó unos 15 minutos en una GPU T4. El estudio se guarda
   en `optuna_tp2.db`, así que se puede retomar si se corta.

## Estructura del notebook

0–3. Configuración inicial, carga de datos, preprocesamiento configurable y representación TF-IDF (reutilizados del TP1).

4. Arquitectura del modelo MLP.

5. Entrenamiento del modelo.

6. Búsqueda de hiperparámetros con Optuna y aplicación de pruning.

7. Entrenamiento final con la mejor configuración.

8. Evaluación final del modelo.

9. Experimento de sobreajuste (overfitting).

10. Visualización de resultados y figuras.

11. Análisis e interpretación de los resultados.

## Resultados principales

- **Búsqueda de hiperparámetros:** se realizaron 100 trials, de los cuales 51 se completaron y 49 fueron podados. La mejor configuración fue una red de una capa oculta con 512 neuronas y activación GELU, optimizador RMSprop, batch_size de 128 y 10.000 features, con un val_acc de 0.7388.

- **Resultados en test:** el MLP obtuvo un accuracy de 0.6808, superando al softmax del TP1 (0.6569) y al softmax con los hiperparámetros del TP2 (0.6559). La incorporación de una capa oculta mejoró el resultado en aproximadamente 2.5 puntos porcentuales, aunque la comparación se realizó con una sola semilla.

- **Sobreajuste:** aparece antes a medida que aumentan el ancho y la profundidad de la red. El modelo de tres capas con 256 neuronas por capa es un ejemplo claro de este comportamiento. Además, a igual ancho, las redes más profundas tienden a obtener peores resultados.

- **Pruning:** permitió ahorrar unos 177 segundos de entrenamiento, lo que representa aproximadamente el 17 % del tiempo total estimado del proceso.

- **Reproducibilidad:** dos ejecuciones limpias con la misma semilla produjeron resultados idénticos, lo que confirma la reproducibilidad bajo las mismas condiciones de ejecución.
