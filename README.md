# *Algorithms-deployment-MLOps*

## 📌 *Objetivos del Proyecto*

*Este proyecto tiene como propósito reforzar y aplicar habilidades clave en ciencia de datos, ingeniería de Machine Learning y despliegue en producción. Los objetivos específicos son*:

-  ***Desarrollar habilidades prácticas en Machine Learning supervisado***, *trabajando con un problema de clasificación multiclase (vino)*.
-  ***Comprender el funcionamiento completo de un pipeline de ML***, *desde la preparación de datos hasta la evaluación y guardado del modelo*.
-  ***Transformar notebooks en scripts modulares y profesionales***, *facilitando la reutilización y mantenibilidad del código.*
-  ***Desplegar modelos como APIs RESTful*** *mediante FastAPI, permitiendo la interacción externa con el modelo*.
-  ***Integrar procesamiento de lenguaje natural (NLP)*** *usando pipelines de Hugging Face para tareas reales (análisis de sentimiento  y resumen)*.
-  ***Adoptar buenas prácticas de MLOps**, incluyendo versionado de modelos, separación de responsabilidades y documentación clara del flujo de trabajo*.

>  *Este proyecto simula un entorno real de desarrollo en producción, con enfoque en la automatización, escalabilidad y facilidad de despliegue*.

---

## 📁 *Estructura del proyecto*

```
├── FastAPI con HuggingFace/
|      ├── main_hf_api.py               # API FastAPI con Endopoints basicos
├── Clasificador + Despliegue API/
|      ├── Scripts/
|      |    ├── main.py                 # Entrena y guarda modelo + scaler
|      |    ├── utils.py                # Funciones reutilizables
|      |    └── main_api.py             # API FastAPI con predicción 
|      └── models/
│           ├── wine_model_multiclass.pkl
│           └── wine_scaler.pkl
├── requirements.txt                    # Dependencias del entorno
├── Documentacion.pdf                   # Documentacion con capturas como prueba del proceso    
└── README.md                           # Documentación del proyecto
````
---

## 🤖 *Tecnologías utilizadas*

- `scikit-learn` – *Entrenamiento y evaluación del modelo de clasificación*
- `joblib` – *Serialización del modelo y scaler*
- `FastAPI` – *Creación de la API REST*
- `Uvicorn` – *Servidor ASGI para despliegue local*
- `transformers` – *NLP con pipelines de Hugging Face*
- `matplotlib`, `pandas` – *Visualización y análisis de datos*
  
## ⚙️ *Cómo ejecutar el proyecto*
>  ***De notebooks a producción: de un modelo entrenado a una API funcional***

1. ***Clona este repositorio***

```bash
git clone https://github.com/tu_usuario/ProyectoVino.git
cd ProyectoVino
```
2. ***Instala las dependencias***
```bash
pip install -r requirements.txt
```
3. ***Entrena el modelo y guarda los archivos .pkl***
```bash
python main.py
```
***Esto generará:***
- *models/wine_model_multiclass.pkl*
- *models/wine_scaler.pkl*

4. ***Lanza la API***
```bash
uvicorn main_api:app --reload
````
*Ve a: Docs: http://127.0.0.1:8000/docs y encontrarás los  Endpoints principales*

---

## ⚠️ *Disclaimer*

*Este proyecto es un trabajo académico/práctico destinado a fines educativos y de aprendizaje. Los modelos y métodos implementados pueden no estar optimizados para producción en entornos críticos o comerciales sin adaptaciones adicionales. El uso del código y modelos es bajo la responsabilidad del usuario. No se garantiza la precisión absoluta ni la adecuación para casos específicos fuera del ámbito académico.*

