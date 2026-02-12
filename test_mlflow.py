import mlflow

# 1. On démarre une "run" (une expérience)
with mlflow.start_run():

    # Simule des paramètres (ex: ce que tu testeras dans le Projet 6)
    n_estimators = 100
    learning_rate = 0.1

    # 2. On loggue les paramètres (les inputs)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)

    # 3. On loggue une métrique (le résultat)
    accuracy = 0.85
    mlflow.log_metric("accuracy", accuracy)

    print("Run enregistrée avec succès !")
