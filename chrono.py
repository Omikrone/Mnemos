# Fonction décoratrice pour mesurer le temps d'exécution d'une fonction
import time

def chrono(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Enregistrer le temps de début
        result = func(*args, **kwargs)  # Appeler la fonction décorée
        end_time = time.time()  # Enregistrer le temps de fin
        elapsed_time = end_time - start_time  # Calculer le temps écoulé
        print(f"Temps d'exécution de {func.__name__}: {elapsed_time:.4f} secondes")
        return result  # Retourner le résultat de la fonction
    return wrapper