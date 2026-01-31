#!/usr/bin/env python3
"""
Script de migration pour convertir un modèle dépendant de CuPy vers NumPy.

Ce script charge un modèle sauvegardé avec des arrays CuPy et le convertit
pour utiliser uniquement NumPy, permettant l'inférence sur des systèmes sans GPU.

Usage:
    python migrate_model.py <input_model.pkl> <output_model.pkl>
    
    Ou en utilisant les chemins par défaut:
    python migrate_model.py
"""

import sys
import pickle
from pathlib import Path
from typing import Any, Union
import argparse


def convert_cupy_to_numpy(obj: Any) -> Any:
    """
    Convertit récursivement tous les arrays CuPy en NumPy dans une structure de données.
    
    Args:
        obj: Objet à convertir (peut être un dict, list, tuple, array, etc.)
    
    Returns:
        L'objet avec tous les arrays CuPy convertis en NumPy
    """
    try:
        import cupy as cp
        has_cupy = True
    except ImportError:
        has_cupy = False
        print("WARNING: CuPy n'est pas installé. Le script tentera quand même de convertir.")
    
    import numpy as np
    
    if has_cupy and isinstance(obj, cp.ndarray):
        print(f"  Conversion array CuPy -> NumPy: shape={obj.shape}, dtype={obj.dtype}")
        return cp.asnumpy(obj)

    if isinstance(obj, dict):
        return {key: convert_cupy_to_numpy(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [convert_cupy_to_numpy(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(convert_cupy_to_numpy(item) for item in obj)

    return obj


def migrate_model(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Migre un modèle de CuPy vers NumPy.
    
    Args:
        input_path: Chemin vers le fichier .pkl du modèle avec CuPy
        output_path: Chemin où sauvegarder le modèle converti en NumPy
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Vérifier que le fichier d'entrée existe
    if not input_path.exists():
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")
    
    print(f"Chargement du modèle depuis: {input_path}")
    
    # Charger le modèle
    try:
        with open(input_path, "rb") as f:
            model_state = pickle.load(f)
        print(f"✓ Modèle chargé avec succès")
    except Exception as e:
        print(f"✗ Erreur lors du chargement: {e}")
        raise
    
    # Convertir CuPy -> NumPy
    print("\nConversion CuPy -> NumPy en cours...")
    converted_state = convert_cupy_to_numpy(model_state)
    print("✓ Conversion terminée")
    
    # Sauvegarder le modèle converti
    print(f"\nSauvegarde du modèle converti vers: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, "wb") as f:
            pickle.dump(converted_state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Modèle sauvegardé avec succès")
    except Exception as e:
        print(f"✗ Erreur lors de la sauvegarde: {e}")
        raise
    
    # Afficher les statistiques
    input_size = input_path.stat().st_size / (1024 * 1024)  # en Mo
    output_size = output_path.stat().st_size / (1024 * 1024)  # en Mo
    
    print(f"\n{'='*60}")
    print(f"Migration réussie!")
    print(f"  Taille originale:  {input_size:.2f} Mo")
    print(f"  Taille convertie:  {output_size:.2f} Mo")
    print(f"{'='*60}")


def verify_model(model_path: Union[str, Path]) -> None:
    """
    Vérifie qu'un modèle peut être chargé sans CuPy.
    
    Args:
        model_path: Chemin vers le fichier .pkl du modèle à vérifier
    """
    model_path = Path(model_path)
    
    print(f"\nVérification du modèle: {model_path}")
    
    # Tenter de charger sans cupy
    try:
        import sys
        # Bloquer temporairement l'import de cupy pour tester
        import builtins
        real_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name.startswith('cupy'):
                raise ImportError(f"CuPy bloqué pour test: {name}")
            return real_import(name, *args, **kwargs)
        
        builtins.__import__ = mock_import
        
        with open(model_path, "rb") as f:
            model_state = pickle.load(f)
        
        builtins.__import__ = real_import
        
        print("✓ Le modèle peut être chargé sans CuPy!")
        return True
        
    except Exception as e:
        builtins.__import__ = real_import
        print(f"✗ Erreur lors de la vérification: {e}")
        return False


def main():
    """Point d'entrée principal du script."""
    
    parser = argparse.ArgumentParser(
        description="Migrer un modèle de CuPy vers NumPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Utiliser les chemins par défaut
  python migrate_model.py
  
  # Spécifier les chemins
  python migrate_model.py save/backup_latest.pkl save/backup_latest_numpy.pkl
  
  # Vérifier un modèle converti
  python migrate_model.py --verify save/backup_latest_numpy.pkl
        """
    )
    
    parser.add_argument(
        "input_path",
        nargs="?",
        default="save/backup_latest.pkl",
        help="Chemin vers le modèle CuPy (défaut: save/backup_latest.pkl)"
    )
    
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Chemin de sortie pour le modèle NumPy (défaut: input_path avec suffixe '_numpy')"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Vérifier que le modèle de sortie peut être chargé sans CuPy"
    )
    
    args = parser.parse_args()

    if args.output_path is None:
        input_path = Path(args.input_path)
        output_path = input_path.parent / f"{input_path.stem}_numpy{input_path.suffix}"
    else:
        output_path = Path(args.output_path)
    
    input_path = Path(args.input_path)
    
    print("="*60)
    print("MIGRATION MODÈLE: CuPy -> NumPy")
    print("="*60)

    try:
        migrate_model(input_path, output_path)

        if args.verify:
            verify_model(output_path)
            
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERREUR: {e}")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
