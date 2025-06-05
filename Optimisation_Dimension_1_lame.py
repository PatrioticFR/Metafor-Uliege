import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Matériaux et modules d'élasticité (en N/mm^2)
MATERIAL_PROPERTIES = {
    "Invar36": 141000,
    "Be-Cu": 131000
}

# Constante
n = 1

# Ranges pour les paramètres
b_range = np.arange(10, 45.0, 0.1)
h_range = np.arange(0.05, 0.30, 0.01)
L_range = np.arange(90, 110.0, 0.1)


# Fonction pour calculer k
def calculate_k(E, b, h, L):
    return (314 * n * E * b * h ** 3) / (24 * L ** 3)


# Fonction principale pour exécuter le programme selon le matériau
def run_simulation(material):
    if material not in MATERIAL_PROPERTIES:
        raise ValueError(f"Matériau non reconnu. Choisir parmi {list(MATERIAL_PROPERTIES.keys())}.")

    E = MATERIAL_PROPERTIES[material]
    results = []

    # Calcul de k et filtrage des résultats
    for b in b_range:
        for h in h_range:
            for L in L_range:
                k = calculate_k(E, b, h, L) #Reference: 0.914470347
                if 0.91445 <= k <= 0.91450 :
                    # Remplacer l'ordre (b, h, L, k) par (L, h, b, k)
                    results.append((L, h, b, k))

    # Affichage des résultats
    print(f"Résultats pour {material}:")
    for result in results:
        print(f"Longueur lame: {result[0]:.2f} mm, Épaisseur lame: {result[1]:.2f} mm, "
              f"Largeur lame: {result[2]:.2f} mm, Constante de raideur: {result[3]:.4f}")

    if not results:
        print("Aucune combinaison ne satisfait la condition sur k.")
        return

    # Clustering pour sélection des cas de test optimaux
    results_array = np.array(results)
    scaler = StandardScaler()
    results_scaled = scaler.fit_transform(results_array[:, :3])  # L, h, b

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(results_scaled)

    cluster_indices = [
        np.argmin(np.linalg.norm(results_scaled - center, axis=1))
        for center in kmeans.cluster_centers_
    ]

    diverse_test_cases = results_array[cluster_indices]

    # Affichage des cas optimaux
    print("\nCas de test optimaux :")
    for case in diverse_test_cases:
        print(f"Longueur lame: {case[0]:.2f} mm, Épaisseur lame: {case[1]:.2f} mm, "
              f"Largeur lame: {case[2]:.2f} mm, Constante de raideur: {case[3]:.4f}")

    # Affichage pour Excel - En-têtes
    print(f"\n=== DONNÉES POUR EXCEL - {material} ===")
    print("Longueur_lame\tEpaisseur_lame\tLargeur_lame\tConstante_raideur")

    # Affichage de tous les résultats (format Excel)
    print("\nTous les résultats (séparés par tabulations) :")
    for result in results:
        print(f"{result[0]:.2f}\t{result[1]:.2f}\t{result[2]:.2f}\t{result[3]:.4f}")

    # Affichage des cas optimaux (format Excel)
    print(f"\nCas de test optimaux (séparés par tabulations) :")
    print("Longueur_lame\tEpaisseur_lame\tLargeur_lame\tConstante_raideur")
    for case in diverse_test_cases:
        print(f"{case[0]:.2f}\t{case[1]:.2f}\t{case[2]:.2f}\t{case[3]:.4f}")


# Exemple d'appel
if __name__ == "__main__":
    run_simulation("Be-Cu")  # Remplace par "Be-Cu" ou "Invar36" pour tester l'autre matériau
