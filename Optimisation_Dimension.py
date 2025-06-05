import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Constants
E_CuBe = 131000  # N/mm^2
E_Invar36 = 141000  # N/mm^2
L_CuBe = 105.25  # mm
n = 1  # constant
enc = 57.32  # mm
decalage = 0.1  # mm

# Fixed thickness for CuBe
h_CuBe = 0.24  # mm

# Ranges for parameters
b_CuBe_range = np.arange(10, 45.1, 0.5)  # New variable
b_Invar_range = np.arange(10, 45.1, 0.5)
h_Invar_range = np.arange(0.05, 0.31, 0.01)

# Function to calculate L_Invar
def calculate_L_Invar(h_CuBe, h_Invar):
    R_BeCu = enc / 2 + h_CuBe / 2
    R_Invar = (enc / 2 - decalage - h_Invar / 2)
    return L_CuBe * (R_Invar / R_BeCu)

# Function to calculate k for a given material
def calculate_k(E, b, h, L):
    return (314 * n * E * b * h**3) / (24 * L**3)

# Collect results
results = []

for b_Invar in b_Invar_range:
    for h_Invar in h_Invar_range:
        for b_CuBe in b_CuBe_range:
            L_Invar = calculate_L_Invar(h_CuBe, h_Invar)

            # Calculate k for Cu-Be and Invar36
            k_CuBe = calculate_k(E_CuBe, b_CuBe, h_CuBe, L_CuBe)
            k_Invar36 = calculate_k(E_Invar36, b_Invar, h_Invar, L_Invar)

            # Effective k
            k_effective = k_CuBe + k_Invar36

            # Check if k_effective is within the desired range
            if 0.84 <= k_effective <= 0.85:
                results.append((L_Invar, h_Invar, b_Invar, b_CuBe, k_effective))

# Print all results
print("All Results:")
for result in results:
    print(f"L_Invar: {result[0]:.2f} mm, h_Invar: {result[1]:.2f} mm, b_Invar: {result[2]:.2f} mm, b_CuBe: {result[3]:.2f} mm, k_effective: {result[4]:.4f}")

# Convert results to a numpy array for clustering
results_array = np.array(results)

# Scale the features
scaler = StandardScaler()
results_array_scaled = scaler.fit_transform(results_array[:, :4])

# Use K-Means to find 6 diverse test cases
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(results_array_scaled)

# Get the indices of the cluster centers
cluster_centers_indices = [np.argmin(np.linalg.norm(results_array_scaled - center, axis=1)) for center in kmeans.cluster_centers_]

# Extract the most diverse test cases
diverse_test_cases = results_array[cluster_centers_indices]

# Print the most optimal test cases
print("\nMost Optimal Test Cases:")
for test_case in diverse_test_cases:
    print(f"L_Invar: {test_case[0]:.2f} mm, h_Invar: {test_case[1]:.2f} mm, b_Invar: {test_case[2]:.2f} mm, b_CuBe: {test_case[3]:.2f} mm, k_effective: {test_case[4]:.4f}")

# Affichage pour Excel
print("\n=== DONNÉES POUR EXCEL - SIMULATION BIMÉTAL ===")
print("Longueur_lame_Invar\tEpaisseur_lame_Invar\tLargeur_lame_Invar\tLargeur_lame_BeCu\tConstante_raideur_objectif_0.844")

# Affichage de tous les résultats (format Excel)
print("\nTous les résultats (séparés par tabulations) :")
for result in results:
    print(f"{result[0]:.2f}\t{result[1]:.2f}\t{result[2]:.2f}\t{result[3]:.2f}\t{result[4]:.4f}")

# Affichage des cas optimaux (format Excel)
print(f"\nCas de test optimaux (séparés par tabulations) :")
print("Longueur_lame_Invar\tEpaisseur_lame_Invar\tLargeur_lame_Invar\tLargeur_lame_BeCu\tConstante_raideur_objectif_0.844")
for test_case in diverse_test_cases:
    print(f"{test_case[0]:.2f}\t{test_case[1]:.2f}\t{test_case[2]:.2f}\t{test_case[3]:.2f}\t{test_case[4]:.4f}")
