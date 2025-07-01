import math

# Paramètres d'entrée
E = 141e3  # Module de Young en N/mm²
width = 60.0  # Largeur en mm
thickness = 0.22  # Épaisseur en mm
length = 104.9  # Longueur en mm
n = 1  # Facteur de correction

# Calcul de la rigidité théorique
kth = (314 * n * E * width * thickness ** 3) / (24 * length ** 3)

# Valeur de référence pour k
k_ref = 0.914470347

# Calcul du ratio
ratio = kth / k_ref

# Calcul de la performance de référence (k × rs²)
# Supposons que rs soit calculé comme |Dx| où Dx est une fonction de la longueur
Dx = -2 * (length / math.pi)  # Exemple de calcul de Dx
rs = abs(Dx)
performance_ref = k_ref * (rs ** 2)

# Calcul de la performance actuelle
performance = kth * (rs ** 2)

# Calcul du perf_ratio
perf_ratio = performance / performance_ref

print("Rigidité théorique (kth):", kth)
print("Ratio (kth / k_ref):", ratio)
print("Performance de référence (k × rs²):", performance_ref)
print("Performance actuelle (k × rs²):", performance)
print("Perf Ratio:", perf_ratio)
print('\n==========================================================\n')


# Calcul de la rigidité théorique
kth = (math.pi ** 2 * E * width * thickness ** 3) / (6 * length ** 3)

# Calcul de Dx et rs
Dx = -2 * (length / math.pi)  # Exemple de calcul de Dx
rs = abs(Dx)

# Calcul de la performance de référence (k × rs²)
# Supposons que k_ref soit calculé à partir d'une configuration de référence
L_ref = 105.25
h_ref = 0.24
w_ref = 45.0
E_ref = 131e3
k_ref = (math.pi ** 2 * E_ref * w_ref * h_ref ** 3) / (6 * L_ref ** 3)
performance_ref = k_ref * (rs ** 2)

# Calcul de la performance actuelle
performance = kth * (rs ** 2)

# Calcul du perf_ratio
perf_ratio = performance / performance_ref

print("Rigidité théorique (kth):", kth)
print("Performance de référence (k × rs²):", performance_ref)
print("Performance actuelle (k × rs²):", performance)
print("Perf Ratio:", perf_ratio)
