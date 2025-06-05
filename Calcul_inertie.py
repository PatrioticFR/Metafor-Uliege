import numpy as np

# === DONNÉES DE BASE ===
print("=== CALCUL DU MOMENT D'INERTIE AVEC MÉTHODE CORRECTE ===\n")

# Dimensions géométriques (mm)
H = 3.875       # épaisseur verticale du rod gauche
D = 39.99       # hauteur du bloc masse
d = 13.96       # longueur du bloc masse
l = 79.2        # longueur totale du rod
r = 7.0         # longueur du rod droit
R = H           # largeur du rod droit (= H)
w = 63.0        # profondeur du système (largeur latérale)

# Déduire la longueur du rod gauche
h = l - r - d   # mm

# Propriétés matériau
rho = 7.85e-6   # kg/mm³ (densité acier)

print(f"Dimensions calculées:")
print(f"  h (rod gauche): {h:.2f} mm")
print(f"  d (bloc masse): {d:.2f} mm")
print(f"  r (rod droit): {r:.2f} mm")
print(f"  w (profondeur): {w:.2f} mm")
print(f"  ρ (densité): {rho:.2e} kg/mm³\n")

# === MÉTHODE CORRECTE: VOLUMES/MASSES + THÉORÈME DE HUYGENS ===

# 1. Rod gauche (rectangle: h × H × w)
print("=== CALCUL PAR COMPOSANTS (MÉTHODE CORRECTE) ===")

# Rod gauche
V_rod_g = h * H * w  # volume
m_rod_g = rho * V_rod_g  # masse
x_rod_g = h / 2  # distance du centre de masse à l'origine

# Inertie propre autour du centre de masse (formule plaque rectangulaire)
# Pour rotation autour de Z: I_g = m * (L² + T²) / 12
# L = h (longueur), T = H (largeur)
I_g_rod_g = m_rod_g * (h**2 + H**2) / 12

# Théorème de Huygens: I_total = I_g + m * x²
I_rod_g = I_g_rod_g + m_rod_g * x_rod_g**2

print(f"Rod gauche:")
print(f"  Volume: {V_rod_g:.2f} mm³")
print(f"  Masse: {m_rod_g:.4f} kg")
print(f"  Distance centre de masse: {x_rod_g:.2f} mm")
print(f"  I_g (inertie propre): {I_g_rod_g:.2f} kg·mm²")
print(f"  I_total: {I_rod_g:.2f} kg·mm²\n")

# 2. Bloc masse (rectangle: d × D × w)
V_mass = d * D * w
m_mass = rho * V_mass
x_mass = h + d / 2  # distance du centre de masse à l'origine

# Inertie propre: L = d, T = D
I_g_mass = m_mass * (d**2 + D**2) / 12
I_mass = I_g_mass + m_mass * x_mass**2

print(f"Bloc masse:")
print(f"  Volume: {V_mass:.2f} mm³")
print(f"  Masse: {m_mass:.4f} kg")
print(f"  Distance centre de masse: {x_mass:.2f} mm")
print(f"  I_g (inertie propre): {I_g_mass:.2f} kg·mm²")
print(f"  I_total: {I_mass:.2f} kg·mm²\n")

# 3. Rod droit (rectangle: r × R × w)
V_rod_d = r * R * w
m_rod_d = rho * V_rod_d
x_rod_d = h + d + r / 2  # distance du centre de masse à l'origine

# Inertie propre: L = r, T = R
I_g_rod_d = m_rod_d * (r**2 + R**2) / 12
I_rod_d = I_g_rod_d + m_rod_d * x_rod_d**2

print(f"Rod droit:")
print(f"  Volume: {V_rod_d:.2f} mm³")
print(f"  Masse: {m_rod_d:.4f} kg")
print(f"  Distance centre de masse: {x_rod_d:.2f} mm")
print(f"  I_g (inertie propre): {I_g_rod_d:.2f} kg·mm²")
print(f"  I_total: {I_rod_d:.2f} kg·mm²\n")

# === RÉSULTATS FINAUX ===
print("=== RÉSULTATS FINAUX ===")

# Inertie totale (méthode correcte)
I_total_correct = I_rod_g + I_mass + I_rod_d

# Masse totale
m_total = m_rod_g + m_mass + m_rod_d

print(f"Masse totale: {m_total:.4f} kg")
print(f"Inertie totale (méthode correcte): {I_total_correct:.2f} kg·mm²")

# === COMPARAISON AVEC LA FORMULE INCORRECTE DU PAPIER ===
print("\n=== COMPARAISON AVEC PAPIER (FORMULE INCORRECTE) ===")

rho_w = rho * w  # densité surfacique

# Formule incorrecte du Papier
I_Papier_incorrect = rho_w * (
    h**3 * H / 4 +
    d * D * (h + d/2)**2 +
    r * R * (h + d + r/2)**2
)

print(f"Formule Papier (incorrecte): {I_Papier_incorrect:.2f} kg·mm²")

# === MÉTHODE ALTERNATIVE (COMME SUR VOTRE PHOTO) ===
print("\n=== MÉTHODE ALTERNATIVE (APPROCHE BARRE) ===")

# Cette méthode traite chaque section comme une "barre" avec inertie I = ρwL*H³/3 + ρwL*H*d²
# où d est la distance du centre de gravité de la barre à l'axe de rotation

# Rod gauche - traité comme barre verticale
# Centre de gravité à x = h/2, hauteur H dans direction perpendiculaire à la rotation
I_barre_rod_g = rho * w * h * (H**3/3 + H * (h/2)**2)

# Bloc masse - traité comme barre horizontale
# Centre de gravité à x = h + d/2, "hauteur" D dans direction perpendiculaire
I_barre_mass = rho * w * d * (D**3/3 + D * (h + d/2)**2)

# Rod droit - traité comme barre verticale
# Centre de gravité à x = h + d + r/2, hauteur R dans direction perpendiculaire
I_barre_rod_d = rho * w * r * (R**3/3 + R * (h + d + r/2)**2)

# Total méthode barre
I_total_barre = I_barre_rod_g + I_barre_mass + I_barre_rod_d

print(f"Rod gauche (barre): {I_barre_rod_g:.2f} kg·mm²")
print(f"Bloc masse (barre): {I_barre_mass:.2f} kg·mm²")
print(f"Rod droit (barre): {I_barre_rod_d:.2f} kg·mm²")
print(f"Total (méthode barre): {I_total_barre:.2f} kg·mm²")

# === COMPARAISON AVEC LA RÉFÉRENCE ===
print("\n=== COMPARAISON AVEC LA RÉFÉRENCE ===")

I_ref = 1486.0  # kg·mm² (valeur de référence)

erreur_correcte = (I_total_correct - I_ref) / I_ref * 100
erreur_Papier = (I_Papier_incorrect - I_ref) / I_ref * 100
erreur_barre = (I_total_barre - I_ref) / I_ref * 100

print(f"Valeur de référence: {I_ref:.2f} kg·mm²")
print(f"Méthode plaques (Huygens): {I_total_correct:.2f} kg·mm² (erreur: {erreur_correcte:+.2f}%)")
print(f"Méthode barres (photo): {I_total_barre:.2f} kg·mm² (erreur: {erreur_barre:+.2f}%)")
print(f"Papier (incorrect): {I_Papier_incorrect:.2f} kg·mm² (erreur: {erreur_Papier:+.2f}%)")

# === TABLEAU RÉCAPITULATIF ===
print("\n=== TABLEAU RÉCAPITULATIF ===")
print("Élément".ljust(15) + "m (kg)".ljust(10) + "x (mm)".ljust(10) + "I_g (kg·mm²)".ljust(15) + "I_total (kg·mm²)")
print("-" * 70)
print(f"Rod gauche".ljust(15) + f"{m_rod_g:.4f}".ljust(10) + f"{x_rod_g:.2f}".ljust(10) + f"{I_g_rod_g:.2f}".ljust(15) + f"{I_rod_g:.2f}")
print(f"Bloc masse".ljust(15) + f"{m_mass:.4f}".ljust(10) + f"{x_mass:.2f}".ljust(10) + f"{I_g_mass:.2f}".ljust(15) + f"{I_mass:.2f}")
print(f"Rod droit".ljust(15) + f"{m_rod_d:.4f}".ljust(10) + f"{x_rod_d:.2f}".ljust(10) + f"{I_g_rod_d:.2f}".ljust(15) + f"{I_rod_d:.2f}")
print("-" * 70)
print(f"TOTAL".ljust(15) + f"{m_total:.4f}".ljust(10) + "".ljust(10) + "".ljust(15) + f"{I_total_correct:.2f}")

print(f"\n🎯 CONCLUSION:")
print(f"Méthode plaques (Huygens): {I_total_correct:.2f} kg·mm² avec {erreur_correcte:+.2f}% d'erreur")
print(f"Méthode barres (photo): {I_total_barre:.2f} kg·mm² avec {erreur_barre:+.2f}% d'erreur")
print(f"Papier (incorrect): {I_Papier_incorrect:.2f} kg·mm² avec {erreur_Papier:+.2f}% d'erreur")
print(f"\nLa méthode des barres semble {'plus' if abs(erreur_barre) < abs(erreur_correcte) else 'moins'} précise que celle des plaques.")
