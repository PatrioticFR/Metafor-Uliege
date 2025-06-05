import math
import numpy as np


class BladeConfigForTargetStabilization:
    """Configuration optimisée pour obtenir une stabilisation à 0.5mm"""

    def __init__(self):
        # Reference blade geometry (from TFE baseline)
        self.thickness_ref = 0.24  # reference thickness (h)
        self.length_ref = 105.25  # reference length (L) - FIXE
        self.width_ref = 45.0  # reference width

        # Current blade geometry - L RESTE CONSTANT
        self.thickness = 0.24
        self.length = 105.25  # FIXE - ne change jamais
        self.width = 45.0

        # Material selection
        self.material = 'BE_CU'

        # Référence k (stabilisation à 0.9mm)
        self.k_ref = self.calculate_k_reference()

        # Système rod+masse (FIXE)
        self.enc_ref = 57.32  # CONSTANTE
        self.Dx_fixe = True

    def get_material_E(self):
        """Module d'Young du matériau (en N/mm²)"""
        if self.material == 'BE_CU':
            return 131e3
        elif self.material == 'INVAR':
            return 141e3
        else:  # STEEL
            return 210e3

    def calculate_k_reference(self):
        """Calcule la raideur k de référence (stabilisation à 0.9mm)"""
        E = self.get_material_E()
        return (math.pi ** 2 * E * self.width_ref * self.thickness_ref ** 3) / (6 * self.length_ref ** 3)

    def calculate_k(self, width, thickness):
        """Calcule k pour des dimensions données (longueur fixe)"""
        E = self.get_material_E()
        return (math.pi ** 2 * E * width * thickness ** 3) / (6 * self.length ** 3)

    def estimate_stabilization_position(self, k_value):
        """
        Estimation grossière de la position de stabilisation basée sur k
        Relation empirique basée sur vos observations :
        - k_ref = 0.914470347 → stabilisation ≈ 0.9mm
        - k_reduced ≈ 0.7 (-23%) → stabilisation ≈ -10mm
        """
        k_ratio = k_value / self.k_ref

        # Relation empirique (à ajuster selon vos tests)
        # Approximation linéaire basée sur vos deux points de données
        stabilization_ref = 0.9  # mm

        if k_ratio >= 1.0:
            # Pour k >= k_ref, relation approximativement linéaire
            return stabilization_ref + (k_ratio - 1.0) * 20  # pente estimée
        else:
            # Pour k < k_ref, relation non-linéaire observée
            # Point connu: k_ratio ≈ 0.77 → stabilisation ≈ -10mm
            # Interpolation entre (1.0, 0.9) et (0.77, -10.0)
            return stabilization_ref + (k_ratio - 1.0) * ((0.9 - (-10.0)) / (1.0 - 0.77))

    def find_k_for_target_stabilization(self, target_stabilization=0.5):
        """
        Trouve la valeur de k nécessaire pour obtenir la stabilisation cible
        """
        # Relation inverse basée sur l'estimation
        if target_stabilization >= 0.9:
            k_ratio = 1.0 + (target_stabilization - 0.9) / 20
        else:
            # Utilisation de la relation observée
            k_ratio = 1.0 + (target_stabilization - 0.9) / ((0.9 - (-10.0)) / (1.0 - 0.77))

        return self.k_ref * k_ratio

    def generate_configurations_for_target(self, target_stabilization=0.5, tolerance=0.02):
        """
        Génère des configurations (h, w) pour longueur fixe et stabilisation cible
        """
        target_k = self.find_k_for_target_stabilization(target_stabilization)

        print(f"=== RECHERCHE DE CONFIGURATIONS ===")
        print(f"Cible de stabilisation: {target_stabilization:.1f} mm")
        print(f"k_référence (0.9mm): {self.k_ref:.6f}")
        print(f"k_cible estimé: {target_k:.6f}")
        print(f"Réduction de k: {((target_k - self.k_ref) / self.k_ref * 100):+.2f}%")
        print(f"Longueur FIXE: {self.length:.2f} mm")
        print("=====================================\n")

        # Ranges de recherche
        h_range = np.arange(0.15, 0.35, 0.005)  # épaisseur
        w_range = np.arange(20.0, 70.0, 0.5)  # largeur

        configurations = []
        E = self.get_material_E()

        for h in h_range:
            for w in w_range:
                k_calc = (math.pi ** 2 * E * w * h ** 3) / (6 * self.length ** 3)

                # Vérifier si k est dans la tolérance
                if abs(k_calc - target_k) / target_k <= tolerance:
                    estimated_stabilization = self.estimate_stabilization_position(k_calc)
                    configurations.append({
                        'thickness': h,
                        'width': w,
                        'k': k_calc,
                        'k_ratio': k_calc / self.k_ref,
                        'estimated_stabilization': estimated_stabilization,
                        'k_error_percent': abs(k_calc - target_k) / target_k * 100
                    })

        # Trier par erreur sur k
        configurations.sort(key=lambda x: x['k_error_percent'])

        return configurations[:20]  # Top 20 configurations

    def analyze_current_configuration(self):
        """Analyse la configuration actuelle"""
        k_current = self.calculate_k(self.width, self.thickness)
        estimated_pos = self.estimate_stabilization_position(k_current)

        print(f"=== ANALYSE CONFIGURATION ACTUELLE ===")
        print(f"Dimensions: L={self.length:.2f} mm, h={self.thickness:.2f} mm, w={self.width:.2f} mm")
        print(f"k_actuel = {k_current:.6f}")
        print(f"k_référence = {self.k_ref:.6f}")
        print(f"Ratio k = {k_current / self.k_ref:.4f}")
        print(f"Stabilisation estimée: {estimated_pos:.1f} mm")
        print("======================================\n")

    def print_configurations(self, configurations, target_stabilization=0.5):
        """Affiche les configurations trouvées"""
        if not configurations:
            print("Aucune configuration trouvée dans les critères.")
            return

        print(f"=== TOP CONFIGURATIONS POUR STABILISATION À {target_stabilization:.1f}mm ===")
        print("Rang\th(mm)\tw(mm)\tk_value\t\tk_ratio\tStab_est(mm)\tErreur_k(%)")
        print("-" * 80)

        for i, config in enumerate(configurations, 1):
            print(f"{i:2d}\t{config['thickness']:.3f}\t{config['width']:.1f}\t"
                  f"{config['k']:.6f}\t{config['k_ratio']:.4f}\t"
                  f"{config['estimated_stabilization']:+.1f}\t\t{config['k_error_percent']:.3f}")

        print("=" * 80)

        # Configurations recommandées
        print(f"\n=== CONFIGURATIONS RECOMMANDÉES ===")
        for i in [0, len(configurations) // 4, len(configurations) // 2, -1]:
            if i < len(configurations):
                config = configurations[i]
                print(f"Option {i + 1}: h={config['thickness']:.3f}mm, w={config['width']:.1f}mm")
                print(f"  → k={config['k']:.6f} (ratio={config['k_ratio']:.4f})")
                print(f"  → Stabilisation estimée: {config['estimated_stabilization']:+.1f}mm")
                print()

    def get_physical_clamping_parameters(self):
        """Parameters de serrage (enc FIXE, Dx selon géométrie)"""
        R = self.length / math.pi
        enc_opt = self.enc_ref  # FIXE - propriété du rod+masse

        if self.Dx_fixe:
            Dx_opt = -67.5227
        else:
            Dx_opt = -2 * R

        return Dx_opt, 0.0, enc_opt


def find_optimal_configurations_for_stabilization(target_stabilization=0.5):
    """Fonction principale pour trouver les configurations optimales"""

    blade_config = BladeConfigForTargetStabilization()

    # Analyse de la configuration de référence
    print("CONFIGURATION DE RÉFÉRENCE:")
    blade_config.analyze_current_configuration()

    # Recherche des configurations pour la cible
    configurations = blade_config.generate_configurations_for_target(
        target_stabilization=target_stabilization,
        tolerance=0.01  # 1% de tolérance sur k
    )

    # Affichage des résultats
    blade_config.print_configurations(configurations, target_stabilization)

    return configurations


# Tests pour différentes cibles
if __name__ == "__main__":
    print("RECHERCHE DE CONFIGURATIONS POUR DIFFÉRENTES STABILISATIONS\n")

    # Test pour 0.5mm
    configs_05 = find_optimal_configurations_for_stabilization(0.5)

    print("\n" + "=" * 60 + "\n")

    # Test pour 0.3mm
    configs_03 = find_optimal_configurations_for_stabilization(0.3)

    print("\n" + "=" * 60 + "\n")

    # Test pour 0.7mm
    configs_07 = find_optimal_configurations_for_stabilization(0.7)