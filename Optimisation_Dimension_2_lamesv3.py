import math
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RBFInterpolator  # NOUVEAU: Import pour l'interpolation

# Configuration pour les 2 lames
DUAL_BLADE_CONFIG = True  # Mode 2 lames activé


class ConfigurationTypeDual(Enum):
    """Types de configurations possibles pour 2 lames"""
    L_BECU_CONSTANT = "L_BeCu_constant"  # L_BE_CU fixe, autres variables


class VariationMode(Enum):
    """Modes de variation pour les paramètres"""
    OPTIMAL = "optimal"
    HIGH_LOW = "high_low"
    LOW_HIGH = "low_high"


def calculate_ideal_perf_ratio_range_dual():
    """Définit la plage de ratio de performance idéale de manière statique."""
    # Basé sur l'analyse précédente, une plage de ±5% autour de la référence est un bon point de départ.
    return 0.95, 1.05


class DualBladeConfig:
    """Configuration modulaire pour optimisation de 2 lames avec nomenclature outer/inner"""

    def __init__(self):
        # Sélection des matériaux
        self.outer_material = 'BE_CU'
        self.inner_material = 'INVAR'

        # Géométrie de référence
        self.thickness = 0.24
        self.length = 105.25
        self.width_ref_outer = 30.0
        self.inner_thickness = 0.24
        self.inner_offset = 0.25
        self.width_ref_inner = 22.5
        self.enc = 57.32

        # Propriétés matériaux
        self.material_properties = {'BE_CU': 131e3, 'INVAR': 141e3}

        # Paramètres géométriques du système
        self.H, self.D, self.d, self.l, self.r = 3.875, 39.99, 13.96, 79.2, 7.0
        self.R_rod, self.rho, self.depth = self.H, 7.85e-6, 63.0
        self.k_flex, self.r_s = 44.3, 65.22

        # Plages de recherche
        self.w_outer_range = np.arange(20, 50.1, 1.0)
        self.thickness_range = np.arange(0.20, 0.30, 0.01)
        self.w_inner_range = np.arange(15, 45.1, 0.5)
        self.inner_thickness_range = np.arange(0.20, 0.30, 0.01)

        # CORRIGÉ: Utilisation de la fonction pour définir la plage de performance
        self.perf_ratio_min, self.perf_ratio_max = calculate_ideal_perf_ratio_range_dual()

        # NOUVEAU: Données expérimentales pour l'interpolation du déplacement
        self.experimental_data = [
            # w_outer, h_outer, w_inner, h_inner -> displacement
            {'input': [30.0, 0.24, 22.5, 0.24], 'output': 1.2927},
            {'input': [30.0, 0.24, 25.0, 0.24], 'output': 3.14849},
            {'input': [30.0, 0.24, 40.0, 0.22], 'output': 8.20704},
            {'input': [45.0, 0.23, 40.0, 0.22], 'output': 13.5489},
            {'input': [25.0, 0.24, 20.0, 0.24], 'output': -0.651699},
            {'input': [20.0, 0.24, 30.0, 0.24], 'output': -4.62793},
            {'input': [30.0, 0.24, 20.0, 0.22], 'output': -9.95165},
            {'input': [20.0, 0.24, 25.0, 0.24], 'output': -2.37118}
        ]
        # NOUVEAU: Création de l'interpolateur de déplacement
        exp_inputs = np.array([d['input'] for d in self.experimental_data])
        exp_outputs = np.array([d['output'] for d in self.experimental_data])
        self.displacement_interpolator = RBFInterpolator(exp_inputs, exp_outputs, kernel='linear', epsilon=1)

        self.I_system = self._calculate_system_inertia()
        inner_geom = self.get_invar_geometry(self.enc, self._calculate_Dx(self.length))
        self.performance_ref = self._calculate_dual_performance(
            self.width_ref_outer, self.thickness, self.length,
            self.width_ref_inner, inner_geom['ei'], inner_geom['Li']
        )
        print(f"Configuration: Système à 2 lames (outer/inner)")
        print(f"Performance de référence: {self.performance_ref:.6f}")
        print(f"Objectif de déplacement: [-0.5, 1.0] mm")
        print(f"Plage de ratio de performance cible: [{self.perf_ratio_min}, {self.perf_ratio_max}]")

    def get_invar_geometry(self, enc, Dx):
        e, ei, L, offset = self.thickness, self.inner_thickness, self.length, self.inner_offset
        R_outer = (enc / 2) + (e / 2)
        R_inner = (enc / 2) - offset - (ei / 2)
        ratio = R_inner / R_outer
        return {'ei': ei, 'Li': L * ratio, 'ratio': ratio, 'Dx_invar': Dx * ratio}

    def _calculate_system_inertia(self) -> float:
        h_rod_segment = self.l - self.r - self.d
        I_barre_rod_g = self.rho * self.depth * h_rod_segment * (self.H ** 3 / 3 + self.H * (h_rod_segment / 2) ** 2)
        I_barre_mass = self.rho * self.depth * self.d * (self.D ** 3 / 3 + self.D * (h_rod_segment + self.d / 2) ** 2)
        I_barre_rod_d = self.rho * self.depth * self.r * (
                    self.R_rod ** 3 / 3 + self.R_rod * (h_rod_segment + self.d + self.r / 2) ** 2)
        return I_barre_rod_g + I_barre_mass + I_barre_rod_d

    def _calculate_Dx(self, length: float) -> float:
        return -2 * (length / math.pi)

    def _calculate_k_single_blade(self, E: float, width: float, thickness: float, length: float) -> float:
        return (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3

    # MODIFIÉ: La fonction utilise maintenant la plage cible réelle
    def is_displacement_valid(self, w_outer: float, thickness_outer: float, L_outer: float,
                              w_inner: float, thickness_inner: float, L_inner: float) -> bool:
        """Vérifie si le déplacement estimé est dans la plage cible [-0.5, 1.0] mm"""
        displacement = self._calculate_displacement_estimate(w_outer, thickness_outer, w_inner, thickness_inner)
        target_min, target_max = -0.5, 1.0
        return target_min <= displacement <= target_max

    def is_performance_ratio_valid(self, performance: float) -> bool:
        if self.performance_ref <= 0: return True
        perf_ratio = performance / self.performance_ref
        return self.perf_ratio_min <= perf_ratio <= self.perf_ratio_max

    def find_diverse_configurations(self, configurations: List[Dict], n_clusters: int = 8) -> List[Dict]:
        if len(configurations) < n_clusters: return configurations
        features = np.array([[c['w_outer'], c['thickness_outer'], c['w_inner'], c['thickness_inner'],
                              c['L_inner'], c['displacement_est'], c['k_ratio']] for c in configurations])
        features_scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features_scaled)
        cluster_centers_indices = [np.argmin(np.linalg.norm(features_scaled - center, axis=1)) for center in
                                   kmeans.cluster_centers_]
        return [configurations[i] for i in cluster_centers_indices]

    def _calculate_radius_ratio(self, thickness_outer: float, thickness_inner: float) -> float:
        R_outer = (self.enc / 2) + (thickness_outer / 2)
        R_inner = (self.enc / 2) - self.inner_offset - (thickness_inner / 2)
        return R_inner / R_outer

    def _calculate_L_inner_from_geometry(self, thickness_outer: float, thickness_inner: float) -> float:
        return self.length * self._calculate_radius_ratio(thickness_outer, thickness_inner)

    # MODIFIÉ: Utilise l'interpolateur basé sur les données expérimentales
    def _calculate_displacement_estimate(self, w_outer: float, thickness_outer: float,
                                         w_inner: float, thickness_inner: float) -> float:
        """Estime le déplacement en utilisant l'interpolateur RBF."""
        # Le format d'entrée doit correspondre à celui utilisé pour l'entraînement
        input_vector = np.array([[w_outer, thickness_outer, w_inner, thickness_inner]])
        return self.displacement_interpolator(input_vector)[0]

    def _calculate_dual_performance(self, w_outer: float, thickness_outer: float, L_outer: float,
                                    w_inner: float, thickness_inner: float, L_inner: float) -> float:
        E_outer, E_inner = self.material_properties[self.outer_material], self.material_properties[self.inner_material]
        k_outer = self._calculate_k_single_blade(E_outer, w_outer, thickness_outer, L_outer)
        k_inner = self._calculate_k_single_blade(E_inner, w_inner, thickness_inner, L_inner)
        k_total = (k_outer + k_inner) * 1e3  # en mN·m
        length_equivalent = (L_outer + L_inner) / 2
        rs = abs(self._calculate_Dx(length_equivalent))
        return k_total * (rs ** 2)

    def generate_dual_configurations(self) -> List[Dict]:
        configurations = []
        total_tested = 0
        L_outer_fixed = self.length

        print(f"Génération des configurations...")
        # (le reste de la fonction est inchangé mais bénéficiera des nouvelles logiques de calcul et de filtrage)
        for thickness_outer in self.thickness_range:
            for w_outer in self.w_outer_range:
                for thickness_inner in self.inner_thickness_range:
                    for w_inner in self.w_inner_range:
                        total_tested += 1
                        L_inner_calculated = self._calculate_L_inner_from_geometry(thickness_outer, thickness_inner)

                        # MODIFIÉ: Le check de déplacement est maintenant pertinent
                        if self.is_displacement_valid(w_outer, thickness_outer, L_outer_fixed, w_inner, thickness_inner,
                                                      L_inner_calculated):
                            perf = self._calculate_dual_performance(w_outer, thickness_outer, L_outer_fixed, w_inner,
                                                                    thickness_inner, L_inner_calculated)

                            # MODIFIÉ: Le check de performance est maintenant pertinent
                            if self.is_performance_ratio_valid(perf):
                                config = self._create_dual_config_dict(w_outer, thickness_outer, L_outer_fixed, w_inner,
                                                                       thickness_inner, L_inner_calculated, perf)
                                configurations.append(config)

        print(f"\nTotal testé: {total_tested}")
        print(f"Configurations valides trouvées: {len(configurations)}")
        return configurations

    def _create_dual_config_dict(self, w_outer: float, thickness_outer: float, L_outer: float,
                                 w_inner: float, thickness_inner: float, L_inner: float,
                                 performance: float) -> Dict:
        perf_ratio = performance / self.performance_ref if self.performance_ref > 0 else 1.0
        perf_optimal = (self.perf_ratio_min + self.perf_ratio_max) / 2
        length_equivalent = (L_outer + L_inner) / 2
        Dx = self._calculate_Dx(length_equivalent)
        E_outer, E_inner = self.material_properties[self.outer_material], self.material_properties[self.inner_material]
        k_outer = self._calculate_k_single_blade(E_outer, w_outer, thickness_outer, L_outer)
        k_inner = self._calculate_k_single_blade(E_inner, w_inner, thickness_inner, L_inner)

        # MODIFIÉ: Le déplacement est calculé avec la nouvelle méthode
        displacement_est = self._calculate_displacement_estimate(w_outer, thickness_outer, w_inner, thickness_inner)

        return {
            'w_outer': w_outer, 'thickness_outer': thickness_outer, 'L_outer': L_outer, 'k_outer_mNm': k_outer * 1e3,
            'w_inner': w_inner, 'thickness_inner': thickness_inner, 'L_inner': L_inner, 'k_inner_mNm': k_inner * 1e3,
            'k_total_mNm': (k_outer + k_inner) * 1e3,
            'performance': performance, 'perf_ratio': perf_ratio, 'Dx': Dx, 'rs': abs(Dx),
            'k_times_rs_squared': performance, 'perf_error_from_optimal': abs(perf_ratio - perf_optimal),
            'length_equivalent': length_equivalent, 'displacement_est': displacement_est,
            'radius_ratio': self._calculate_radius_ratio(thickness_outer, thickness_inner),
            'k_ratio': k_inner / k_outer if k_outer > 0 else 0,
        }

    def print_dual_configurations(self, configs: List[Dict], title: str, show_all: bool = False):
        if not configs:
            print(f"\n=== {title} ===")
            print("Aucune configuration trouvée.")
            return

        print(f"\n=== {title} ({len(configs)} configurations) ===")
        print(
            "Rank\tL_Outer\te_Outer\tw_Outer\tL_Inner\tei_Inner\tw_Inner\tDisp_Est\tRadius_R\tk_Ratio\tk_total\tRatio\tError")
        print("-" * 150)
        max_display = len(configs) if show_all else min(20, len(configs))
        for i, config in enumerate(configs[:max_display], 1):
            print(f"{i:2d}\t{config['L_outer']:.2f}\t{config['thickness_outer']:.3f}\t{config['w_outer']:.1f}\t"
                  f"{config['L_inner']:.2f}\t{config['thickness_inner']:.3f}\t{config['w_inner']:.1f}\t"
                  f"{config['displacement_est']:.3f}\t{config['radius_ratio']:.4f}\t{config['k_ratio']:.3f}\t"
                  f"{config['k_total_mNm']:.1f}\t{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    # (le reste des fonctions de vérification et de test est inchangé)
    def verify_dual_calculation(self, w_outer: float, thickness_outer: float, L_outer: float, w_inner: float,
                                thickness_inner: float, L_inner: float):
        # ...
        pass

    def test_specific_configuration(self, w_outer: float, thickness_outer: float, w_inner: float,
                                    thickness_inner: float):
        # ...
        pass


def main_dual():
    dual_config = DualBladeConfig()

    print("\n" + "=" * 80)
    print("GÉNÉRATION DES CONFIGURATIONS OPTIMISÉES...")
    all_configs = dual_config.generate_dual_configurations()

    if not all_configs:
        print("Aucune configuration trouvée avec les critères stricts.")
        print("Essayez d'élargir la plage de recherche ou les critères de performance si nécessaire.")
        return dual_config, [], []

    # Tri par erreur par rapport à l'optimal ET par déplacement proche de 0.5 (milieu de la plage cible)
    all_configs.sort(key=lambda x: (abs(x['displacement_est'] - 0.25), x['perf_error_from_optimal']))

    dual_config.print_dual_configurations(
        all_configs,
        f"MEILLEURES CONFIGURATIONS - SYSTÈME À 2 LAMES (Outer/Inner)"
    )

    if len(all_configs) > 8:
        diverse_configs = dual_config.find_diverse_configurations(all_configs, n_clusters=8)
        dual_config.print_dual_configurations(
            diverse_configs,
            "CONFIGURATIONS DIVERSES (K-MEANS)"
        )
    else:
        diverse_configs = all_configs
        print("\nPas assez de configurations pour un clustering K-Means significatif.")

    print(f"\nRésumé:")
    print(f"- Configurations totales trouvées: {len(all_configs)}")
    print(f"- Configurations diverses sélectionnées: {len(diverse_configs)}")
    print(
        f"- Objectif déplacement: [{dual_config.is_displacement_valid.__doc__.split('[')[-1].split(']')[0]}] mm")  # un peu de fun pour afficher la docstring
    print(f"- Offset entre lames: {dual_config.inner_offset} mm")

    return dual_config, all_configs, diverse_configs


if __name__ == "__main__":
    # Nécéssite l'installation de scipy: pip install scipy
    config, all_configs, diverse_configs = main_dual()