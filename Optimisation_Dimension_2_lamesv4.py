import math
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

# Configuration pour les 2 lames
DUAL_BLADE_CONFIG = True


class ConfigurationTypeDual(Enum):
    """Types de configurations possibles pour 2 lames"""
    L_BECU_CONSTANT = "L_BeCu_constant"


class VariationMode(Enum):
    """Modes de variation pour les paramètres"""
    OPTIMAL = "optimal"
    HIGH_LOW = "high_low"
    LOW_HIGH = "low_high"


def calculate_ideal_perf_ratio_range_dual():
    """Définit la plage de ratio de performance idéale de manière statique."""
    return 0.95, 1.05


class DualBladeConfig:
    """Configuration modulaire pour optimisation de 2 lames avec nomenclature outer/inner"""

    def __init__(self, use_strict_criteria=True):
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

        # Plages de recherche étendues pour plus de diversité
        self.w_outer_range = np.arange(20, 50.1, 1.0)
        self.thickness_range = np.arange(0.20, 0.30, 0.01)
        self.w_inner_range = np.arange(15, 45.1, 0.5)
        self.inner_thickness_range = np.arange(0.20, 0.30, 0.01)

        # Critères de sélection
        self.use_strict_criteria = use_strict_criteria
        self.displacement_target = (-0.5, 1.0)  # mm
        self.perf_ratio_min, self.perf_ratio_max = calculate_ideal_perf_ratio_range_dual()

        # Données expérimentales améliorées avec plus de points
        self.experimental_data = [
            {'input': [30.0, 0.24, 22.5, 0.24], 'output': 1.2927},
            {'input': [30.0, 0.24, 25.0, 0.24], 'output': 3.14849},
            {'input': [30.0, 0.24, 40.0, 0.22], 'output': 8.20704},
            {'input': [45.0, 0.23, 40.0, 0.22], 'output': 13.5489},
            {'input': [25.0, 0.24, 20.0, 0.24], 'output': -0.651699},
            {'input': [20.0, 0.24, 30.0, 0.24], 'output': -4.62793},
            {'input': [30.0, 0.24, 20.0, 0.22], 'output': -9.95165},
            {'input': [20.0, 0.24, 25.0, 0.24], 'output': -2.37118}
        ]

        # Création de l'interpolateur avec paramètres optimisés
        exp_inputs = np.array([d['input'] for d in self.experimental_data])
        exp_outputs = np.array([d['output'] for d in self.experimental_data])

        # Utilisation d'un kernel plus robuste
        self.displacement_interpolator = RBFInterpolator(
            exp_inputs, exp_outputs,
            kernel='thin_plate_spline',  # Plus stable que 'linear'
            smoothing=0.1  # Légère régularisation
        )

        self.I_system = self._calculate_system_inertia()
        inner_geom = self.get_invar_geometry(self.enc, self._calculate_Dx(self.length))
        self.performance_ref = self._calculate_dual_performance(
            self.width_ref_outer, self.thickness, self.length,
            self.width_ref_inner, inner_geom['ei'], inner_geom['Li']
        )

        print(f"Configuration: Système à 2 lames (outer/inner)")
        print(f"Performance de référence: {self.performance_ref:.6f}")
        print(f"Objectif de déplacement: {self.displacement_target} mm")
        print(f"Plage de ratio de performance: [{self.perf_ratio_min}, {self.perf_ratio_max}]")
        print(f"Critères stricts: {self.use_strict_criteria}")

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

    def is_displacement_valid(self, displacement: float) -> bool:
        """Vérifie si le déplacement est dans la plage cible"""
        return self.displacement_target[0] <= displacement <= self.displacement_target[1]

    def is_performance_ratio_valid(self, performance: float) -> bool:
        if self.performance_ref <= 0:
            return True
        perf_ratio = performance / self.performance_ref
        return self.perf_ratio_min <= perf_ratio <= self.perf_ratio_max

    def _calculate_radius_ratio(self, thickness_outer: float, thickness_inner: float) -> float:
        R_outer = (self.enc / 2) + (thickness_outer / 2)
        R_inner = (self.enc / 2) - self.inner_offset - (thickness_inner / 2)
        return R_inner / R_outer

    def _calculate_L_inner_from_geometry(self, thickness_outer: float, thickness_inner: float) -> float:
        return self.length * self._calculate_radius_ratio(thickness_outer, thickness_inner)

    def _calculate_displacement_estimate(self, w_outer: float, thickness_outer: float,
                                         w_inner: float, thickness_inner: float) -> float:
        """Estime le déplacement en utilisant l'interpolateur RBF amélioré."""
        input_vector = np.array([[w_outer, thickness_outer, w_inner, thickness_inner]])

        # Vérification des limites d'interpolation
        exp_inputs = np.array([d['input'] for d in self.experimental_data])

        # Avertissement si on extrapole trop loin des données
        min_vals = np.min(exp_inputs, axis=0)
        max_vals = np.max(exp_inputs, axis=0)

        if (input_vector[0] < min_vals).any() or (input_vector[0] > max_vals).any():
            # Si on extrapole, on peut être moins confiant
            extrapolation_factor = 1.2  # Facteur d'incertitude
        else:
            extrapolation_factor = 1.0

        displacement = self.displacement_interpolator(input_vector)[0]
        return displacement * extrapolation_factor

    def _calculate_dual_performance(self, w_outer: float, thickness_outer: float, L_outer: float,
                                    w_inner: float, thickness_inner: float, L_inner: float) -> float:
        E_outer, E_inner = self.material_properties[self.outer_material], self.material_properties[self.inner_material]
        k_outer = self._calculate_k_single_blade(E_outer, w_outer, thickness_outer, L_outer)
        k_inner = self._calculate_k_single_blade(E_inner, w_inner, thickness_inner, L_inner)
        k_total = (k_outer + k_inner) * 1e3  # en mN·m
        length_equivalent = (L_outer + L_inner) / 2
        rs = abs(self._calculate_Dx(length_equivalent))
        return k_total * (rs ** 2)

    def evaluate_configuration_quality(self, config: Dict) -> float:
        """Évalue la qualité globale d'une configuration (score composite)"""
        # Facteurs de pondération
        w_displacement = 0.4  # Importance du déplacement
        w_performance = 0.3  # Importance de la performance
        w_stability = 0.2  # Importance de la stabilité (proche de 0)
        w_manufacturability = 0.1  # Facilité de fabrication

        # Score de déplacement (meilleur si proche de 0.25, milieu de la plage cible)
        target_center = (self.displacement_target[0] + self.displacement_target[1]) / 2
        displacement_error = abs(config['displacement_est'] - target_center)
        displacement_score = max(0, 1 - displacement_error / 1.0)  # Normalisé sur 1mm

        # Score de performance (meilleur si proche de 1.0)
        performance_error = abs(config['perf_ratio'] - 1.0)
        performance_score = max(0, 1 - performance_error / 0.2)  # Normalisé sur 20%

        # Score de stabilité (meilleur si déplacement proche de 0)
        stability_score = max(0, 1 - abs(config['displacement_est']) / 2.0)

        # Score de manufacturabilité (préfère les géométries "standards")
        thickness_penalty = abs(config['thickness_outer'] - 0.24) + abs(config['thickness_inner'] - 0.24)
        manufacturability_score = max(0, 1 - thickness_penalty / 0.1)

        # Score composite
        total_score = (w_displacement * displacement_score +
                       w_performance * performance_score +
                       w_stability * stability_score +
                       w_manufacturability * manufacturability_score)

        return total_score

    def get_global_minmax_configs(self, all_configs: List[Dict], config_type: ConfigurationTypeDual) -> Dict:
        """Extract global min/max configurations for dual blade system"""
        if not all_configs:
            return {}

        minmax_configs = {}

        if config_type == ConfigurationTypeDual.L_BECU_CONSTANT:
            # Pour L constant, chercher les min/max des paramètres géométriques principaux

            # Min/Max pour l'épaisseur outer
            thickness_outer_values = [config['thickness_outer'] for config in all_configs]
            thickness_outer_min = min(thickness_outer_values)
            thickness_outer_max = max(thickness_outer_values)

            # Min/Max pour la largeur outer
            w_outer_values = [config['w_outer'] for config in all_configs]
            w_outer_min = min(w_outer_values)
            w_outer_max = max(w_outer_values)

            # Min/Max pour l'épaisseur inner
            thickness_inner_values = [config['thickness_inner'] for config in all_configs]
            thickness_inner_min = min(thickness_inner_values)
            thickness_inner_max = max(thickness_inner_values)

            # Min/Max pour la largeur inner
            w_inner_values = [config['w_inner'] for config in all_configs]
            w_inner_min = min(w_inner_values)
            w_inner_max = max(w_inner_values)

            # Min/Max pour le déplacement
            displacement_values = [config['displacement_est'] for config in all_configs]
            displacement_min = min(displacement_values)
            displacement_max = max(displacement_values)

            # Min/Max pour la performance
            performance_values = [config['performance'] for config in all_configs]
            performance_min = min(performance_values)
            performance_max = max(performance_values)

            # Trouver les configurations correspondantes (première occurrence)
            minmax_configs = {
                'thickness_outer_min': next(
                    config for config in all_configs if config['thickness_outer'] == thickness_outer_min),
                'thickness_outer_max': next(
                    config for config in all_configs if config['thickness_outer'] == thickness_outer_max),
                'w_outer_min': next(config for config in all_configs if config['w_outer'] == w_outer_min),
                'w_outer_max': next(config for config in all_configs if config['w_outer'] == w_outer_max),
                'thickness_inner_min': next(
                    config for config in all_configs if config['thickness_inner'] == thickness_inner_min),
                'thickness_inner_max': next(
                    config for config in all_configs if config['thickness_inner'] == thickness_inner_max),
                'w_inner_min': next(config for config in all_configs if config['w_inner'] == w_inner_min),
                'w_inner_max': next(config for config in all_configs if config['w_inner'] == w_inner_max),
                'displacement_min': next(
                    config for config in all_configs if config['displacement_est'] == displacement_min),
                'displacement_max': next(
                    config for config in all_configs if config['displacement_est'] == displacement_max),
                'performance_min': next(config for config in all_configs if config['performance'] == performance_min),
                'performance_max': next(config for config in all_configs if config['performance'] == performance_max)
            }

        return minmax_configs

    def print_minmax_configurations(self, minmax_configs: Dict, title: str):
        """Affiche les configurations min/max avec formatage amélioré"""
        if not minmax_configs:
            print(f"\n=== {title} ===")
            print("Aucune configuration min/max trouvée.")
            return

        print(f"\n=== {title} ===")
        print("Paramètre\tType\tL_Outer\te_Outer\tw_Outer\tL_Inner\tei_Inner\tw_Inner\tDisp_Est\tPerformance\tRatio")
        print("-" * 120)

        for param_name, config in minmax_configs.items():
            param_display = param_name.replace('_', ' ').title()

            print(
                f"{param_display}\t{config['L_outer']:.2f}\t{config['thickness_outer']:.3f}\t{config['w_outer']:.1f}\t"
                f"{config['L_inner']:.2f}\t{config['thickness_inner']:.3f}\t{config['w_inner']:.1f}\t"
                f"{config['displacement_est']:.3f}\t{config['performance']:.2f}\t{config['perf_ratio']:.4f}")

    def generate_dual_configurations(self) -> List[Dict]:
        """Génère les configurations avec critères adaptatifs"""
        configurations = []
        total_tested = 0
        L_outer_fixed = self.length

        print(f"Génération des configurations...")

        for thickness_outer in self.thickness_range:
            for w_outer in self.w_outer_range:
                for thickness_inner in self.inner_thickness_range:
                    for w_inner in self.w_inner_range:
                        total_tested += 1

                        L_inner_calculated = self._calculate_L_inner_from_geometry(thickness_outer, thickness_inner)

                        # Calcul du déplacement
                        displacement = self._calculate_displacement_estimate(w_outer, thickness_outer, w_inner,
                                                                             thickness_inner)

                        # Calcul de la performance
                        perf = self._calculate_dual_performance(w_outer, thickness_outer, L_outer_fixed,
                                                                w_inner, thickness_inner, L_inner_calculated)

                        # Critères de sélection
                        displacement_valid = self.is_displacement_valid(displacement)
                        performance_valid = self.is_performance_ratio_valid(perf)

                        # Si critères stricts, on applique les deux
                        if self.use_strict_criteria:
                            if displacement_valid and performance_valid:
                                config = self._create_dual_config_dict(w_outer, thickness_outer, L_outer_fixed,
                                                                       w_inner, thickness_inner, L_inner_calculated,
                                                                       perf)
                                configurations.append(config)
                        else:
                            # Critères plus souples - on garde si au moins un critère est respecté
                            if displacement_valid or performance_valid:
                                config = self._create_dual_config_dict(w_outer, thickness_outer, L_outer_fixed,
                                                                       w_inner, thickness_inner, L_inner_calculated,
                                                                       perf)
                                configurations.append(config)

        print(f"Total testé: {total_tested}")
        print(f"Configurations trouvées: {len(configurations)}")

        # Ajout du score de qualité à chaque configuration
        for config in configurations:
            config['quality_score'] = self.evaluate_configuration_quality(config)

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
            'displacement_valid': self.is_displacement_valid(displacement_est),
            'performance_valid': self.is_performance_ratio_valid(performance),
        }

    def find_diverse_configurations(self, configurations: List[Dict], n_clusters: int = 8) -> List[Dict]:
        """Trouve des configurations diverses en utilisant K-Means avec features améliorées"""
        if len(configurations) < n_clusters:
            return configurations

        # Features plus complètes pour le clustering
        features = np.array([
            [c['w_outer'], c['thickness_outer'], c['w_inner'], c['thickness_inner'],
             c['L_inner'], c['displacement_est'], c['k_ratio'], c['performance'], c['quality_score']]
            for c in configurations
        ])

        features_scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features_scaled)

        cluster_centers_indices = [
            np.argmin(np.linalg.norm(features_scaled - center, axis=1))
            for center in kmeans.cluster_centers_
        ]

        return [configurations[i] for i in cluster_centers_indices]

    def print_dual_configurations(self, configs: List[Dict], title: str, show_all: bool = False):
        """Affiche les configurations avec indicateurs de validité"""
        if not configs:
            print(f"\n=== {title} ===")
            print("Aucune configuration trouvée.")
            return

        print(f"\n=== {title} ({len(configs)} configurations) ===")
        print(
            "Rank\tL_Outer\te_Outer\tw_Outer\tL_Inner\tei_Inner\tw_Inner\tDisp_Est\tD_Valid\tP_Valid\tQuality\tk_total\tRatio")
        print("-" * 140)

        max_display = len(configs) if show_all else min(20, len(configs))

        for i, config in enumerate(configs[:max_display], 1):
            d_valid = "✓" if config.get('displacement_valid', False) else "✗"
            p_valid = "✓" if config.get('performance_valid', False) else "✗"
            quality = config.get('quality_score', 0)

            print(f"{i:2d}\t{config['L_outer']:.2f}\t{config['thickness_outer']:.3f}\t{config['w_outer']:.1f}\t"
                  f"{config['L_inner']:.2f}\t{config['thickness_inner']:.3f}\t{config['w_inner']:.1f}\t"
                  f"{config['displacement_est']:.3f}\t{d_valid}\t{p_valid}\t{quality:.3f}\t"
                  f"{config['k_total_mNm']:.1f}\t{config['perf_ratio']:.4f}")

    def analyze_results(self, configs: List[Dict]):
        """Analyse statistique des résultats"""
        if not configs:
            return

        displacements = [c['displacement_est'] for c in configs]
        performances = [c['perf_ratio'] for c in configs]
        qualities = [c.get('quality_score', 0) for c in configs]

        print(f"\n=== ANALYSE STATISTIQUE ===")
        print(f"Nombre de configurations: {len(configs)}")
        print(
            f"Déplacement - Min: {min(displacements):.3f}, Max: {max(displacements):.3f}, Moyenne: {np.mean(displacements):.3f}")
        print(
            f"Performance ratio - Min: {min(performances):.3f}, Max: {max(performances):.3f}, Moyenne: {np.mean(performances):.3f}")
        print(
            f"Score qualité - Min: {min(qualities):.3f}, Max: {max(qualities):.3f}, Moyenne: {np.mean(qualities):.3f}")

        # Compter les configurations valides
        displacement_valid_count = sum(1 for c in configs if c.get('displacement_valid', False))
        performance_valid_count = sum(1 for c in configs if c.get('performance_valid', False))
        both_valid_count = sum(
            1 for c in configs if c.get('displacement_valid', False) and c.get('performance_valid', False))

        print(f"\nValidité des critères:")
        print(
            f"- Déplacement valide: {displacement_valid_count}/{len(configs)} ({100 * displacement_valid_count / len(configs):.1f}%)")
        print(
            f"- Performance valide: {performance_valid_count}/{len(configs)} ({100 * performance_valid_count / len(configs):.1f}%)")
        print(f"- Les deux valides: {both_valid_count}/{len(configs)} ({100 * both_valid_count / len(configs):.1f}%)")


def main_dual_enhanced():
    """Version améliorée du main avec analyse comparative et configurations min/max"""
    print("=== COMPARAISON CRITÈRES STRICTS vs SOUPLES ===")

    # Test avec critères stricts
    print("\n1. GÉNÉRATION AVEC CRITÈRES STRICTS")
    dual_config_strict = DualBladeConfig(use_strict_criteria=True)
    configs_strict = dual_config_strict.generate_dual_configurations()

    # Test avec critères souples
    print("\n2. GÉNÉRATION AVEC CRITÈRES SOUPLES")
    dual_config_flexible = DualBladeConfig(use_strict_criteria=False)
    configs_flexible = dual_config_flexible.generate_dual_configurations()

    # Tri par score de qualité
    if configs_strict:
        configs_strict.sort(key=lambda x: x['quality_score'], reverse=True)
        dual_config_strict.print_dual_configurations(configs_strict, "MEILLEURES CONFIGURATIONS (CRITÈRES STRICTS)")
        dual_config_strict.analyze_results(configs_strict)

        # Extraction et affichage des configurations min/max pour les critères stricts
        minmax_strict = dual_config_strict.get_global_minmax_configs(configs_strict,
                                                                     ConfigurationTypeDual.L_BECU_CONSTANT)
        dual_config_strict.print_minmax_configurations(minmax_strict, "CONFIGURATIONS MIN/MAX (CRITÈRES STRICTS)")

    if configs_flexible:
        configs_flexible.sort(key=lambda x: x['quality_score'], reverse=True)
        dual_config_flexible.print_dual_configurations(configs_flexible, "MEILLEURES CONFIGURATIONS (CRITÈRES SOUPLES)")
        dual_config_flexible.analyze_results(configs_flexible)

        # Extraction et affichage des configurations min/max pour les critères souples
        minmax_flexible = dual_config_flexible.get_global_minmax_configs(configs_flexible,
                                                                         ConfigurationTypeDual.L_BECU_CONSTANT)
        dual_config_flexible.print_minmax_configurations(minmax_flexible, "CONFIGURATIONS MIN/MAX (CRITÈRES SOUPLES)")

    # Configurations diverses
    if len(configs_flexible) > 8:
        diverse_configs = dual_config_flexible.find_diverse_configurations(configs_flexible, n_clusters=8)
        dual_config_flexible.print_dual_configurations(diverse_configs, "CONFIGURATIONS DIVERSES (K-MEANS)")

    return dual_config_strict, configs_strict, dual_config_flexible, configs_flexible


if __name__ == "__main__":
    # Exécution du programme amélioré
    strict_config, strict_results, flexible_config, flexible_results = main_dual_enhanced()