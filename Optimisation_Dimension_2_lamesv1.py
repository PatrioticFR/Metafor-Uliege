import math
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    """
    Calcule les valeurs idéales pour perf_ratio_min et perf_ratio_max
    pour le système à 2 lames basé sur les données expérimentales
    """
    # Données observées pour le système à 2 lames
    dual_configs = {
        'config1': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 40.0,
            'L_Invar': 100.0, 'h_Invar': 0.20, 'w_Invar': 35.0,
            'perf_ratio': 1.015, 'stabilization': 0.75
        },
        'config2': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 45.0,
            'L_Invar': 95.0, 'h_Invar': 0.25, 'w_Invar': 40.0,
            'perf_ratio': 1.025, 'stabilization': 0.60
        }
    }

    config1 = dual_configs['config1']
    config2 = dual_configs['config2']

    target_min = 0.5  # mm
    target_max = 1.0  # mm

    print(f"=== CALCUL DU PERF_RATIO IDÉAL POUR SYSTÈME À 2 LAMES ===")
    print(f"Configuration 1: L_BeCu={config1['L_BeCu']}, h_BeCu={config1['h_BeCu']}, w_BeCu={config1['w_BeCu']}")
    print(f"                 L_Invar={config1['L_Invar']}, h_Invar={config1['h_Invar']}, w_Invar={config1['w_Invar']}")
    print(f"  perf_ratio = {config1['perf_ratio']:.4f} -> stabilisation = {config1['stabilization']:.3f} mm")
    print(f"Configuration 2: L_BeCu={config2['L_BeCu']}, h_BeCu={config2['h_BeCu']}, w_BeCu={config2['w_BeCu']}")
    print(f"                 L_Invar={config2['L_Invar']}, h_Invar={config2['h_Invar']}, w_Invar={config2['w_Invar']}")
    print(f"  perf_ratio = {config2['perf_ratio']:.4f} -> stabilisation = {config2['stabilization']:.3f} mm")
    print()

    # Calcul par interpolation linéaire
    delta_stab = config1['stabilization'] - config2['stabilization']
    delta_perf = config1['perf_ratio'] - config2['perf_ratio']

    a = delta_stab / delta_perf
    b = config1['stabilization'] - a * config1['perf_ratio']

    print(f"Relation linéaire: stabilisation = {a:.2f} * perf_ratio + {b:.2f}")

    perf_ratio_for_target_min = (target_min - b) / a
    perf_ratio_for_target_max = (target_max - b) / a

    perf_ratio_min = min(perf_ratio_for_target_min, perf_ratio_for_target_max)
    perf_ratio_max = max(perf_ratio_for_target_min, perf_ratio_for_target_max)

    print(f"VALEURS IDÉALES POUR SYSTÈME À 2 LAMES:")
    print(f"self.perf_ratio_min = {perf_ratio_min:.6f}")
    print(f"self.perf_ratio_max = {perf_ratio_max:.6f}")
    print()

    return perf_ratio_min, perf_ratio_max


class DualBladeConfig:
    """Configuration modulaire pour optimisation de 2 lames avec effet de bras de levier"""

    def __init__(self):
        # Géométrie de référence BE_CU
        self.thickness_ref_becu = 0.24  # h référence BE_CU
        self.length_ref_becu = 105.25  # L référence BE_CU (FIXE)
        self.width_ref_becu = 40.0  # w référence BE_CU

        # Paramètres de référence INVAR
        self.thickness_ref_invar = 0.20  # h référence INVAR
        self.length_ref_invar = 100.0  # L référence INVAR
        self.width_ref_invar = 35.0  # w référence INVAR

        self.enc_ref = 57.32  # Position encodeur sur la masse
        self.decalage = 0.1  # mm (décalage entre les lames)

        # Propriétés matériaux (en N/mm²) - identiques au système 1 lame
        self.material_properties = {
            'BE_CU': 131e3,
            'INVAR': 141e3,
        }

        # Paramètres géométriques du système tige/masse (identiques au système 1 lame)
        self.H = 3.875
        self.D = 39.99
        self.d = 13.96
        self.l = 79.2
        self.r = 7.0
        self.R_rod = self.H
        self.rho = 7.85e-6
        self.depth = 63.0

        # Paramètres système flexure
        self.k_flex = 44.3
        self.r_s = 65.22

        # Plages de recherche
        self.w_becu_range = np.arange(10, 50.1, 0.5)  # Largeur BE_CU
        self.h_becu_range = np.arange(0.15, 0.31, 0.01)  # Épaisseur BE_CU

        self.w_invar_range = np.arange(10, 50.1, 0.5)  # Largeur INVAR
        self.h_invar_range = np.arange(0.05, 0.31, 0.01)  # Épaisseur INVAR
        self.L_invar_range = np.arange(90.0, 110.1, 0.5)  # Longueur INVAR

        # Calcul des ratios de performance idéaux
        self.perf_ratio_min, self.perf_ratio_max = calculate_ideal_perf_ratio_range_dual()

        # Inertie système (constante) - même calcul que système 1 lame
        self.I_system = self._calculate_system_inertia()

        # Performance de référence
        self.performance_ref = self._calculate_dual_performance(
            self.width_ref_becu, self.thickness_ref_becu, self.length_ref_becu,
            self.width_ref_invar, self.thickness_ref_invar, self.length_ref_invar
        )

        print(f"Configuration: Système à 2 lames")
        print(f"Perf_ratio range: [{self.perf_ratio_min:.6f}, {self.perf_ratio_max:.6f}]")
        print(f"Performance de référence: {self.performance_ref:.6f}")
        print(f"Inertie système: {self.I_system:.3f} kg·mm²")

    def _calculate_system_inertia(self) -> float:
        """Calcule l'inertie totale du système tige/masse (identique au système 1 lame)"""
        h_rod_segment = self.l - self.r - self.d

        I_barre_rod_g = self.rho * self.depth * h_rod_segment * (
                self.H ** 3 / 3 + self.H * (h_rod_segment / 2) ** 2
        )
        I_barre_mass = self.rho * self.depth * self.d * (
                self.D ** 3 / 3 + self.D * (h_rod_segment + self.d / 2) ** 2
        )
        I_barre_rod_d = self.rho * self.depth * self.r * (
                self.R_rod ** 3 / 3 + self.R_rod * (h_rod_segment + self.d + self.r / 2) ** 2
        )

        return I_barre_rod_g + I_barre_mass + I_barre_rod_d

    def _calculate_L_invar_from_geometry(self, h_becu: float, h_invar: float) -> float:
        """Calcule L_Invar basé sur la géométrie relative des lames"""
        R_BeCu = self.enc_ref / 2 + h_becu / 2
        R_Invar = (self.enc_ref / 2 - self.decalage - h_invar / 2)
        return self.length_ref_becu * (R_Invar / R_BeCu)

    def _calculate_Dx(self, length: float) -> float:
        """Calcule Dx selon la formule du système 1 lame"""
        R = length / math.pi
        return -2 * R

    def _calculate_k_single_blade(self, E: float, width: float, thickness: float, length: float) -> float:
        """Calcule la raideur d'une lame individuelle [N·mm] - identique au système 1 lame"""
        return (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3

    def _calculate_dual_performance(self, w_becu: float, h_becu: float, L_becu: float,
                                    w_invar: float, h_invar: float, L_invar: float) -> float:
        """Calcule k_total * rs² pour le système à 2 lames"""

        # Raideurs individuelles avec même formule que système 1 lame
        E_becu = self.material_properties['BE_CU']
        E_invar = self.material_properties['INVAR']

        k_becu = self._calculate_k_single_blade(E_becu, w_becu, h_becu, L_becu)
        k_invar = self._calculate_k_single_blade(E_invar, w_invar, h_invar, L_invar)

        k_total = k_becu + k_invar  # Raideur totale en parallèle
        k_total_mNm = k_total * 1e3  # Conversion en mN·m

        # Longueur équivalente pour le calcul du bras de levier
        length_equivalent = (L_becu + L_invar) / 2

        # Calcul du bras de levier avec même formule que système 1 lame
        Dx = self._calculate_Dx(length_equivalent)
        rs = abs(Dx)

        # Performance = k_total * rs² (même principe que système 1 lame)
        performance = k_total_mNm * (rs ** 2)

        return performance

    def is_performance_ratio_valid(self, performance: float) -> bool:
        """Vérifie si le ratio de performance est dans la plage idéale"""
        perf_ratio = performance / self.performance_ref
        return self.perf_ratio_min <= perf_ratio <= self.perf_ratio_max

    def generate_dual_configurations(self) -> List[Dict]:
        """Génère les configurations pour le système à 2 lames"""
        configurations = []

        # L_BE_CU est fixe, on fait varier tous les autres paramètres
        L_becu_fixed = self.length_ref_becu

        for h_becu in self.h_becu_range:
            for w_becu in self.w_becu_range:
                for h_invar in self.h_invar_range:
                    for w_invar in self.w_invar_range:
                        # Calcul de L_invar basé sur la géométrie
                        L_invar_geometric = self._calculate_L_invar_from_geometry(h_becu, h_invar)

                        # Vérifier si L_invar_geometric est dans la plage acceptable
                        if not (self.L_invar_range[0] <= L_invar_geometric <= self.L_invar_range[-1]):
                            continue

                        # Calcul de la performance
                        perf = self._calculate_dual_performance(
                            w_becu, h_becu, L_becu_fixed,
                            w_invar, h_invar, L_invar_geometric
                        )

                        if self.is_performance_ratio_valid(perf):
                            config = self._create_dual_config_dict(
                                w_becu, h_becu, L_becu_fixed,
                                w_invar, h_invar, L_invar_geometric,
                                perf
                            )
                            configurations.append(config)

        return configurations

    def _create_dual_config_dict(self, w_becu: float, h_becu: float, L_becu: float,
                                 w_invar: float, h_invar: float, L_invar: float,
                                 performance: float) -> Dict:
        """Crée le dictionnaire de configuration pour 2 lames"""
        perf_ratio = performance / self.performance_ref
        perf_optimal = (self.perf_ratio_min + self.perf_ratio_max) / 2

        # Longueur équivalente pour Dx
        length_equivalent = (L_becu + L_invar) / 2
        Dx = self._calculate_Dx(length_equivalent)
        rs = abs(Dx)

        # Raideurs individuelles pour information
        E_becu = self.material_properties['BE_CU']
        E_invar = self.material_properties['INVAR']

        k_becu = self._calculate_k_single_blade(E_becu, w_becu, h_becu, L_becu)
        k_invar = self._calculate_k_single_blade(E_invar, w_invar, h_invar, L_invar)
        k_total = k_becu + k_invar
        k_total_mNm = k_total * 1e3

        return {
            # Paramètres BE_CU
            'w_becu': w_becu,
            'h_becu': h_becu,
            'L_becu': L_becu,
            'k_becu_mNm': k_becu * 1e3,

            # Paramètres INVAR
            'w_invar': w_invar,
            'h_invar': h_invar,
            'L_invar': L_invar,
            'k_invar_mNm': k_invar * 1e3,

            # Paramètres système
            'k_total_mNm': k_total_mNm,
            'performance': performance,  # k_total * rs²
            'perf_ratio': perf_ratio,
            'Dx': Dx,
            'rs': rs,
            'k_times_rs_squared': performance,
            'perf_error_from_optimal': abs(perf_ratio - perf_optimal),
            'length_equivalent': length_equivalent,
        }

    def _get_minmax_configs_dual(self, configs: List[Dict]) -> Dict:
        """Extrait les configurations min/max pour le système à 2 lames"""
        if not configs:
            return {}

        minmax_configs = {}

        # Min/Max pour w_becu
        w_becu_values = [config['w_becu'] for config in configs]
        w_becu_min = min(w_becu_values)
        w_becu_max = max(w_becu_values)
        w_becu_min_config = next(config for config in configs if config['w_becu'] == w_becu_min)
        w_becu_max_config = next(config for config in configs if config['w_becu'] == w_becu_max)
        minmax_configs['w_becu_min'] = w_becu_min_config
        minmax_configs['w_becu_max'] = w_becu_max_config

        # Min/Max pour w_invar
        w_invar_values = [config['w_invar'] for config in configs]
        w_invar_min = min(w_invar_values)
        w_invar_max = max(w_invar_values)
        w_invar_min_config = next(config for config in configs if config['w_invar'] == w_invar_min)
        w_invar_max_config = next(config for config in configs if config['w_invar'] == w_invar_max)
        minmax_configs['w_invar_min'] = w_invar_min_config
        minmax_configs['w_invar_max'] = w_invar_max_config

        # Min/Max pour L_invar
        L_invar_values = [config['L_invar'] for config in configs]
        L_invar_min = min(L_invar_values)
        L_invar_max = max(L_invar_values)
        L_invar_min_config = next(config for config in configs if config['L_invar'] == L_invar_min)
        L_invar_max_config = next(config for config in configs if config['L_invar'] == L_invar_max)
        minmax_configs['L_invar_min'] = L_invar_min_config
        minmax_configs['L_invar_max'] = L_invar_max_config

        return minmax_configs

    def find_diverse_configurations(self, configurations: List[Dict], n_clusters: int = 6) -> List[Dict]:
        """Trouve des configurations diverses using K-Means clustering"""
        if len(configurations) < n_clusters:
            return configurations

        # Préparer les données pour le clustering
        features = np.array([
            [config['w_becu'], config['h_becu'], config['w_invar'],
             config['h_invar'], config['L_invar']]
            for config in configurations
        ])

        # Normalisation
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Clustering K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(features_scaled)

        # Trouver les configurations les plus proches des centres de clusters
        cluster_centers_indices = [
            np.argmin(np.linalg.norm(features_scaled - center, axis=1))
            for center in kmeans.cluster_centers_
        ]

        return [configurations[i] for i in cluster_centers_indices]

    def print_dual_configurations(self, configs: List[Dict], title: str, show_all: bool = False):
        """Affiche les configurations pour le système à 2 lames - format similaire au système 1 lame"""
        if not configs:
            print(f"\n=== {title} ===")
            print("Aucune configuration trouvée.")
            return

        print(f"\n=== {title} ({len(configs)} configurations) ===")
        print("Rank\tL_BeCu\th_BeCu\tw_BeCu\tL_Invar\th_Invar\tw_Invar\tDx\trs\tk_total\trs²\tk×rs²\tRatio\tError")
        print("-" * 150)

        max_display = len(configs) if show_all else min(20, len(configs))

        for i, config in enumerate(configs[:max_display], 1):
            rs_squared = config['rs'] ** 2
            print(f"{i:2d}\t{config['L_becu']:.2f}\t{config['h_becu']:.3f}\t"
                  f"{config['w_becu']:.1f}\t{config['L_invar']:.2f}\t{config['h_invar']:.3f}\t"
                  f"{config['w_invar']:.1f}\t{config['Dx']:.2f}\t{config['rs']:.2f}\t"
                  f"{config['k_total_mNm']:.1f}\t{rs_squared:.1f}\t{config['k_times_rs_squared']:.1f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    def _print_minmax_dual(self, configs: List[Dict], title: str):
        """Affiche les valeurs min/max pour le système à 2 lames"""
        if not configs:
            return

        minmax_configs = self._get_minmax_configs_dual(configs)
        if not minmax_configs:
            return

        base_title = title.split('(')[0].strip()
        print(f"\n=== {base_title} (Min / Max Values) ===")
        print("Info\t\tL_BeCu\th_BeCu\tw_BeCu\tL_Invar\th_Invar\tw_Invar\tDx\trs\tk_total\trs²\tk×rs²\tRatio\tError")
        print("-" * 160)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            rs_squared = config['rs'] ** 2
            print(f"{label}\t{config['L_becu']:.2f}\t{config['h_becu']:.3f}\t"
                  f"{config['w_becu']:.1f}\t{config['L_invar']:.2f}\t{config['h_invar']:.3f}\t"
                  f"{config['w_invar']:.1f}\t{config['Dx']:.2f}\t{config['rs']:.2f}\t"
                  f"{config['k_total_mNm']:.1f}\t{rs_squared:.1f}\t{config['k_times_rs_squared']:.1f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    def print_dual_configurations_with_minmax(self, configs: List[Dict], title: str, show_all: bool = False):
        """Affiche les configurations avec les valeurs min/max"""
        self.print_dual_configurations(configs, title, show_all)
        self._print_minmax_dual(configs, title)

    def print_excel_format(self, configs: List[Dict], title: str):
        """Affiche les configurations au format Excel"""
        print(f"\n=== {title} - FORMAT EXCEL ===")
        print("L_BeCu\th_BeCu\tw_BeCu\tL_Invar\th_Invar\tw_Invar\tk_total_mNm\tperf_ratio")

        for config in configs:
            print(f"{config['L_becu']:.2f}\t{config['h_becu']:.3f}\t{config['w_becu']:.1f}\t"
                  f"{config['L_invar']:.2f}\t{config['h_invar']:.3f}\t{config['w_invar']:.1f}\t"
                  f"{config['k_total_mNm']:.2f}\t{config['perf_ratio']:.6f}")

    def verify_dual_calculation(self, w_becu: float, h_becu: float, L_becu: float,
                                w_invar: float, h_invar: float, L_invar: float):
        """Vérification d'un calcul spécifique pour 2 lames - style système 1 lame"""
        print(f"\n=== VÉRIFICATION CALCUL SYSTÈME À 2 LAMES ===")
        print(f"BE_CU:  L={L_becu}, h={h_becu}, w={w_becu}")
        print(f"INVAR:  L={L_invar}, h={h_invar}, w={w_invar}")

        # Calculs individuels avec mêmes formules que système 1 lame
        E_becu = self.material_properties['BE_CU']
        E_invar = self.material_properties['INVAR']

        k_becu = self._calculate_k_single_blade(E_becu, w_becu, h_becu, L_becu)
        k_invar = self._calculate_k_single_blade(E_invar, w_invar, h_invar, L_invar)
        k_total = k_becu + k_invar
        k_total_mNm = k_total * 1e3

        # Bras de levier avec même formule que système 1 lame
        length_equivalent = (L_becu + L_invar) / 2
        Dx = self._calculate_Dx(length_equivalent)
        rs = abs(Dx)

        performance = self._calculate_dual_performance(w_becu, h_becu, L_becu, w_invar, h_invar, L_invar)
        perf_ratio = performance / self.performance_ref

        print(f"E_BE_CU = {E_becu:.0f} N/mm²")
        print(f"E_INVAR = {E_invar:.0f} N/mm²")
        print(f"k_BE_CU = {k_becu * 1e3:.2f} mN·m")
        print(f"k_INVAR = {k_invar * 1e3:.2f} mN·m")
        print(f"k_total = {k_total_mNm:.2f} mN·m")
        print(f"Longueur équivalente = {length_equivalent:.2f} mm")
        print(f"Dx = {Dx:.4f} mm")
        print(f"rs = |Dx| = {rs:.4f} mm")
        print(f"rs² = {rs ** 2:.2f} mm²")
        print(f"k_total × rs² = {performance:.2f} mN·m·mm²")
        print(f"Performance de référence = {self.performance_ref:.2f} mN·m·mm²")
        print(f"Ratio = {perf_ratio:.6f}")
        print(f"Dans la plage acceptable? {self.is_performance_ratio_valid(performance)}")
        print()
        print("INTERPRÉTATION PHYSIQUE:")
        print(f"- Pour maintenir l'équilibre, k_total × rs² doit rester proche de {self.performance_ref:.2f}")
        print(f"- Si rs diminue, k_total doit augmenter proportionnellement à 1/rs²")
        print(f"- Si rs augmente, k_total peut diminuer proportionnellement à 1/rs²")


def main_dual():
    """Fonction principale pour le système à 2 lames"""

    # Configuration avec calculs identiques au système 1 lame
    dual_config = DualBladeConfig()

    # Vérification avec configuration de référence
    print("VÉRIFICATION CONFIGURATION DE RÉFÉRENCE:")
    dual_config.verify_dual_calculation(
        dual_config.width_ref_becu, dual_config.thickness_ref_becu, dual_config.length_ref_becu,
        dual_config.width_ref_invar, dual_config.thickness_ref_invar, dual_config.length_ref_invar
    )

    # Génération de toutes les configurations valides
    print("\nGénération des configurations...")
    all_configs = dual_config.generate_dual_configurations()

    if not all_configs:
        print("Aucune configuration trouvée dans les critères spécifiés!")
        return dual_config

    # Tri par erreur par rapport à l'optimal
    all_configs.sort(key=lambda x: x['perf_error_from_optimal'])

    # Affichage des meilleures configurations avec min/max
    dual_config.print_dual_configurations_with_minmax(
        all_configs,
        f"MEILLEURES CONFIGURATIONS - SYSTÈME À 2 LAMES (L_BE_CU = {dual_config.length_ref_becu} mm fixe)"
    )

    # Configurations diverses avec K-Means
    diverse_configs = dual_config.find_diverse_configurations(all_configs, n_clusters=6)
    dual_config.print_dual_configurations_with_minmax(
        diverse_configs,
        "CONFIGURATIONS DIVERSES (K-MEANS)"
    )

    # Format Excel
    dual_config.print_excel_format(diverse_configs, "CONFIGURATIONS DIVERSES")

    print(f"\nRésumé:")
    print(f"- Configurations totales trouvées: {len(all_configs)}")
    print(f"- Configurations diverses sélectionnées: {len(diverse_configs)}")
    print(f"- Plage perf_ratio cible: [{dual_config.perf_ratio_min:.6f}, {dual_config.perf_ratio_max:.6f}]")

    return dual_config, all_configs, diverse_configs


if __name__ == "__main__":
    config, all_configs, diverse_configs = main_dual()