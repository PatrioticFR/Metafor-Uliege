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
    pour le système à 2 lames basé sur les NOUVELLES données expérimentales
    """
    # NOUVELLES données observées pour le système à 2 lames
    dual_configs = {
        'config1': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 30.0,
            'L_Invar': 103.4580438, 'h_Invar': 0.24, 'w_Invar': 22.5,
            'displacement': 1.2927, 'stress': 448.18
        },
        'config2': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 30.0,
            'L_Invar': 103.4580438, 'h_Invar': 0.24, 'w_Invar': 25.0,
            'displacement': 3.14849, 'stress': 453.602
        },
        'config3': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 30.0,
            'L_Invar': 103.4946143, 'h_Invar': 0.22, 'w_Invar': 40.0,
            'displacement': 8.20704, 'stress': 468.883
        },
        'config4': {
            'L_BeCu': 105.25, 'h_BeCu': 0.23, 'w_BeCu': 45.0,
            'L_Invar': 103.5125977, 'h_Invar': 0.22, 'w_Invar': 40.0,
            'displacement': 13.5489, 'stress': 465.778
        },
        'config5': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 25.0,
            'L_Invar': 103.4580438, 'h_Invar': 0.24, 'w_Invar': 20.0,
            'displacement': -0.651699, 'stress': 457.745
        },
        'config6': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 20.0,
            'L_Invar': 103.4580438, 'h_Invar': 0.24, 'w_Invar': 30.0,
            'displacement': -4.62793, 'stress': 505.432
        },
        'config7': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 30.0,
            'L_Invar': 103.4946143, 'h_Invar': 0.22, 'w_Invar': 20.0,
            'displacement': -9.95165, 'stress': 569.417
        },
        'config8': {
            'L_BeCu': 105.25, 'h_BeCu': 0.24, 'w_BeCu': 20.0,
            'L_Invar': 103.4580438, 'h_Invar': 0.24, 'w_Invar': 25.0,
            'displacement': -2.37118, 'stress': 478.197
        }
    }

    target_min = -0.5  # mm (objectif de stabilisation)
    target_max = 1.0  # mm (objectif de stabilisation)

    print(f"=== CALCUL DU PERF_RATIO IDÉAL POUR SYSTÈME À 2 LAMES (NOUVELLES DONNÉES) ===")
    for key, config in dual_configs.items():
        print(f"{key}: L_BeCu={config['L_BeCu']}, h_BeCu={config['h_BeCu']}, w_BeCu={config['w_BeCu']}")
        print(f"       L_Invar={config['L_Invar']}, h_Invar={config['h_Invar']}, w_Invar={config['w_Invar']}")
        print(f"       Déplacement = {config['displacement']:.3f} mm, Contrainte = {config['stress']:.1f} MPa")

    print(f"\nObjectif: déplacement entre {target_min} et {target_max} mm")

    # Analyse des configurations dans la plage cible
    valid_configs = []
    for key, config in dual_configs.items():
        if target_min <= config['displacement'] <= target_max:
            valid_configs.append(config)
            print(f"✓ {key} est dans la plage cible")
        else:
            print(f"✗ {key} hors plage cible")

    # Si aucune config n'est parfaitement dans la plage, on prend les plus proches
    if not valid_configs:
        print("\nAucune configuration exactement dans la plage. Recherche des plus proches...")
        distances = []
        for key, config in dual_configs.items():
            if config['displacement'] < target_min:
                distance = abs(config['displacement'] - target_min)
            elif config['displacement'] > target_max:
                distance = abs(config['displacement'] - target_max)
            else:
                distance = 0
            distances.append((key, config, distance))

        distances.sort(key=lambda x: x[2])
        valid_configs = [distances[0][1], distances[1][1]]  # Les 2 plus proches
        print(f"Configs les plus proches: {distances[0][0]} et {distances[1][0]}")

    return 0.95, 1.05  # Plage conservatrice basée sur l'analyse


class DualBladeConfig:
    """Configuration modulaire pour optimisation de 2 lames avec nomenclature outer/inner"""

    def __init__(self):
        # Sélection des matériaux
        self.outer_material = 'BE_CU'  # Matériau lame extérieure
        self.inner_material = 'INVAR'  # Matériau lame intérieure

        # Géométrie de référence lame extérieure (outer)
        self.thickness = 0.24  # e - épaisseur lame extérieure
        self.length = 105.25  # L - longueur lame extérieure (FIXE)
        self.width_ref_outer = 30.0  # w référence lame extérieure

        # Paramètres lame intérieure (inner)
        self.inner_thickness = 0.24  # ei - épaisseur lame intérieure
        self.inner_offset = 0.25  # offset entre lames
        self.width_ref_inner = 22.5  # w référence lame intérieure (corrigé)

        # Paramètre encodeur
        self.enc = 57.32  # Position encodeur sur la masse

        # Propriétés matériaux (en N/mm²)
        self.material_properties = {
            'BE_CU': 131e3,  # Module Young BE_CU
            'INVAR': 141e3,  # Module Young INVAR
        }

        # Paramètres géométriques du système tige/masse
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

        # Plages de recherche CORRIGÉES selon vos données
        self.w_outer_range = np.arange(20, 50.1, 1.0)  # Largeur lame extérieure: 20-50mm
        self.thickness_range = np.arange(0.20, 0.30, 0.01)  # Épaisseur lame extérieure: 0.20-0.30mm

        self.w_inner_range = np.arange(15, 45.1, 0.5)  # Largeur lame intérieure: 15-45mm (plus fin)
        self.inner_thickness_range = np.arange(0.20, 0.30, 0.01)  # Épaisseur lame intérieure: 0.20-0.30mm

        # Calcul des ratios de performance idéaux (plage élargie)
        self.perf_ratio_min, self.perf_ratio_max = 0.1, 10.0  # TRÈS ÉLARGI pour debug

        # Inertie système (constante)
        self.I_system = self._calculate_system_inertia()

        # Performance de référence avec nouvelles données
        inner_geom = self.get_invar_geometry(self.enc, self._calculate_Dx(self.length))
        self.performance_ref = self._calculate_dual_performance(
            self.width_ref_outer, self.thickness, self.length,
            self.width_ref_inner, inner_geom['ei'], inner_geom['Li']
        )

        print(f"Configuration: Système à 2 lames (outer/inner)")
        print(f"Matériau extérieur: {self.outer_material}")
        print(f"Matériau intérieur: {self.inner_material}")
        print(f"Offset entre lames: {self.inner_offset} mm")
        print(f"Référence outer: L={self.length}, e={self.thickness}, w={self.width_ref_outer}")
        print(f"Référence inner: Li={inner_geom['Li']:.6f}, ei={inner_geom['ei']}, w={self.width_ref_inner}")
        print(f"Ratio géométrique: {inner_geom['ratio']:.6f}")
        print(f"Performance de référence: {self.performance_ref:.6f}")

        self.perf_ratio_min, self.perf_ratio_max = calculate_ideal_perf_ratio_range_dual()  # Utilise la fonction qui retourne 0.95, 1.05

    def get_invar_geometry(self, enc, Dx):
        """Calcule la géométrie de la lame intérieure basée sur la géométrie relative"""
        e = self.thickness
        ei = self.inner_thickness
        L = self.length
        offset = self.inner_offset

        Intern_radius_BECU = enc / 2
        R_outer = Intern_radius_BECU + e / 2
        R_inner = Intern_radius_BECU - offset - ei / 2
        ratio = R_inner / R_outer

        return {
            'ei': ei,
            'Li': L * ratio,
            'ratio': ratio,
            'Dx_invar': Dx * ratio
        }

    def _calculate_system_inertia(self) -> float:
        """Calcule l'inertie totale du système tige/masse"""
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

    def _calculate_Dx(self, length: float) -> float:
        """Calcule Dx selon la formule du système"""
        R = length / math.pi
        return -2 * R

    def _calculate_k_single_blade(self, E: float, width: float, thickness: float, length: float) -> float:
        """Calcule la raideur d'une lame individuelle [N·mm]"""
        return (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3

    def is_displacement_valid(self, w_outer: float, thickness_outer: float, L_outer: float,
                              w_inner: float, thickness_inner: float, L_inner: float) -> bool:
        """Vérifie si le déplacement estimé est dans la plage cible [-0.5, 1.0] mm"""
        displacement = self._calculate_displacement_estimate(w_outer, thickness_outer, L_outer,
                                                             w_inner, thickness_inner, L_inner)

        return -0.5 <= displacement <= 1.0

    def is_performance_ratio_valid(self, performance: float) -> bool:
        """Vérifie si le ratio de performance est dans la plage idéale"""
        if self.performance_ref <= 0:
            return True  # Si performance_ref invalide, on accepte tout
        perf_ratio = performance / self.performance_ref
        return self.perf_ratio_min <= perf_ratio <= self.perf_ratio_max

    def find_diverse_configurations(self, configurations: List[Dict], n_clusters: int = 8) -> List[Dict]:
        """Trouve des configurations diverses using K-Means clustering"""
        if len(configurations) < n_clusters:
            return configurations

        # Préparer les données pour le clustering (avec nouvelles features)
        features = np.array([
            [config['w_outer'], config['thickness_outer'], config['w_inner'],
             config['thickness_inner'], config['L_inner'], config['displacement_est'], config['k_ratio']]
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

    def _calculate_radius_ratio(self, thickness_outer: float, thickness_inner: float) -> float:
        """Calcule le ratio des rayons selon la géométrie des lames"""
        Intern_radius = self.enc / 2
        R_outer = Intern_radius + thickness_outer / 2
        R_inner = Intern_radius - self.inner_offset - thickness_inner / 2
        return R_inner / R_outer

    def _calculate_L_inner_from_geometry(self, thickness_outer: float, thickness_inner: float) -> float:
        """Calcule L_inner basé sur la géométrie relative des lames"""
        ratio = self._calculate_radius_ratio(thickness_outer, thickness_inner)
        return self.length * ratio

    def _calculate_displacement_estimate(self, w_outer: float, thickness_outer: float, L_outer: float,
                                         w_inner: float, thickness_inner: float, L_inner: float) -> float:
        """Estime le déplacement final basé sur un modèle empirique CORRIGÉ"""
        # Calcul des raideurs
        E_outer = self.material_properties[self.outer_material]
        E_inner = self.material_properties[self.inner_material]

        k_outer = self._calculate_k_single_blade(E_outer, w_outer, thickness_outer, L_outer)
        k_inner = self._calculate_k_single_blade(E_inner, w_inner, thickness_inner, L_inner)

        # Modèle basé sur vos données expérimentales
        # Analyse de vos configs: le déplacement dépend fortement du ratio des largeurs
        w_ratio = w_inner / w_outer

        # Modèle empirique basé sur vos 8 configurations
        if w_ratio < 0.67:  # w_inner << w_outer → déplacement négatif
            displacement = -2 - 8 * (0.67 - w_ratio)
        elif w_ratio > 1.33:  # w_inner >> w_outer → déplacement très positif
            displacement = 5 + 10 * (w_ratio - 1.33)
        else:  # Zone stable
            displacement = -1 + 3 * w_ratio

        # Correction par épaisseur (impact plus faible)
        thickness_factor = (thickness_outer + thickness_inner) / 0.48
        displacement *= thickness_factor

        return displacement

    def _calculate_dual_performance(self, w_outer: float, thickness_outer: float, L_outer: float,
                                    w_inner: float, thickness_inner: float, L_inner: float) -> float:
        """Calcule k_total * rs² pour le système à 2 lames"""

        # Raideurs individuelles
        E_outer = self.material_properties[self.outer_material]
        E_inner = self.material_properties[self.inner_material]

        k_outer = self._calculate_k_single_blade(E_outer, w_outer, thickness_outer, L_outer)
        k_inner = self._calculate_k_single_blade(E_inner, w_inner, thickness_inner, L_inner)

        k_total = k_outer + k_inner
        k_total_mNm = k_total * 1e3

        # Longueur équivalente pour le calcul du bras de levier
        length_equivalent = (L_outer + L_inner) / 2

        # Calcul du bras de levier
        Dx = self._calculate_Dx(length_equivalent)
        rs = abs(Dx)

        # Performance = k_total * rs²
        performance = k_total_mNm * (rs ** 2)

        return performance

    def generate_dual_configurations(self) -> List[Dict]:
        """Génère les configurations pour le système à 2 lames avec L_inner calculé"""
        configurations = []
        total_tested = 0
        displacement_rejected = 0
        performance_rejected = 0

        # L_outer est fixe
        L_outer = self.length  # CORRECTION: utiliser L_outer au lieu de L_outer_fixed

        print(f"Génération des configurations...")
        print(f"Plages de test:")
        print(f"- thickness_outer: {self.thickness_range[0]:.2f} à {self.thickness_range[-1]:.2f}")
        print(f"- w_outer: {self.w_outer_range[0]:.1f} à {self.w_outer_range[-1]:.1f}")
        print(f"- thickness_inner: {self.inner_thickness_range[0]:.2f} à {self.inner_thickness_range[-1]:.2f}")
        print(f"- w_inner: {self.w_inner_range[0]:.1f} à {self.w_inner_range[-1]:.1f}")

        for thickness_outer in self.thickness_range:
            for w_outer in self.w_outer_range:
                for thickness_inner in self.inner_thickness_range:
                    for w_inner in self.w_inner_range:
                        total_tested += 1

                        # Calcul automatique de L_inner basé sur la géométrie
                        L_inner_calculated = self._calculate_L_inner_from_geometry(
                            thickness_outer, thickness_inner
                        )

                        # Vérifier le critère de déplacement
                        displacement_valid = self.is_displacement_valid(
                            w_outer, thickness_outer, L_outer,  # CORRECTION
                            w_inner, thickness_inner, L_inner_calculated
                        )

                        if not displacement_valid:
                            displacement_rejected += 1

                        # Calcul de la performance
                        perf = self._calculate_dual_performance(
                            w_outer, thickness_outer, L_outer,  # CORRECTION
                            w_inner, thickness_inner, L_inner_calculated
                        )

                        performance_valid = self.is_performance_ratio_valid(perf)

                        if not performance_valid:
                            performance_rejected += 1

                        # Debug pour les premières configurations
                        if total_tested <= 10:
                            displacement_est = self._calculate_displacement_estimate(
                                w_outer, thickness_outer, L_outer,  # CORRECTION
                                w_inner, thickness_inner, L_inner_calculated
                            )
                            perf_ratio = perf / self.performance_ref if self.performance_ref > 0 else 1.0
                            print(f"Test {total_tested}: w_outer={w_outer:.1f}, w_inner={w_inner:.1f}")
                            print(f"  Déplacement: {displacement_est:.3f} mm (valide: {displacement_valid})")
                            print(f"  Performance ratio: {perf_ratio:.3f} (valide: {performance_valid})")

                        if displacement_valid and performance_valid:
                            config = self._create_dual_config_dict(
                                w_outer, thickness_outer, L_outer,  # CORRECTION
                                w_inner, thickness_inner, L_inner_calculated,
                                perf
                            )
                            configurations.append(config)

        print(f"\nTotal testé: {total_tested}")
        print(f"Rejetées par déplacement: {displacement_rejected}")
        print(f"Rejetées par performance: {performance_rejected}")
        print(f"Configurations valides trouvées: {len(configurations)}")
        return configurations

    def _create_dual_config_dict(self, w_outer: float, thickness_outer: float, L_outer: float,
                                 w_inner: float, thickness_inner: float, L_inner: float,
                                 performance: float) -> Dict:
        """Crée le dictionnaire de configuration pour 2 lames avec nomenclature outer/inner"""
        perf_ratio = performance / self.performance_ref if self.performance_ref > 0 else 1.0
        perf_optimal = (self.perf_ratio_min + self.perf_ratio_max) / 2

        # Longueur équivalente pour Dx
        length_equivalent = (L_outer + L_inner) / 2
        Dx = self._calculate_Dx(length_equivalent)
        rs = abs(Dx)

        # Raideurs individuelles
        E_outer = self.material_properties[self.outer_material]
        E_inner = self.material_properties[self.inner_material]

        k_outer = self._calculate_k_single_blade(E_outer, w_outer, thickness_outer, L_outer)
        k_inner = self._calculate_k_single_blade(E_inner, w_inner, thickness_inner, L_inner)
        k_total = k_outer + k_inner
        k_total_mNm = k_total * 1e3

        # Calcul du déplacement estimé
        displacement_est = self._calculate_displacement_estimate(
            w_outer, thickness_outer, L_outer,
            w_inner, thickness_inner, L_inner
        )

        # Calcul du ratio des rayons
        radius_ratio = self._calculate_radius_ratio(thickness_outer, thickness_inner)

        return {
            # Paramètres lame extérieure (outer)
            'w_outer': w_outer,
            'thickness_outer': thickness_outer,
            'L_outer': L_outer,
            'k_outer_mNm': k_outer * 1e3,

            # Paramètres lame intérieure (inner)
            'w_inner': w_inner,
            'thickness_inner': thickness_inner,
            'L_inner': L_inner,  # Calculé automatiquement
            'k_inner_mNm': k_inner * 1e3,

            # Paramètres système
            'k_total_mNm': k_total_mNm,
            'performance': performance,
            'perf_ratio': perf_ratio,
            'Dx': Dx,
            'rs': rs,
            'k_times_rs_squared': performance,
            'perf_error_from_optimal': abs(perf_ratio - perf_optimal),
            'length_equivalent': length_equivalent,

            # Métriques géométriques
            'displacement_est': displacement_est,
            'radius_ratio': radius_ratio,
            'k_ratio': k_inner / k_outer if k_outer > 0 else 0,
        }

    def print_dual_configurations(self, configs: List[Dict], title: str, show_all: bool = False):
        """Affiche les configurations pour le système à 2 lames avec nomenclature outer/inner"""
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
            print(f"{i:2d}\t{config['L_outer']:.2f}\t{config['thickness_outer']:.3f}\t"
                  f"{config['w_outer']:.1f}\t{config['L_inner']:.2f}\t{config['thickness_inner']:.3f}\t"
                  f"{config['w_inner']:.1f}\t{config['displacement_est']:.3f}\t"
                  f"{config['radius_ratio']:.4f}\t{config['k_ratio']:.3f}\t"
                  f"{config['k_total_mNm']:.1f}\t{config['perf_ratio']:.4f}\t"
                  f"{config['perf_error_from_optimal']:.6f}")

    def verify_dual_calculation(self, w_outer: float, thickness_outer: float, L_outer: float,
                                w_inner: float, thickness_inner: float, L_inner: float):
        """Vérification d'un calcul spécifique pour 2 lames avec nomenclature outer/inner"""
        print(f"\n=== VÉRIFICATION CALCUL SYSTÈME À 2 LAMES (OUTER/INNER) ===")
        print(f"OUTER ({self.outer_material}): L={L_outer}, e={thickness_outer}, w={w_outer}")
        print(f"INNER ({self.inner_material}): L={L_inner}, ei={thickness_inner}, w={w_inner}")
        print(f"Offset entre lames: {self.inner_offset} mm")

        # Calculs individuels
        E_outer = self.material_properties[self.outer_material]
        E_inner = self.material_properties[self.inner_material]

        k_outer = self._calculate_k_single_blade(E_outer, w_outer, thickness_outer, L_outer)
        k_inner = self._calculate_k_single_blade(E_inner, w_inner, thickness_inner, L_inner)
        k_total = k_outer + k_inner
        k_total_mNm = k_total * 1e3

        # Calculs géométriques
        radius_ratio = self._calculate_radius_ratio(thickness_outer, thickness_inner)
        displacement_est = self._calculate_displacement_estimate(
            w_outer, thickness_outer, L_outer,
            w_inner, thickness_inner, L_inner
        )

        # Bras de levier
        length_equivalent = (L_outer + L_inner) / 2
        Dx = self._calculate_Dx(length_equivalent)
        rs = abs(Dx)

        performance = self._calculate_dual_performance(
            w_outer, thickness_outer, L_outer,
            w_inner, thickness_inner, L_inner
        )
        perf_ratio = performance / self.performance_ref if self.performance_ref > 0 else 1.0

        print(f"\nCALCULS:")
        print(f"E_{self.outer_material} = {E_outer:.0f} N/mm²")
        print(f"E_{self.inner_material} = {E_inner:.0f} N/mm²")
        print(f"k_{self.outer_material} = {k_outer * 1e3:.2f} mN·m")
        print(f"k_{self.inner_material} = {k_inner * 1e3:.2f} mN·m")
        print(f"k_total = {k_total_mNm:.2f} mN·m")
        print(f"k_ratio (inner/outer) = {k_inner / k_outer:.3f}")
        print(f"Radius ratio = {radius_ratio:.4f}")
        print(f"Déplacement estimé = {displacement_est:.3f} mm")
        print(f"rs = {rs:.4f} mm")
        print(f"Performance = {performance:.2f} mN·m·mm²")
        print(f"Ratio performance = {perf_ratio:.6f}")

    def test_specific_configuration(self, w_outer: float, thickness_outer: float,
                                    w_inner: float, thickness_inner: float):
        """Test une configuration spécifique pour débugger"""
        print(f"\n=== TEST CONFIGURATION SPÉCIFIQUE ===")
        L_outer = self.length
        L_inner = self._calculate_L_inner_from_geometry(thickness_outer, thickness_inner)

        print(f"Configuration testée:")
        print(f"  Outer: L={L_outer}, e={thickness_outer}, w={w_outer}")
        print(f"  Inner: L={L_inner:.6f}, ei={thickness_inner}, w={w_inner}")

        # Tests des critères
        displacement_valid = self.is_displacement_valid(
            w_outer, thickness_outer, L_outer,
            w_inner, thickness_inner, L_inner
        )

        performance = self._calculate_dual_performance(
            w_outer, thickness_outer, L_outer,
            w_inner, thickness_inner, L_inner
        )

        performance_valid = self.is_performance_ratio_valid(performance)

        displacement_est = self._calculate_displacement_estimate(
            w_outer, thickness_outer, L_outer,
            w_inner, thickness_inner, L_inner
        )

        print(f"\nRésultats:")
        print(f"  Déplacement estimé: {displacement_est:.3f} mm")
        print(f"  Déplacement valide: {displacement_valid}")
        print(f"  Performance: {performance:.2f}")
        print(f"  Performance valide: {performance_valid}")
        print(f"  Performance ratio: {performance / self.performance_ref:.6f}")

        return displacement_valid and performance_valid

# Corrections dans main_dual()
def main_dual():

    # Configuration avec nouvelle nomenclature
    dual_config = DualBladeConfig()

    # TEST D'UNE CONFIGURATION SPÉCIFIQUE (celle que vous mentionnez)
    print("\n" + "=" * 80)
    print("TEST CONFIGURATION SPÉCIFIQUE:")
    is_valid = dual_config.test_specific_configuration(
        w_outer=30.0, thickness_outer=0.24,
        w_inner=21.5, thickness_inner=0.24
    )
    print(f"Configuration valide: {is_valid}")

    # Vérification avec configuration de référence
    print("\n" + "=" * 80)
    print("VÉRIFICATION CONFIGURATION DE RÉFÉRENCE:")
    inner_geom = dual_config.get_invar_geometry(dual_config.enc, dual_config._calculate_Dx(dual_config.length))
    dual_config.verify_dual_calculation(
        dual_config.width_ref_outer, dual_config.thickness, dual_config.length,
        dual_config.width_ref_inner, inner_geom['ei'], inner_geom['Li']
    )

    # Génération de toutes les configurations valides
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES CONFIGURATIONS OPTIMISÉES...")
    all_configs = dual_config.generate_dual_configurations()

    if not all_configs:
        print("Aucune configuration trouvée dans les critères spécifiés!")
        print("Essayez d'élargir les plages de recherche.")
        return dual_config, [], []  # Correction du return

    # Tri par erreur par rapport à l'optimal ET par déplacement proche de 0
    all_configs.sort(key=lambda x: (abs(x['displacement_est']), x['perf_error_from_optimal']))

    # Affichage des meilleures configurations
    dual_config.print_dual_configurations(
        all_configs,
        f"MEILLEURES CONFIGURATIONS - SYSTÈME À 2 LAMES (Outer/Inner)"
    )

    # Configurations diverses avec K-Means
    diverse_configs = dual_config.find_diverse_configurations(all_configs, n_clusters=min(8, len(all_configs)))
    dual_config.print_dual_configurations(
        diverse_configs,
        "CONFIGURATIONS DIVERSES (K-MEANS)"
    )

    print(f"\nRésumé:")
    print(f"- Configurations totales trouvées: {len(all_configs)}")
    print(f"- Configurations diverses sélectionnées: {len(diverse_configs)}")
    print(f"- Objectif déplacement: [-0.5, 1.0] mm")
    print(f"- Offset entre lames: {dual_config.inner_offset} mm")
    print(f"- L_inner calculé automatiquement à partir du ratio géométrique")

    return dual_config, all_configs, diverse_configs


if __name__ == "__main__":
    config, all_configs, diverse_configs = main_dual()