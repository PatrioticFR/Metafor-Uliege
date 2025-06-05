import math
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional


class ConfigurationType(Enum):
    """Types de configurations possibles"""
    L_CONSTANT = "L_constant"  # L fixe, varier h et w
    H_CONSTANT = "h_constant"  # h fixe, varier L et w
    W_CONSTANT = "w_constant"  # w fixe, varier L et h


class VariationMode(Enum):
    """Modes de variation pour les paramètres"""
    HIGH_LOW = "high_low"  # Une valeur haute, une valeur basse
    LOW_HIGH = "low_high"  # Une valeur basse, une valeur haute
    OPTIMAL = "optimal"  # Recherche autour de la valeur optimale


class ModularBladeConfig:
    """Configuration modulaire pour l'optimisation de lames"""

    def __init__(self, material: str = 'BE_CU'):
        # Géométrie de référence
        self.thickness_ref = 0.24  # h de référence
        self.length_ref = 105.25  # L de référence
        self.width_ref = 45.0  # w de référence

        # Matériau
        self.material = material

        # Propriétés des matériaux (en N/mm²)
        self.material_properties = {
            'BE_CU': 131e3,
            'INVAR': 141e3,
            'INVAR36': 141e3,  # Alias pour Invar36
            'STEEL': 210e3
        }

        # Plages de recherche - ajustées pour inclure exactement les valeurs de référence
        self.w_range = np.arange(10, 45.0, 0.1)
        self.h_range = np.arange(0.05, 0.30, 0.01)
        # Ajuster L_range pour s'assurer que 105.25 soit inclus exactement
        L_start = 90.0
        L_end = 110.0
        L_step = 0.05
        # Créer le range et s'assurer que 105.25 est inclus
        self.L_range = np.arange(L_start, L_end + L_step, L_step)
        # Vérifier et ajouter 105.25 si nécessaire
        if 105.25 not in self.L_range:
            self.L_range = np.append(self.L_range, 105.25)
            self.L_range = np.sort(self.L_range)

        # Ratio k idéal (basé sur la référence BE_CU)
        self.k_ratio_min = 0.987
        self.k_ratio_max = 0.990

        # Référence k (calculée avec BE_CU par défaut)
        self.k_ref = self.calculate_k_reference_material('BE_CU')

        # Système rod+masse (FIXE)
        self.enc_ref = 57.32
        self.Dx_fixe = True

    def get_material_E(self, material: str = None) -> float:
        """Module d'Young du matériau (en N/mm²)"""
        if material is None:
            material = self.material

        if material not in self.material_properties:
            available_materials = list(self.material_properties.keys())
            raise ValueError(f"Matériau '{material}' non reconnu. Matériaux disponibles: {available_materials}")

        return self.material_properties[material]

    def calculate_k_reference_material(self, material: str) -> float:
        """Calcule la raideur k de référence pour un matériau donné"""
        E = self.get_material_E(material)
        # Formule correcte pour la raideur de flexion d'une poutre encastrée-libre
        return (math.pi ** 2 * E * self.width_ref * self.thickness_ref ** 3) / (12 * self.length_ref ** 3)

    def calculate_k_reference(self) -> float:
        """Calcule la raideur k de référence"""
        return self.calculate_k_reference_material(self.material)

    def calculate_k(self, width: float, thickness: float, length: float, material: str = None) -> float:
        """Calcule k pour des dimensions données"""
        E = self.get_material_E(material)
        return (math.pi ** 2 * E * width * thickness ** 3) / (12 * length ** 3)

    def is_k_ratio_valid(self, k_value: float) -> bool:
        """Vérifie si le ratio k est dans la plage idéale (par rapport à la référence BE_CU)"""
        k_ratio = k_value / self.k_ref
        return self.k_ratio_min <= k_ratio <= self.k_ratio_max

    def estimate_stabilization_position(self, k_value: float) -> float:
        """
        Estimation de la position de stabilisation basée sur k
        Points de référence connus :
        - k_ratio = 1.0 → stabilisation ≈ 0.9mm (référence originale)
        - k_ratio = 0.9921 → stabilisation ≈ 0.62mm (Be-Cu 105.25, 0.25, 39.5)
        """
        k_ratio = k_value / self.k_ref

        # Points de référence pour l'interpolation
        k_ratio_ref1 = 1.0  # référence originale
        stabilization_ref1 = 0.9  # mm

        k_ratio_ref2 = 0.9921  # nouvelle référence Be-Cu
        stabilization_ref2 = 0.62  # mm

        if k_ratio >= k_ratio_ref1:
            # Pour k_ratio >= 1.0, extrapolation linéaire
            # Pente estimée basée sur la tendance observée
            slope = (stabilization_ref1 - stabilization_ref2) / (k_ratio_ref1 - k_ratio_ref2)
            return stabilization_ref1 + (k_ratio - k_ratio_ref1) * slope
        else:
            # Pour k_ratio < 1.0, interpolation/extrapolation basée sur les deux points connus
            slope = (stabilization_ref1 - stabilization_ref2) / (k_ratio_ref1 - k_ratio_ref2)
            return stabilization_ref1 + (k_ratio - k_ratio_ref1) * slope

    def generate_L_constant_configs(self, L_fixed: float, mode: VariationMode) -> List[Dict]:
        """Génère des configurations avec L constant"""
        configurations = []

        for h in self.h_range:
            for w in self.w_range:
                k_calc = self.calculate_k(w, h, L_fixed)

                if self.is_k_ratio_valid(k_calc):
                    config = self._create_config_dict(w, h, L_fixed, k_calc, ConfigurationType.L_CONSTANT)
                    configurations.append(config)

        return self._filter_by_mode(configurations, mode, 'thickness', 'width')

    def generate_h_constant_configs(self, h_fixed: float, mode: VariationMode) -> List[Dict]:
        """Génère des configurations avec h constant"""
        configurations = []

        for L in self.L_range:
            for w in self.w_range:
                k_calc = self.calculate_k(w, h_fixed, L)

                if self.is_k_ratio_valid(k_calc):
                    config = self._create_config_dict(w, h_fixed, L, k_calc, ConfigurationType.H_CONSTANT)
                    configurations.append(config)

        return self._filter_by_mode(configurations, mode, 'length', 'width')

    def generate_w_constant_configs(self, w_fixed: float, mode: VariationMode) -> List[Dict]:
        """Génère des configurations avec w constant"""
        configurations = []

        for L in self.L_range:
            for h in self.h_range:
                k_calc = self.calculate_k(w_fixed, h, L)

                if self.is_k_ratio_valid(k_calc):
                    config = self._create_config_dict(w_fixed, h, L, k_calc, ConfigurationType.W_CONSTANT)
                    configurations.append(config)

        return self._filter_by_mode(configurations, mode, 'length', 'thickness')

    def _create_config_dict(self, width: float, thickness: float, length: float,
                            k_value: float, config_type: ConfigurationType) -> Dict:
        """Crée un dictionnaire de configuration"""
        return {
            'width': width,
            'thickness': thickness,
            'length': length,
            'k': k_value,
            'k_ratio': k_value / self.k_ref,
            'estimated_stabilization': self.estimate_stabilization_position(k_value),
            'config_type': config_type,
            'k_error_from_optimal': abs(k_value / self.k_ref - ((self.k_ratio_min + self.k_ratio_max) / 2)),
            'material': self.material
        }

    def _filter_by_mode(self, configurations: List[Dict], mode: VariationMode,
                        param1: str, param2: str) -> List[Dict]:
        """Filtre les configurations selon le mode de variation"""
        if not configurations:
            return []

        if mode == VariationMode.OPTIMAL:
            # Trier par erreur par rapport au ratio k optimal
            configurations.sort(key=lambda x: x['k_error_from_optimal'])
            return configurations[:20]  # Top 20

        elif mode == VariationMode.HIGH_LOW:
            # Configurations avec param1 haut et param2 bas
            configs_filtered = []
            param1_values = sorted(set(config[param1] for config in configurations))
            param2_values = sorted(set(config[param2] for config in configurations))

            # Prendre le quartile supérieur pour param1 et inférieur pour param2
            param1_threshold = np.percentile(param1_values, 75)
            param2_threshold = np.percentile(param2_values, 25)

            for config in configurations:
                if config[param1] >= param1_threshold and config[param2] <= param2_threshold:
                    configs_filtered.append(config)

            configs_filtered.sort(key=lambda x: x['k_error_from_optimal'])
            return configs_filtered[:10]

        elif mode == VariationMode.LOW_HIGH:
            # Configurations avec param1 bas et param2 haut
            configs_filtered = []
            param1_values = sorted(set(config[param1] for config in configurations))
            param2_values = sorted(set(config[param2] for config in configurations))

            # Prendre le quartile inférieur pour param1 et supérieur pour param2
            param1_threshold = np.percentile(param1_values, 25)
            param2_threshold = np.percentile(param2_values, 75)

            for config in configurations:
                if config[param1] <= param1_threshold and config[param2] >= param2_threshold:
                    configs_filtered.append(config)

            configs_filtered.sort(key=lambda x: x['k_error_from_optimal'])
            return configs_filtered[:10]

        return configurations

    def find_all_configurations(self, materials: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
        """Trouve toutes les configurations possibles pour tous les matériaux"""
        if materials is None:
            materials = [self.material]

        all_results = {}

        for material in materials:
            print(f"\n{'=' * 80}")
            print(f"RECHERCHE POUR LE MATÉRIAU: {material}")
            print(f"{'=' * 80}")

            # Changer temporairement le matériau
            original_material = self.material
            self.material = material

            print(f"Module d'élasticité: {self.get_material_E():.0f} N/mm²")
            print(f"Ratio k idéal: {self.k_ratio_min:.3f} - {self.k_ratio_max:.3f}")
            print(f"k_référence (BE_CU): {self.k_ref:.6f}")

            all_configs = {}

            # 1. L constant (référence)
            L_fixed = self.length_ref
            print(f"\n1. L CONSTANT = {L_fixed:.2f} mm")

            all_configs['L_constant_optimal'] = self.generate_L_constant_configs(L_fixed, VariationMode.OPTIMAL)
            all_configs['L_constant_h_low_w_high'] = self.generate_L_constant_configs(L_fixed, VariationMode.LOW_HIGH)
            all_configs['L_constant_h_high_w_low'] = self.generate_L_constant_configs(L_fixed, VariationMode.HIGH_LOW)

            # 2. h constant (référence)
            h_fixed = self.thickness_ref
            print(f"\n2. h CONSTANT = {h_fixed:.3f} mm")

            all_configs['h_constant_optimal'] = self.generate_h_constant_configs(h_fixed, VariationMode.OPTIMAL)
            all_configs['h_constant_L_low_w_high'] = self.generate_h_constant_configs(h_fixed, VariationMode.LOW_HIGH)
            all_configs['h_constant_L_high_w_low'] = self.generate_h_constant_configs(h_fixed, VariationMode.HIGH_LOW)

            # 3. w constant (référence)
            w_fixed = self.width_ref
            print(f"\n3. w CONSTANT = {w_fixed:.1f} mm")

            all_configs['w_constant_optimal'] = self.generate_w_constant_configs(w_fixed, VariationMode.OPTIMAL)
            all_configs['w_constant_L_low_h_high'] = self.generate_w_constant_configs(w_fixed, VariationMode.LOW_HIGH)
            all_configs['w_constant_L_high_h_low'] = self.generate_w_constant_configs(w_fixed, VariationMode.HIGH_LOW)

            all_results[material] = all_configs

            # Restaurer le matériau original
            self.material = original_material

        return all_results

    def print_configuration_summary(self, all_configs: Dict[str, List[Dict]], material: str):
        """Affiche un résumé de toutes les configurations pour un matériau"""
        print(f"\n{'=' * 80}")
        print(f"RÉSUMÉ DES CONFIGURATIONS TROUVÉES - {material}")
        print(f"{'=' * 80}")

        for config_name, configs in all_configs.items():
            if configs:
                print(f"\n{config_name.upper().replace('_', ' ')}: {len(configs)} configurations")

                # Afficher les 3 meilleures
                for i, config in enumerate(configs[:3], 1):
                    print(f"  {i}. L={config['length']:.2f}, h={config['thickness']:.3f}, "
                          f"w={config['width']:.1f} → k_ratio={config['k_ratio']:.4f} "
                          f"(stab≈{config['estimated_stabilization']:+.1f}mm)")
            else:
                print(f"\n{config_name.upper().replace('_', ' ')}: Aucune configuration trouvée")

    def print_detailed_configs(self, configs: List[Dict], title: str, material: str):
        """Affiche les configurations détaillées"""
        if not configs:
            print(f"\n=== {title} - {material} ===")
            print("Aucune configuration trouvée.")
            return

        print(f"\n=== {title} - {material} ({len(configs)} configurations) ===")
        print("Rang\tL(mm)\t\th(mm)\tw(mm)\tk_ratio\tStab_est(mm)\tType")
        print("-" * 75)

        for i, config in enumerate(configs, 1):
            print(f"{i:2d}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['k_ratio']:.4f}\t"
                  f"{config['estimated_stabilization']:+.1f}\t\t{config['config_type'].value}")

    def get_physical_clamping_parameters(self, length: float) -> Tuple[float, float, float]:
        """Paramètres de serrage adaptés à la longueur"""
        R = length / math.pi
        enc_opt = self.enc_ref  # FIXE

        if self.Dx_fixe:
            Dx_opt = -67.5227
        else:
            Dx_opt = -2 * R

        return Dx_opt, 0.0, enc_opt

    def print_key_configurations_summary(self, all_configs: Dict[str, List[Dict]], material: str):
        """Affiche les configurations extrêmes pour chaque type"""
        print(f"\n{'=' * 80}")
        print(f"CONFIGURATIONS EXTRÊMES PAR TYPE - {material}")
        print(f"{'=' * 80}")

        # L CONSTANT - extrêmes sur h (qui correspondent aux extrêmes sur w)
        l_configs = all_configs.get('L_constant_optimal', [])
        if l_configs:
            print(f"\nL CONSTANT (L={self.length_ref:.2f}mm):")

            # Trouver h min et h max
            h_min_config = min(l_configs, key=lambda x: x['thickness'])
            h_max_config = max(l_configs, key=lambda x: x['thickness'])

            print(f"  h MIN → L={h_min_config['length']:.2f}, h={h_min_config['thickness']:.3f}, "
                  f"w={h_min_config['width']:.1f}, k_ratio={h_min_config['k_ratio']:.4f}, "
                  f"stab={h_min_config['estimated_stabilization']:+.1f}mm")
            print(f"  h MAX → L={h_max_config['length']:.2f}, h={h_max_config['thickness']:.3f}, "
                  f"w={h_max_config['width']:.1f}, k_ratio={h_max_config['k_ratio']:.4f}, "
                  f"stab={h_max_config['estimated_stabilization']:+.1f}mm")
        else:
            print(f"\nL CONSTANT (L={self.length_ref:.2f}mm): Aucune configuration valide.")

        # h CONSTANT - extrêmes sur L et w
        h_configs = all_configs.get('h_constant_optimal', [])
        if h_configs:
            print(f"\nh CONSTANT (h={self.thickness_ref:.3f}mm):")

            # Trouver L min et L max
            L_min_config = min(h_configs, key=lambda x: x['length'])
            L_max_config = max(h_configs, key=lambda x: x['length'])

            # Trouver w min et w max
            w_min_config = min(h_configs, key=lambda x: x['width'])
            w_max_config = max(h_configs, key=lambda x: x['width'])

            print(f"  L MIN → L={L_min_config['length']:.2f}, h={L_min_config['thickness']:.3f}, "
                  f"w={L_min_config['width']:.1f}, k_ratio={L_min_config['k_ratio']:.4f}, "
                  f"stab={L_min_config['estimated_stabilization']:+.1f}mm")
            print(f"  L MAX → L={L_max_config['length']:.2f}, h={L_max_config['thickness']:.3f}, "
                  f"w={L_max_config['width']:.1f}, k_ratio={L_max_config['k_ratio']:.4f}, "
                  f"stab={L_max_config['estimated_stabilization']:+.1f}mm")
            print(f"  w MIN → L={w_min_config['length']:.2f}, h={w_min_config['thickness']:.3f}, "
                  f"w={w_min_config['width']:.1f}, k_ratio={w_min_config['k_ratio']:.4f}, "
                  f"stab={w_min_config['estimated_stabilization']:+.1f}mm")
            print(f"  w MAX → L={w_max_config['length']:.2f}, h={w_max_config['thickness']:.3f}, "
                  f"w={w_max_config['width']:.1f}, k_ratio={w_max_config['k_ratio']:.4f}, "
                  f"stab={w_max_config['estimated_stabilization']:+.1f}mm")
        else:
            print(f"\nh CONSTANT (h={self.thickness_ref:.3f}mm): Aucune configuration valide.")

        # w CONSTANT - extrêmes sur L et h
        w_configs = all_configs.get('w_constant_optimal', [])
        if w_configs:
            print(f"\nw CONSTANT (w={self.width_ref:.1f}mm):")

            # Trouver L min et L max
            L_min_config = min(w_configs, key=lambda x: x['length'])
            L_max_config = max(w_configs, key=lambda x: x['length'])

            # Trouver h min et h max
            h_min_config = min(w_configs, key=lambda x: x['thickness'])
            h_max_config = max(w_configs, key=lambda x: x['thickness'])

            print(f"  L MIN → L={L_min_config['length']:.2f}, h={L_min_config['thickness']:.3f}, "
                  f"w={L_min_config['width']:.1f}, k_ratio={L_min_config['k_ratio']:.4f}, "
                  f"stab={L_min_config['estimated_stabilization']:+.1f}mm")
            print(f"  L MAX → L={L_max_config['length']:.2f}, h={L_max_config['thickness']:.3f}, "
                  f"w={L_max_config['width']:.1f}, k_ratio={L_max_config['k_ratio']:.4f}, "
                  f"stab={L_max_config['estimated_stabilization']:+.1f}mm")
            print(f"  h MIN → L={h_min_config['length']:.2f}, h={h_min_config['thickness']:.3f}, "
                  f"w={h_min_config['width']:.1f}, k_ratio={h_min_config['k_ratio']:.4f}, "
                  f"stab={h_min_config['estimated_stabilization']:+.1f}mm")
            print(f"  h MAX → L={h_max_config['length']:.2f}, h={h_max_config['thickness']:.3f}, "
                  f"w={h_max_config['width']:.1f}, k_ratio={h_max_config['k_ratio']:.4f}, "
                  f"stab={h_max_config['estimated_stabilization']:+.1f}mm")
        else:
            print(f"\nw CONSTANT (w={self.width_ref:.1f}mm): Aucune configuration valide.")

    def print_comparison_summary(self, all_results: Dict[str, Dict[str, List[Dict]]]):
        """Affiche un résumé comparatif entre matériaux"""
        print(f"\n{'=' * 80}")
        print("COMPARAISON ENTRE MATÉRIAUX")
        print(f"{'=' * 80}")

        materials = list(all_results.keys())
        config_types = ['L_constant_optimal', 'h_constant_optimal', 'w_constant_optimal']

        for config_type in config_types:
            print(f"\n{config_type.upper().replace('_', ' ')}:")
            print("-" * 50)

            for material in materials:
                configs = all_results[material].get(config_type, [])
                if configs:
                    best_config = configs[0]  # Le meilleur (déjà trié)
                    print(f"{material:10} → L={best_config['length']:.2f}, h={best_config['thickness']:.3f}, "
                          f"w={best_config['width']:.1f}, k_ratio={best_config['k_ratio']:.4f}")
                else:
                    print(f"{material:10} → Aucune configuration")


def main():
    """Fonction principale pour tester toutes les configurations avec différents matériaux"""

    # Liste des matériaux à tester
    materials_to_test = ['BE_CU', 'INVAR', 'STEEL']

    # Créer l'instance avec le premier matériau
    blade_config = ModularBladeConfig(material=materials_to_test[0])

    # Recherche de toutes les configurations pour tous les matériaux
    all_results = blade_config.find_all_configurations(materials_to_test)

    # Affichage des résultats pour chaque matériau
    for material, all_configs in all_results.items():
        # Résumé général
        blade_config.print_configuration_summary(all_configs, material)

        # Affichage détaillé des meilleures configurations de chaque type
        print(f"\n{'=' * 80}")
        print(f"CONFIGURATIONS DÉTAILLÉES - {material}")
        print(f"{'=' * 80}")

        # Configurations optimales
        blade_config.print_detailed_configs(
            all_configs['L_constant_optimal'],
            "L CONSTANT - CONFIGURATIONS OPTIMALES", material
        )

        blade_config.print_detailed_configs(
            all_configs['h_constant_optimal'],
            "h CONSTANT - CONFIGURATIONS OPTIMALES", material
        )

        blade_config.print_detailed_configs(
            all_configs['w_constant_optimal'],
            "w CONSTANT - CONFIGURATIONS OPTIMALES", material
        )

        # Affichage des configurations extrêmes par type
        blade_config.print_key_configurations_summary(all_configs, material)

    # Comparaison finale entre matériaux
    blade_config.print_comparison_summary(all_results)

    return all_results


if __name__ == "__main__":
    configurations = main()