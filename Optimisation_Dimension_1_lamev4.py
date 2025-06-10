import math
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple


class ConfigurationType(Enum):
    """Types de configurations possibles"""
    L_CONSTANT = "L_constant"
    H_CONSTANT = "h_constant"
    W_CONSTANT = "w_constant"


class VariationMode(Enum):
    """Modes de variation pour les paramètres"""
    OPTIMAL = "optimal"
    HIGH_LOW = "high_low"
    LOW_HIGH = "low_high"


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
            'INVAR36': 141e3,
            'STEEL': 210e3
        }

        # Plages de recherche
        self.w_range = np.arange(10, 60.1, 0.1)
        self.h_range = np.arange(0.05, 0.31, 0.01)
        self.L_range = np.arange(93.0, 110.1, 0.05)

        # Ratio k idéal (basé sur la référence BE_CU)
        self.k_ratio_min = 1.040
        self.k_ratio_max = 1.050

        # CORRECTION: Utiliser la valeur directement de votre simulation
        # Votre logiciel donne ratio=1 pour L=105.25, h=0.24, w=45.0
        self.k_ref = 0.914470347  # Valeur de référence de votre simulation

        # Calculer le facteur de correction entre formule théorique et simulation
        k_theoretical = self.calculate_k_theoretical(self.width_ref, self.thickness_ref, self.length_ref, 'BE_CU')
        self.correction_factor = self.k_ref / k_theoretical

        print(f"k_ref (simulation): {self.k_ref:.9f}")
        print(f"k_theoretical: {k_theoretical:.9f}")
        print(f"Facteur de correction: {self.correction_factor:.6f}")

    def get_material_E(self, material: str = None) -> float:
        """Module d'Young du matériau (en N/mm²)"""
        if material is None:
            material = self.material

        if material not in self.material_properties:
            available_materials = list(self.material_properties.keys())
            raise ValueError(f"Matériau '{material}' non reconnu. Matériaux disponibles: {available_materials}")

        return self.material_properties[material]

    def calculate_k_theoretical(self, width: float, thickness: float, length: float, material: str = None) -> float:
        """Calcule k théorique avec la formule classique"""
        E = self.get_material_E(material)
        # Formule théorique : k = (π² × E × b × h³) / (24 × L³)
        return (math.pi ** 2 * E * width * thickness ** 3) / (24 * length ** 3)

    def calculate_k(self, width: float, thickness: float, length: float, material: str = None) -> float:
        """Calcule k corrigé pour correspondre à votre simulation"""
        k_theoretical = self.calculate_k_theoretical(width, thickness, length, material)
        return k_theoretical * self.correction_factor

    def is_k_ratio_valid(self, k_value: float) -> bool:
        """Vérifie si le ratio k est dans la plage idéale"""
        k_ratio = k_value / self.k_ref
        return self.k_ratio_min <= k_ratio <= self.k_ratio_max

    def generate_configs(self, config_type: ConfigurationType, fixed_value: float, mode: VariationMode) -> List[Dict]:
        """Génère des configurations selon le type"""
        configurations = []

        if config_type == ConfigurationType.L_CONSTANT:
            # L fixe, varier h et w
            for h in self.h_range:
                for w in self.w_range:
                    k_calc = self.calculate_k(w, h, fixed_value)
                    if self.is_k_ratio_valid(k_calc):
                        config = self._create_config_dict(w, h, fixed_value, k_calc, config_type)
                        configurations.append(config)
            return self._filter_by_mode(configurations, mode, 'thickness', 'width')

        elif config_type == ConfigurationType.H_CONSTANT:
            # h fixe, varier L et w
            for L in self.L_range:
                for w in self.w_range:
                    k_calc = self.calculate_k(w, fixed_value, L)
                    if self.is_k_ratio_valid(k_calc):
                        config = self._create_config_dict(w, fixed_value, L, k_calc, config_type)
                        configurations.append(config)
            return self._filter_by_mode(configurations, mode, 'length', 'width')

        elif config_type == ConfigurationType.W_CONSTANT:
            # w fixe, varier L et h
            for L in self.L_range:
                for h in self.h_range:
                    k_calc = self.calculate_k(fixed_value, h, L)
                    if self.is_k_ratio_valid(k_calc):
                        config = self._create_config_dict(fixed_value, h, L, k_calc, config_type)
                        configurations.append(config)
            return self._filter_by_mode(configurations, mode, 'length', 'thickness')

        return []

    def _create_config_dict(self, width: float, thickness: float, length: float,
                            k_value: float, config_type: ConfigurationType) -> Dict:
        """Crée un dictionnaire de configuration"""
        k_ratio = k_value / self.k_ref
        k_optimal = (self.k_ratio_min + self.k_ratio_max) / 2

        return {
            'width': width,
            'thickness': thickness,
            'length': length,
            'k': k_value,
            'k_ratio': k_ratio,
            'config_type': config_type,
            'k_error_from_optimal': abs(k_ratio - k_optimal),
            'material': self.material
        }

    def _filter_by_mode(self, configurations: List[Dict], mode: VariationMode,
                        param1: str, param2: str) -> List[Dict]:
        """Filtre les configurations selon le mode de variation"""
        if not configurations:
            return []

        if mode == VariationMode.OPTIMAL:
            configurations.sort(key=lambda x: x['k_error_from_optimal'])
            return configurations[:20]

        elif mode == VariationMode.HIGH_LOW:
            param1_values = sorted(set(config[param1] for config in configurations))
            param2_values = sorted(set(config[param2] for config in configurations))

            param1_threshold = np.percentile(param1_values, 75)
            param2_threshold = np.percentile(param2_values, 25)

            configs_filtered = [config for config in configurations
                                if config[param1] >= param1_threshold and config[param2] <= param2_threshold]

            configs_filtered.sort(key=lambda x: x['k_error_from_optimal'])
            return configs_filtered[:10]

        elif mode == VariationMode.LOW_HIGH:
            param1_values = sorted(set(config[param1] for config in configurations))
            param2_values = sorted(set(config[param2] for config in configurations))

            param1_threshold = np.percentile(param1_values, 25)
            param2_threshold = np.percentile(param2_values, 75)

            configs_filtered = [config for config in configurations
                                if config[param1] <= param1_threshold and config[param2] >= param2_threshold]

            configs_filtered.sort(key=lambda x: x['k_error_from_optimal'])
            return configs_filtered[:10]

        return configurations

    def find_all_configurations(self, materials: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
        """Trouve toutes les configurations possibles pour tous les matériaux"""
        if materials is None:
            materials = [self.material]

        all_results = {}

        for material in materials:
            print(f"\n{'=' * 60}")
            print(f"MATÉRIAU: {material}")
            print(f"{'=' * 60}")

            original_material = self.material
            self.material = material

            print(f"Module d'élasticité: {self.get_material_E():.0f} N/mm²")
            print(f"Ratio k idéal: {self.k_ratio_min:.3f} - {self.k_ratio_max:.3f}")

            all_configs = {}

            # L constant
            all_configs['L_constant_optimal'] = self.generate_configs(
                ConfigurationType.L_CONSTANT, self.length_ref, VariationMode.OPTIMAL)

            # h constant
            all_configs['h_constant_optimal'] = self.generate_configs(
                ConfigurationType.H_CONSTANT, self.thickness_ref, VariationMode.OPTIMAL)

            # w constant
            all_configs['w_constant_optimal'] = self.generate_configs(
                ConfigurationType.W_CONSTANT, self.width_ref, VariationMode.OPTIMAL)

            all_results[material] = all_configs
            self.material = original_material

        return all_results

    def print_configurations(self, configs: List[Dict], title: str):
        """Affiche les configurations"""
        if not configs:
            print(f"\n=== {title} ===")
            print("Aucune configuration trouvée.")
            return

        print(f"\n=== {title} ({len(configs)} configurations) ===")
        print("Rang\tL(mm)\t\th(mm)\tw(mm)\tk_ratio\tErreur")
        print("-" * 65)

        for i, config in enumerate(configs[:10], 1):  # Top 10
            print(f"{i:2d}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['k_ratio']:.4f}\t"
                  f"{config['k_error_from_optimal']:.6f}")

    def verify_calculation(self, length: float, thickness: float, width: float, material: str = None):
        """Vérifie un calcul spécifique"""
        if material is None:
            material = self.material

        print(f"\n=== VÉRIFICATION CALCUL ===")
        print(f"Paramètres: L={length}, h={thickness}, w={width}, matériau={material}")

        E = self.get_material_E(material)
        k_theoretical = self.calculate_k_theoretical(width, thickness, length, material)
        k_calc = self.calculate_k(width, thickness, length, material)
        k_ratio = k_calc / self.k_ref

        print(f"E = {E:.0f} N/mm²")
        print(f"k théorique = {k_theoretical:.9f}")
        print(f"k corrigé = {k_calc:.9f}")
        print(f"k_ref = {self.k_ref:.9f}")
        print(f"k_ratio = {k_ratio:.6f}")
        print(f"Dans la plage acceptable? {self.is_k_ratio_valid(k_calc)}")

    def verify_simulation_data(self):
        """Vérifie les données de votre simulation"""
        print("\n" + "=" * 80)
        print("VÉRIFICATION DES DONNÉES DE SIMULATION")
        print("=" * 80)

        # Cas 1: Configuration de référence
        print("\nCas 1: Configuration de référence")
        self.verify_calculation(105.25, 0.24, 45.0, 'BE_CU')

        # Cas 2: L=104, ratio=1.013
        print("\nCas 2: L=104, ratio attendu=1.013")
        self.verify_calculation(104.0, 0.24, 45.0, 'BE_CU')

        # Cas 3: L=90, w=28.7, ratio=1.02
        print("\nCas 3: L=90, w=28.7, ratio attendu=1.02")
        self.verify_calculation(90.0, 0.24, 28.7, 'BE_CU')


def main():
    """Fonction principale corrigée"""
    materials_to_test = ['BE_CU', 'INVAR', 'STEEL']

    blade_config = ModularBladeConfig(material=materials_to_test[0])

    # Vérification des données de simulation
    blade_config.verify_simulation_data()

    # Recherche des configurations
    all_results = blade_config.find_all_configurations(materials_to_test)

    # Affichage des résultats
    for material, all_configs in all_results.items():
        print(f"\n{'=' * 80}")
        print(f"RÉSULTATS POUR {material}")
        print(f"{'=' * 80}")

        blade_config.print_configurations(
            all_configs['L_constant_optimal'],
            f"L CONSTANT = {blade_config.length_ref:.2f}mm - {material}"
        )

        blade_config.print_configurations(
            all_configs['h_constant_optimal'],
            f"h CONSTANT = {blade_config.thickness_ref:.3f}mm - {material}"
        )

        blade_config.print_configurations(
            all_configs['w_constant_optimal'],
            f"w CONSTANT = {blade_config.width_ref:.1f}mm - {material}"
        )

    return all_results


if __name__ == "__main__":
    configurations = main()