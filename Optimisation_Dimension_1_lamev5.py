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
    """Configuration modulaire pour l'optimisation de lames avec effet bras de levier"""

    def __init__(self, material: str = 'BE_CU', Dx_fixe: bool = False):
        # Géométrie de référence
        self.thickness_ref = 0.24  # h de référence
        self.length_ref = 105.25  # L de référence
        self.width_ref = 45.0  # w de référence
        self.enc_ref = 57.32  # Position encodeur sur la masse

        # Mode de calcul Dx
        self.Dx_fixe = Dx_fixe
        self.Dx_fixed_value = -67.5227  # Valeur fixe si Dx_fixe = True

        # Matériau
        self.material = material

        # Propriétés des matériaux (en N/mm²)
        self.material_properties = {
            'BE_CU': 131e3,
            'INVAR': 141e3,
            'INVAR36': 141e3,
            'STEEL': 210e3
        }

        # Paramètres géométriques du système rod/mass
        self.H = 3.875  # rod vertical thickness (mm)
        self.D = 39.99  # mass block height (mm)
        self.d = 13.96  # mass block length (mm)
        self.l = 79.2  # total rod length (mm)
        self.r = 7.0  # rod right segment (mm)
        self.R_rod = self.H  # rod right width
        self.rho = 7.85e-6  # steel density (kg/mm³)
        self.depth = 63.0  # structure width/depth (mm)

        # Paramètres du système flexure
        self.k_flex = 44.3  # mN·m
        self.r_s = 65.22  # mm

        # Plages de recherche
        self.w_range = np.arange(10, 60.1, 0.1)
        self.h_range = np.arange(0.05, 0.41, 0.01)
        self.L_range = np.arange(93.0, 110.1, 0.05)

        # Calcul de l'inertie du système (constante)
        self.I_system = self._calculate_system_inertia()

        # Calcul de la performance de référence (basée sur la géométrie de référence)
        self.performance_ref = self._calculate_performance(self.width_ref, self.thickness_ref, self.length_ref)

        # Ratio de performance idéal (basé sur vos spécifications originales)
        self.perf_ratio_min = 1.030
        self.perf_ratio_max = 1.040

        print(f"Configuration: Dx_fixe = {self.Dx_fixe}")
        print(f"Performance de référence: {self.performance_ref:.6f}")
        print(f"Inertie système: {self.I_system:.3f} kg·mm²")

    def _calculate_system_inertia(self) -> float:
        """Calcule l'inertie totale du système rod/mass"""
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

    def get_material_E(self, material: str = None) -> float:
        """Module d'Young du matériau (en N/mm²)"""
        if material is None:
            material = self.material
        return self.material_properties[material]

    def _calculate_Dx(self, length: float) -> float:
        """Calcule Dx selon le mode choisi"""
        if self.Dx_fixe:
            return self.Dx_fixed_value
        else:
            R = length / math.pi
            return -2 * R

    def _calculate_lever_arm(self, length: float) -> float:
        """Calcule le bras de levier effectif"""
        Dx = self._calculate_Dx(length)
        # Le bras de levier est la distance entre enc_ref et le point de clampage
        return abs(self.enc_ref - abs(Dx))

    def _calculate_performance(self, width: float, thickness: float, length: float, material: str = None) -> float:
        """Calcule la performance du système (intègre raideur et effet bras de levier)"""
        E = self.get_material_E(material)

        # Raideur théorique de la lame [N·mm]
        kth = (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3
        kth_mNm = kth * 1e3  # Conversion en mN·m

        # Bras de levier effectif
        lever_arm = self._calculate_lever_arm(length)

        # Performance combinant raideur et effet bras de levier
        # Plus le bras de levier est grand, plus l'effet de la raideur est amplifié
        performance = kth_mNm * (lever_arm ** 2)

        return performance

    def is_performance_ratio_valid(self, performance: float) -> bool:
        """Vérifie si le ratio de performance est dans la plage idéale"""
        perf_ratio = performance / self.performance_ref
        return self.perf_ratio_min <= perf_ratio <= self.perf_ratio_max

    def generate_configs(self, config_type: ConfigurationType, fixed_value: float, mode: VariationMode) -> List[Dict]:
        """Génère des configurations selon le type"""
        configurations = []

        if config_type == ConfigurationType.L_CONSTANT:
            # L fixe, varier h et w
            for h in self.h_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, h, fixed_value)
                    if self.is_performance_ratio_valid(perf):
                        config = self._create_config_dict(w, h, fixed_value, perf, config_type)
                        configurations.append(config)
            return self._filter_by_mode(configurations, mode, 'thickness', 'width')

        elif config_type == ConfigurationType.H_CONSTANT:
            # h fixe, varier L et w
            for L in self.L_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, fixed_value, L)
                    if self.is_performance_ratio_valid(perf):
                        config = self._create_config_dict(w, fixed_value, L, perf, config_type)
                        configurations.append(config)
            return self._filter_by_mode(configurations, mode, 'length', 'width')

        elif config_type == ConfigurationType.W_CONSTANT:
            # w fixe, varier L et h
            for L in self.L_range:
                for h in self.h_range:
                    perf = self._calculate_performance(fixed_value, h, L)
                    if self.is_performance_ratio_valid(perf):
                        config = self._create_config_dict(fixed_value, h, L, perf, config_type)
                        configurations.append(config)
            return self._filter_by_mode(configurations, mode, 'length', 'thickness')

        return []

    def _create_config_dict(self, width: float, thickness: float, length: float,
                            performance: float, config_type: ConfigurationType) -> Dict:
        """Crée un dictionnaire de configuration"""
        perf_ratio = performance / self.performance_ref
        perf_optimal = (self.perf_ratio_min + self.perf_ratio_max) / 2
        Dx = self._calculate_Dx(length)
        lever_arm = self._calculate_lever_arm(length)

        # Calcul de la raideur pour information
        E = self.get_material_E()
        kth = (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3
        kth_mNm = kth * 1e3

        return {
            'width': width,
            'thickness': thickness,
            'length': length,
            'performance': performance,
            'perf_ratio': perf_ratio,
            'Dx': Dx,
            'lever_arm': lever_arm,
            'kth_mNm': kth_mNm,
            'config_type': config_type,
            'perf_error_from_optimal': abs(perf_ratio - perf_optimal),
            'material': self.material
        }

    def _filter_by_mode(self, configurations: List[Dict], mode: VariationMode,
                        param1: str, param2: str) -> List[Dict]:
        """Filtre les configurations selon le mode de variation"""
        if not configurations:
            return []

        if mode == VariationMode.OPTIMAL:
            configurations.sort(key=lambda x: x['perf_error_from_optimal'])
            return configurations[:20]

        elif mode == VariationMode.HIGH_LOW:
            param1_values = sorted(set(config[param1] for config in configurations))
            param2_values = sorted(set(config[param2] for config in configurations))

            param1_threshold = np.percentile(param1_values, 75)
            param2_threshold = np.percentile(param2_values, 25)

            configs_filtered = [config for config in configurations
                                if config[param1] >= param1_threshold and config[param2] <= param2_threshold]

            configs_filtered.sort(key=lambda x: x['perf_error_from_optimal'])
            return configs_filtered[:10]

        elif mode == VariationMode.LOW_HIGH:
            param1_values = sorted(set(config[param1] for config in configurations))
            param2_values = sorted(set(config[param2] for config in configurations))

            param1_threshold = np.percentile(param1_values, 25)
            param2_threshold = np.percentile(param2_values, 75)

            configs_filtered = [config for config in configurations
                                if config[param1] <= param1_threshold and config[param2] >= param2_threshold]

            configs_filtered.sort(key=lambda x: x['perf_error_from_optimal'])
            return configs_filtered[:10]

        return configurations

    def find_all_configurations(self) -> Dict[str, List[Dict]]:
        """Trouve toutes les configurations possibles"""
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

        return all_configs

    def _get_minmax_configs(self, configs: List[Dict], config_type: ConfigurationType) -> Dict:
        """Extrait les configurations min/max selon le type"""
        if not configs:
            return {}

        minmax_configs = {}

        if config_type == ConfigurationType.L_CONSTANT:
            # Pour L constant, on cherche h min et h max
            h_values = [config['thickness'] for config in configs]
            h_min = min(h_values)
            h_max = max(h_values)

            # Trouver les configurations correspondantes
            h_min_config = next(config for config in configs if config['thickness'] == h_min)
            h_max_config = next(config for config in configs if config['thickness'] == h_max)

            minmax_configs = {
                'h_min': h_min_config,
                'h_max': h_max_config
            }

        elif config_type == ConfigurationType.H_CONSTANT:
            # Pour h constant, on cherche L min et L max
            L_values = [config['length'] for config in configs]
            L_min = min(L_values)
            L_max = max(L_values)

            # Trouver les configurations correspondantes
            L_min_config = next(config for config in configs if config['length'] == L_min)
            L_max_config = next(config for config in configs if config['length'] == L_max)

            minmax_configs = {
                'L_min': L_min_config,
                'L_max': L_max_config
            }

        elif config_type == ConfigurationType.W_CONSTANT:
            # Pour w constant, on cherche L min et L max
            L_values = [config['length'] for config in configs]
            L_min = min(L_values)
            L_max = max(L_values)

            # Trouver les configurations correspondantes
            L_min_config = next(config for config in configs if config['length'] == L_min)
            L_max_config = next(config for config in configs if config['length'] == L_max)

            minmax_configs = {
                'L_min': L_min_config,
                'L_max': L_max_config
            }

        return minmax_configs

    def print_configurations_fixed_dx(self, configs: List[Dict], title: str):
        """Affiche les configurations pour Dx fixe"""
        if not configs:
            print(f"\n=== {title} ===")
            print("Aucune configuration trouvée.")
            return

        print(f"\n=== {title} ({len(configs)} configurations) ===")
        print("Rang\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\tBras(mm)\tperf_ratio\tErreur")
        print("-" * 85)

        for i, config in enumerate(configs[:20], 1):
            print(f"{i:2d}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['lever_arm']:.2f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

        # Affichage des valeurs min/max
        self._print_minmax_fixed_dx(configs, title)

    def print_configurations_variable_dx(self, configs: List[Dict], title: str):
        """Affiche les configurations pour Dx variable avec plus d'infos sur le bras de levier"""
        if not configs:
            print(f"\n=== {title} ===")
            print("Aucune configuration trouvée.")
            return

        print(f"\n=== {title} ({len(configs)} configurations) ===")
        print("Rang\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\tBras(mm)\tkth(mN·m)\tLever²\tperf_ratio\tErreur")
        print("-" * 110)

        # Calcul du bras de levier de référence pour comparaison
        lever_ref = self._calculate_lever_arm(self.length_ref)

        for i, config in enumerate(configs[:20], 1):
            lever_squared = config['lever_arm'] ** 2
            lever_ratio = config['lever_arm'] / lever_ref if lever_ref > 0 else 0

            print(f"{i:2d}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['lever_arm']:.2f}\t"
                  f"{config['kth_mNm']:.1f}\t\t{lever_squared:.1f}\t{config['perf_ratio']:.4f}\t"
                  f"{config['perf_error_from_optimal']:.6f}")

        # Affichage des valeurs min/max
        self._print_minmax_variable_dx(configs, title)

    def _print_minmax_fixed_dx(self, configs: List[Dict], title: str):
        """Affiche les valeurs min/max pour Dx fixe"""
        if not configs:
            return

        # Déterminer le type de configuration
        config_type = configs[0]['config_type']
        minmax_configs = self._get_minmax_configs(configs, config_type)

        if not minmax_configs:
            return

        # Extraire le nom de base du titre
        base_title = title.split('(')[0].strip()
        print(f"\n=== {base_title} (Valeurs Min / Max) ===")
        print("Info\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\tBras(mm)\tperf_ratio\tErreur")
        print("-" * 85)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            print(f"{label}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['lever_arm']:.2f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    def _print_minmax_variable_dx(self, configs: List[Dict], title: str):
        """Affiche les valeurs min/max pour Dx variable"""
        if not configs:
            return

        # Déterminer le type de configuration
        config_type = configs[0]['config_type']
        minmax_configs = self._get_minmax_configs(configs, config_type)

        if not minmax_configs:
            return

        # Extraire le nom de base du titre
        base_title = title.split('(')[0].strip()
        print(f"\n=== {base_title} (Valeurs Min / Max) ===")
        print("Info\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\tBras(mm)\tkth(mN·m)\tLever²\tperf_ratio\tErreur")
        print("-" * 110)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            lever_squared = config['lever_arm'] ** 2
            print(f"{label}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['lever_arm']:.2f}\t"
                  f"{config['kth_mNm']:.1f}\t\t{lever_squared:.1f}\t{config['perf_ratio']:.4f}\t"
                  f"{config['perf_error_from_optimal']:.6f}")

    def verify_calculation(self, length: float, thickness: float, width: float, material: str = None):
        """Vérifie un calcul spécifique"""
        if material is None:
            material = self.material

        print(f"\n=== VÉRIFICATION CALCUL ===")
        print(f"Paramètres: L={length}, h={thickness}, w={width}, matériau={material}")
        print(f"Mode Dx: {'Fixe' if self.Dx_fixe else 'Variable'}")

        E = self.get_material_E(material)
        Dx = self._calculate_Dx(length)
        lever_arm = self._calculate_lever_arm(length)
        performance = self._calculate_performance(width, thickness, length, material)
        perf_ratio = performance / self.performance_ref

        kth = (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3
        kth_mNm = kth * 1e3

        print(f"E = {E:.0f} N/mm²")
        print(f"Dx = {Dx:.4f} mm")
        print(f"Bras de levier = {lever_arm:.4f} mm")
        print(f"kth = {kth_mNm:.2f} mN·m")
        print(f"Performance = {performance:.2f}")
        print(f"perf_ref = {self.performance_ref:.2f}")
        print(f"perf_ratio = {perf_ratio:.6f}")
        print(f"Dans la plage acceptable? {self.is_performance_ratio_valid(performance)}")

    def generate_all_configs_perf_filtered(self, config_type: ConfigurationType, fixed_value: float) -> List[Dict]:
        """Génère toutes les configurations qui respectent le critère de perf_ratio (sans filtrage par percentiles)"""
        valid_configurations = []

        if config_type == ConfigurationType.L_CONSTANT:
            # L fixe, varier h et w
            for h in self.h_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, h, fixed_value)
                    if self.is_performance_ratio_valid(perf):  # Filtrage par perf_ratio uniquement
                        config = self._create_config_dict(w, h, fixed_value, perf, config_type)
                        valid_configurations.append(config)

        elif config_type == ConfigurationType.H_CONSTANT:
            # h fixe, varier L et w
            for L in self.L_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, fixed_value, L)
                    if self.is_performance_ratio_valid(perf):  # Filtrage par perf_ratio uniquement
                        config = self._create_config_dict(w, fixed_value, L, perf, config_type)
                        valid_configurations.append(config)

        elif config_type == ConfigurationType.W_CONSTANT:
            # w fixe, varier L et h
            for L in self.L_range:
                for h in self.h_range:
                    perf = self._calculate_performance(fixed_value, h, L)
                    if self.is_performance_ratio_valid(perf):  # Filtrage par perf_ratio uniquement
                        config = self._create_config_dict(fixed_value, h, L, perf, config_type)
                        valid_configurations.append(config)

        return valid_configurations

    def _get_global_minmax_configs(self, all_configs: List[Dict], config_type: ConfigurationType) -> Dict:
        """Extrait les configurations min/max globales selon le type (sans filtrage perf_ratio)"""
        if not all_configs:
            return {}

        minmax_configs = {}

        if config_type == ConfigurationType.L_CONSTANT:
            # Pour L constant, on cherche h min et h max globaux
            h_values = [config['thickness'] for config in all_configs]
            h_min = min(h_values)
            h_max = max(h_values)

            # Trouver les configurations correspondantes (prendre la première occurrence)
            h_min_config = next(config for config in all_configs if config['thickness'] == h_min)
            h_max_config = next(config for config in all_configs if config['thickness'] == h_max)

            minmax_configs = {
                'h_min': h_min_config,
                'h_max': h_max_config
            }

        elif config_type == ConfigurationType.H_CONSTANT:
            # Pour h constant, on cherche L min et L max globaux
            L_values = [config['length'] for config in all_configs]
            L_min = min(L_values)
            L_max = max(L_values)

            # Trouver les configurations correspondantes
            L_min_config = next(config for config in all_configs if config['length'] == L_min)
            L_max_config = next(config for config in all_configs if config['length'] == L_max)

            minmax_configs = {
                'L_min': L_min_config,
                'L_max': L_max_config
            }

        elif config_type == ConfigurationType.W_CONSTANT:
            # Pour w constant, on cherche L min et L max globaux
            L_values = [config['length'] for config in all_configs]
            L_min = min(L_values)
            L_max = max(L_values)

            # Trouver les configurations correspondantes
            L_min_config = next(config for config in all_configs if config['length'] == L_min)
            L_max_config = next(config for config in all_configs if config['length'] == L_max)

            minmax_configs = {
                'L_min': L_min_config,
                'L_max': L_max_config
            }

        return minmax_configs

    def print_global_extremes_fixed_dx(self, config_type: ConfigurationType, fixed_value: float, title_base: str):
        """Affiche les valeurs extrêmes globales pour Dx fixe"""
        # Générer toutes les configurations sans filtrage
        all_configs = self.generate_all_configs_perf_filtered(config_type, fixed_value)

        if not all_configs:
            print(f"\n=== {title_base} (Valeurs Min / Max des valeurs non filtrées) ===")
            print("Aucune configuration trouvée.")
            return

        # Obtenir les configurations min/max
        minmax_configs = self._get_global_minmax_configs(all_configs, config_type)

        if not minmax_configs:
            return

        print(f"\n=== {title_base} (Valeurs Min / Max des valeurs non filtrées) ===")
        print("Info\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\tBras(mm)\tperf_ratio\tErreur")
        print("-" * 85)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            print(f"{label}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['lever_arm']:.2f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    def print_global_extremes_variable_dx(self, config_type: ConfigurationType, fixed_value: float, title_base: str):
        """Affiche les valeurs extrêmes globales pour Dx variable"""
        # Générer toutes les configurations sans filtrage
        all_configs = self.generate_all_configs_perf_filtered(config_type, fixed_value)

        if not all_configs:
            print(f"\n=== {title_base} (Valeurs Min / Max des valeurs non filtrées) ===")
            print("Aucune configuration trouvée.")
            return

        # Obtenir les configurations min/max
        minmax_configs = self._get_global_minmax_configs(all_configs, config_type)

        if not minmax_configs:
            return

        print(f"\n=== {title_base} (Valeurs Min / Max des valeurs non filtrées) ===")
        print("Info\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\tBras(mm)\tkth(mN·m)\tLever²\tperf_ratio\tErreur")
        print("-" * 110)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            lever_squared = config['lever_arm'] ** 2
            print(f"{label}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['lever_arm']:.2f}\t"
                  f"{config['kth_mNm']:.1f}\t\t{lever_squared:.1f}\t{config['perf_ratio']:.4f}\t"
                  f"{config['perf_error_from_optimal']:.6f}")

    def compare_modes_detailed_with_global_extremes(self):
        """Compare les deux modes avec tableaux détaillés + valeurs extrêmes globales"""
        print(f"\n{'=' * 100}")
        print("COMPARAISON DÉTAILLÉE DES MODES Dx FIXE vs VARIABLE (avec extrêmes globaux)")
        print(f"{'=' * 100}")

        # Génération des configurations pour Dx fixe
        print("\n" + "=" * 50 + " Dx FIXE " + "=" * 50)
        self.Dx_fixe = True
        configs_fixed = self.find_all_configurations()

        # Tableaux filtrés existants
        self.print_configurations_fixed_dx(
            configs_fixed['L_constant_optimal'],
            f"L CONSTANT = {self.length_ref:.2f}mm - {self.material} (Dx FIXE)"
        )

        # NOUVEAU: Tableaux des extrêmes globaux
        self.print_global_extremes_fixed_dx(
            ConfigurationType.L_CONSTANT,
            self.length_ref,
            f"L CONSTANT = {self.length_ref:.2f}mm - {self.material}"
        )

        self.print_configurations_fixed_dx(
            configs_fixed['h_constant_optimal'],
            f"h CONSTANT = {self.thickness_ref:.3f}mm - {self.material} (Dx FIXE)"
        )

        # NOUVEAU: Tableaux des extrêmes globaux
        self.print_global_extremes_fixed_dx(
            ConfigurationType.H_CONSTANT,
            self.thickness_ref,
            f"h CONSTANT = {self.thickness_ref:.3f}mm - {self.material}"
        )

        self.print_configurations_fixed_dx(
            configs_fixed['w_constant_optimal'],
            f"w CONSTANT = {self.width_ref:.1f}mm - {self.material} (Dx FIXE)"
        )

        # NOUVEAU: Tableaux des extrêmes globaux
        self.print_global_extremes_fixed_dx(
            ConfigurationType.W_CONSTANT,
            self.width_ref,
            f"w CONSTANT = {self.width_ref:.1f}mm - {self.material}"
        )

        # Génération des configurations pour Dx variable
        print("\n" + "=" * 50 + " Dx VARIABLE " + "=" * 50)
        self.Dx_fixe = False
        # Recalcul de la référence pour le mode variable
        self.performance_ref = self._calculate_performance(self.width_ref, self.thickness_ref, self.length_ref)
        configs_variable = self.find_all_configurations()

        # Tableaux filtrés existants
        self.print_configurations_variable_dx(
            configs_variable['L_constant_optimal'],
            f"L CONSTANT = {self.length_ref:.2f}mm - {self.material} (Dx VARIABLE)"
        )

        # NOUVEAU: Tableaux des extrêmes globaux
        self.print_global_extremes_variable_dx(
            ConfigurationType.L_CONSTANT,
            self.length_ref,
            f"L CONSTANT = {self.length_ref:.2f}mm - {self.material}"
        )

        self.print_configurations_variable_dx(
            configs_variable['h_constant_optimal'],
            f"h CONSTANT = {self.thickness_ref:.3f}mm - {self.material} (Dx VARIABLE)"
        )

        # NOUVEAU: Tableaux des extrêmes globaux
        self.print_global_extremes_variable_dx(
            ConfigurationType.H_CONSTANT,
            self.thickness_ref,
            f"h CONSTANT = {self.thickness_ref:.3f}mm - {self.material}"
        )

        self.print_configurations_variable_dx(
            configs_variable['w_constant_optimal'],
            f"w CONSTANT = {self.width_ref:.1f}mm - {self.material} (Dx VARIABLE)"
        )

        # NOUVEAU: Tableaux des extrêmes globaux
        self.print_global_extremes_variable_dx(
            ConfigurationType.W_CONSTANT,
            self.width_ref,
            f"w CONSTANT = {self.width_ref:.1f}mm - {self.material}"
        )


def main():
    """Fonction principale avec comparaison des modes"""

    # Analyse avec BE_CU
    blade_config = ModularBladeConfig(material='BE_CU', Dx_fixe=False)

    # Vérification avec la configuration de référence
    print("VÉRIFICATION MODE VARIABLE:")
    blade_config.verify_calculation(105.25, 0.24, 45.0, 'BE_CU')

    blade_config.Dx_fixe = True
    blade_config.performance_ref = blade_config._calculate_performance(
        blade_config.width_ref, blade_config.thickness_ref, blade_config.length_ref
    )
    print("\nVÉRIFICATION MODE FIXE:")
    blade_config.verify_calculation(105.25, 0.24, 45.0, 'BE_CU')

    # Comparaison détaillée des modes
    blade_config.compare_modes_detailed_with_global_extremes()

    return blade_config


if __name__ == "__main__":
    config = main()