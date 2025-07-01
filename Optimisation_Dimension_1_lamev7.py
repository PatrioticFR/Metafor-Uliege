import math
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple

MATERIAL_CONFIG = 'BE_CU'  # Change this to 'INVAR' or 'STEEL' or 'BE_CU' as needed


class ConfigurationType(Enum):
    """Types of possible configurations"""
    L_CONSTANT = "L_constant"
    H_CONSTANT = "h_constant"
    W_CONSTANT = "w_constant"


class VariationMode(Enum):
    """Variation modes for parameters"""
    OPTIMAL = "optimal"
    HIGH_LOW = "high_low"
    LOW_HIGH = "low_high"


def calculate_ideal_perf_ratio_range(material=MATERIAL_CONFIG):
    """
    Calculate ideal values for perf_ratio_min and perf_ratio_max
    based on observed stabilization data for specific material
    """

    # Material-specific observed data
    material_configs = {
        'BE_CU': {
            'config1': {
                'L': 93.1, 'h': 0.24, 'w': 40.8,
                'perf_ratio': 1.025, 'stabilization': 0.571853
            },
            'config2': {
                'L': 93.1, 'h': 0.24, 'w': 40.4,
                'perf_ratio': 1.015, 'stabilization': 0.256159
            }
        },
        'STEEL': {
            # You need to provide experimental data for Steel here
            'config1': {
                'L': 93.1, 'h': 0.24, 'w': 40.8,
                'perf_ratio': 1.020, 'stabilization': 0.500  # Example values - replace with real data if needed
            },
            'config2': {
                'L': 93.1, 'h': 0.24, 'w': 40.4,
                'perf_ratio': 1.010, 'stabilization': 0.300  # Example values - replace with real data if needed
            }
        },
        'INVAR': {
            'config1': {
                'L': 105.25, 'h': 0.24, 'w': 45.0,
                'perf_ratio': 1.0076, 'stabilization': -2.97027
            },
            'config2': {
                'L': 105.25, 'h': 0.23, 'w': 52.4,
                'perf_ratio': 1.0249, 'stabilization': -0.00145
            }
        }
    }

    if material not in material_configs:
        print(f"Warning: No experimental data for material {material}, using {MATERIAL_CONFIG} data")
        material = MATERIAL_CONFIG

    config1 = material_configs[material]['config1']
    config2 = material_configs[material]['config2']

    # Rest of the function remains the same...
    target_min = 0.5  # mm
    target_max = 1.0  # mm

    print(f"=== CALCULATION OF IDEAL PERF_RATIO FOR {material} ===")
    print(f"Configuration 1: L={config1['L']}, h={config1['h']}, w={config1['w']}")
    print(f"  perf_ratio = {config1['perf_ratio']:.4f} -> stabilization = {config1['stabilization']:.3f} mm")
    print(f"Configuration 2: L={config2['L']}, h={config2['h']}, w={config2['w']}")
    print(f"  perf_ratio = {config2['perf_ratio']:.4f} -> stabilization = {config2['stabilization']:.3f} mm")
    print()
    print(f"Target: stabilization between {target_min} and {target_max} mm")
    print()

    # Calculation by linear interpolation/extrapolation
    # Relation: stabilization = a * perf_ratio + b

    # Calculate line coefficients
    delta_stab = config1['stabilization'] - config2['stabilization']
    delta_perf = config1['perf_ratio'] - config2['perf_ratio']

    # Line slope
    a = delta_stab / delta_perf

    # Y-intercept
    b = config1['stabilization'] - a * config1['perf_ratio']

    print(f"Linear relation: stabilization = {a:.2f} * perf_ratio + {b:.2f}")
    print()

    # Calculate perf_ratio corresponding to targets
    # target = a * perf_ratio + b
    # perf_ratio = (target - b) / a

    perf_ratio_for_target_min = (target_min - b) / a
    perf_ratio_for_target_max = (target_max - b) / a

    # Determine min and max (as relation can be decreasing)
    perf_ratio_min = min(perf_ratio_for_target_min, perf_ratio_for_target_max)
    perf_ratio_max = max(perf_ratio_for_target_min, perf_ratio_for_target_max)

    print("=== RESULTS ===")
    print(f"For stabilization of {target_min} mm: perf_ratio = {perf_ratio_for_target_min:.6f}")
    print(f"For stabilization of {target_max} mm: perf_ratio = {perf_ratio_for_target_max:.6f}")
    print()
    print(f"IDEAL VALUES TO USE:")
    print(f"self.perf_ratio_min = {perf_ratio_min:.6f}")
    print(f"self.perf_ratio_max = {perf_ratio_max:.6f}")
    print()

    # Verification
    print("=== VERIFICATION ===")
    stab_check_min = a * perf_ratio_min + b
    stab_check_max = a * perf_ratio_max + b
    print(f"With perf_ratio = {perf_ratio_min:.6f} -> stabilization = {stab_check_min:.3f} mm")
    print(f"With perf_ratio = {perf_ratio_max:.6f} -> stabilization = {stab_check_max:.3f} mm")
    print()

    # Trend analysis
    if a > 0:
        print("Trend: Higher perf_ratio increases stabilization")
        print(f"   perf_ratio_min ({perf_ratio_min:.6f}) -> stabilization towards {target_min} mm")
        print(f"   perf_ratio_max ({perf_ratio_max:.6f}) -> stabilization towards {target_max} mm")
    else:
        print("Trend: Higher perf_ratio decreases stabilization")
        print(f"   perf_ratio_min ({perf_ratio_min:.6f}) -> stabilization towards {target_max} mm")
        print(f"   perf_ratio_max ({perf_ratio_max:.6f}) -> stabilization towards {target_min} mm")

    return perf_ratio_min, perf_ratio_max


class ModularBladeConfig:
    """Modular configuration for blade optimization with lever arm effect"""

    perf_min, perf_max = calculate_ideal_perf_ratio_range()

    def __init__(self, material: str = MATERIAL_CONFIG, Dx_fixed: bool = False):
        # Reference geometry
        self.thickness_ref = 0.24  # h reference
        self.length_ref = 105.25  # L reference
        self.width_ref = 45.0  # w reference
        self.enc_ref = 57.32  # Encoder position on mass

        # Dx calculation mode
        self.Dx_fixed = Dx_fixed
        self.Dx_fixed_value = -67.5227  # Fixed value if Dx_fixed = True

        # Material
        self.material = material

        # Material properties (in N/mm²)
        self.material_properties = {
            'BE_CU': 131e3,
            'INVAR': 141e3,
            'STEEL': 210e3
        }

        # Geometric parameters of rod/mass system
        self.H = 3.875  # rod vertical thickness (mm)
        self.D = 39.99  # mass block height (mm)
        self.d = 13.96  # mass block length (mm)
        self.l = 79.2  # total rod length (mm)
        self.r = 7.0  # rod right segment (mm)
        self.R_rod = self.H  # rod right width
        self.rho = 7.85e-6  # steel density (kg/mm³)
        self.depth = 63.0  # structure width/depth (mm)

        # Flexure system parameters
        self.k_flex = 44.3  # mN·m
        self.r_s = 65.22  # mm

        # Search ranges
        self.w_range = np.arange(10, 60.1, 0.1)
        self.h_range = np.arange(0.05, 0.41, 0.01)
        self.L_range = np.arange(93.0, 110.1, 0.05)

        # Calculate material-specific performance ratios
        self.perf_min, self.perf_max = calculate_ideal_perf_ratio_range(material)

        # System inertia calculation (constant)
        self.I_system = self._calculate_system_inertia()

        # Reference performance calculation (based on reference geometry)
        self.performance_ref = self._calculate_performance(self.width_ref, self.thickness_ref, self.length_ref)

        # Use calculated values for this material
        self.perf_ratio_min = self.perf_min
        self.perf_ratio_max = self.perf_max

        # self.perf_ratio_min = 1.02
        # self.perf_ratio_max = 1.03

        print(f"Configuration: Material = {self.material}, Dx_fixed = {self.Dx_fixed}")
        print(f"Perf_ratio range: [{self.perf_ratio_min:.6f}, {self.perf_ratio_max:.6f}]")
        print(f"Configuration: Dx_fixed = {self.Dx_fixed}")
        print(f"Reference performance: {self.performance_ref:.6f}")
        print(f"System inertia: {self.I_system:.3f} kg·mm²")

    def _calculate_system_inertia(self) -> float:
        """Calculate total inertia of rod/mass system"""
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
        """Young's modulus of material (in N/mm²)"""
        if material is None:
            material = self.material
        return self.material_properties[material]

    def _calculate_Dx(self, length: float) -> float:
        """Calculate Dx according to chosen mode"""
        if self.Dx_fixed:
            return self.Dx_fixed_value
        else:
            R = length / math.pi
            return -2 * R

    def _calculate_performance(self, width: float, thickness: float, length: float, material: str = None) -> float:
        """Calculate k*rs² which must remain constant according to equilibrium equation"""
        E = self.get_material_E(material)

        # Theoretical blade stiffness [N·mm]
        k = (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3
        k_mNm = k * 1e3  # Conversion to mN·m

        # rs is the dx distance (lever arm)
        Dx = self._calculate_Dx(length)
        rs = abs(Dx)  # rs = |Dx|

        # Performance = k * rs²
        # This value must remain constant to maintain equilibrium
        performance = k_mNm * (rs ** 2)

        return performance

    def _calculate_lever_arm(self, length: float) -> float:
        """Calculate rs (dx distance) - the lever arm in equilibrium equation"""
        Dx = self._calculate_Dx(length)
        return abs(Dx)  # rs = |Dx|

    def is_performance_ratio_valid(self, performance: float) -> bool:
        """Check if performance ratio is within ideal range"""
        perf_ratio = performance / self.performance_ref
        return self.perf_ratio_min <= perf_ratio <= self.perf_ratio_max

    def generate_configs(self, config_type: ConfigurationType, fixed_value: float, mode: VariationMode) -> List[Dict]:
        """Generate configurations according to type"""
        configurations = []

        if config_type == ConfigurationType.L_CONSTANT:
            # L fixed, vary h and w
            for h in self.h_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, h, fixed_value)
                    if self.is_performance_ratio_valid(perf):
                        config = self._create_config_dict(w, h, fixed_value, perf, config_type)
                        configurations.append(config)
            return self._filter_by_mode(configurations, mode, 'thickness', 'width')

        elif config_type == ConfigurationType.H_CONSTANT:
            # h fixed, vary L and w
            for L in self.L_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, fixed_value, L)
                    if self.is_performance_ratio_valid(perf):
                        config = self._create_config_dict(w, fixed_value, L, perf, config_type)
                        configurations.append(config)
            return self._filter_by_mode(configurations, mode, 'length', 'width')

        elif config_type == ConfigurationType.W_CONSTANT:
            # w fixed, vary L and h
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
        """Create configuration dictionary"""
        perf_ratio = performance / self.performance_ref
        perf_optimal = (self.perf_ratio_min + self.perf_ratio_max) / 2
        Dx = self._calculate_Dx(length)
        rs = abs(Dx)  # rs = |Dx|

        # Stiffness calculation for information
        E = self.get_material_E()
        k = (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3
        k_mNm = k * 1e3

        return {
            'width': width,
            'thickness': thickness,
            'length': length,
            'performance': performance,  # k * rs²
            'perf_ratio': perf_ratio,
            'Dx': Dx,
            'rs': rs,  # Lever arm (dx distance)
            'k_mNm': k_mNm,  # Stiffness alone
            'k_times_rs_squared': performance,  # k * rs² (same as performance)
            'config_type': config_type,
            'perf_error_from_optimal': abs(perf_ratio - perf_optimal),
            'material': self.material
        }

    def _filter_by_mode(self, configurations: List[Dict], mode: VariationMode,
                        param1: str, param2: str) -> List[Dict]:
        """Filter configurations according to variation mode"""
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
        """Find all possible configurations"""
        all_configs = {}

        # L constant - UTILISEZ LA VALEUR DÉSIRÉE ICI
        all_configs['L_constant_optimal'] = self.generate_configs(
            ConfigurationType.L_CONSTANT, self.length_ref, VariationMode.OPTIMAL)  # Au lieu de self.length_ref

        # h constant
        all_configs['h_constant_optimal'] = self.generate_configs(
            ConfigurationType.H_CONSTANT, self.thickness_ref, VariationMode.OPTIMAL) # Au lieu de self.thickness_ref

        # w constant
        all_configs['w_constant_optimal'] = self.generate_configs(
            ConfigurationType.W_CONSTANT, 55.0, VariationMode.OPTIMAL) # Au lieu de self.width_ref

        return all_configs

    def _get_minmax_configs(self, configs: List[Dict], config_type: ConfigurationType) -> Dict:
        """Extract min/max configurations according to type"""
        if not configs:
            return {}

        minmax_configs = {}

        if config_type == ConfigurationType.L_CONSTANT:
            # For L constant, look for h min and h max
            h_values = [config['thickness'] for config in configs]
            h_min = min(h_values)
            h_max = max(h_values)

            # Find corresponding configurations
            h_min_config = next(config for config in configs if config['thickness'] == h_min)
            h_max_config = next(config for config in configs if config['thickness'] == h_max)

            minmax_configs = {
                'h_min': h_min_config,
                'h_max': h_max_config
            }

        elif config_type == ConfigurationType.H_CONSTANT:
            # For h constant, look for L min and L max
            L_values = [config['length'] for config in configs]
            L_min = min(L_values)
            L_max = max(L_values)

            # Find corresponding configurations
            L_min_config = next(config for config in configs if config['length'] == L_min)
            L_max_config = next(config for config in configs if config['length'] == L_max)

            minmax_configs = {
                'L_min': L_min_config,
                'L_max': L_max_config
            }

        elif config_type == ConfigurationType.W_CONSTANT:
            # For w constant, look for L min and L max
            L_values = [config['length'] for config in configs]
            L_min = min(L_values)
            L_max = max(L_values)

            # Find corresponding configurations
            L_min_config = next(config for config in configs if config['length'] == L_min)
            L_max_config = next(config for config in configs if config['length'] == L_max)

            minmax_configs = {
                'L_min': L_min_config,
                'L_max': L_max_config
            }

        return minmax_configs

    def print_configurations_fixed_dx(self, configs: List[Dict], title: str):
        """Display configurations for fixed Dx"""
        if not configs:
            print(f"\n=== {title} ===")
            print("No configuration found.")
            return

        print(f"\n=== {title} ({len(configs)} configurations) ===")
        print("Rank\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\trs(mm)\tperf_ratio\tError")
        print("-" * 85)

        for i, config in enumerate(configs[:20], 1):
            print(f"{i:2d}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['rs']:.2f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

        # Display min/max values
        self._print_minmax_fixed_dx(configs, title)

    def print_configurations_variable_dx(self, configs: List[Dict], title: str):
        """Display configurations for variable Dx with focus on k*rs²"""
        if not configs:
            print(f"\n=== {title} ===")
            print("No configuration found.")
            return

        print(f"\n=== {title} ({len(configs)} configurations) ===")
        print("Rank\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\trs(mm)\tk(mN·m)\trs²(mm²)\tk×rs²\tRatio\tError")
        print("-" * 120)

        for i, config in enumerate(configs[:20], 1):
            rs_squared = config['rs'] ** 2
            print(f"{i:2d}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['rs']:.2f}\t"
                  f"{config['k_mNm']:.1f}\t\t{rs_squared:.1f}\t\t{config['k_times_rs_squared']:.1f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

        # Display min/max values
        self._print_minmax_variable_dx(configs, title)

    def _print_minmax_fixed_dx(self, configs: List[Dict], title: str):
        """Display min/max values for fixed Dx"""
        if not configs:
            return

        # Determine configuration type
        config_type = configs[0]['config_type']
        minmax_configs = self._get_minmax_configs(configs, config_type)

        if not minmax_configs:
            return

        # Extract base title name
        base_title = title.split('(')[0].strip()
        print(f"\n=== {base_title} (Min / Max Values) ===")
        print("Info\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\trs(mm)\tperf_ratio\tError")
        print("-" * 85)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            print(f"{label}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['rs']:.2f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    def _print_minmax_variable_dx(self, configs: List[Dict], title: str):
        """Display min/max values for variable Dx"""
        if not configs:
            return

        # Determine configuration type
        config_type = configs[0]['config_type']
        minmax_configs = self._get_minmax_configs(configs, config_type)

        if not minmax_configs:
            return

        # Extract base title name
        base_title = title.split('(')[0].strip()
        print(f"\n=== {base_title} (Min / Max Values) ===")
        print("Info\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\trs(mm)\tk(mN·m)\trs²(mm²)\tk×rs²\tRatio\tError")
        print("-" * 120)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            rs_squared = config['rs'] ** 2
            print(f"{label}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['rs']:.2f}\t"
                  f"{config['k_mNm']:.1f}\t\t{rs_squared:.1f}\t\t{config['k_times_rs_squared']:.1f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    def verify_calculation(self, length: float, thickness: float, width: float, material: str = None):
        """Verify a specific calculation"""
        if material is None:
            material = self.material

        print(f"\n=== CALCULATION VERIFICATION ===")
        print(f"Parameters: L={length}, h={thickness}, w={width}, material={material}")
        print(f"Dx mode: {'Fixed' if self.Dx_fixed else 'Variable'}")

        E = self.get_material_E(material)
        Dx = self._calculate_Dx(length)
        rs = abs(Dx)
        performance = self._calculate_performance(width, thickness, length, material)
        perf_ratio = performance / self.performance_ref

        k = (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3
        k_mNm = k * 1e3

        print(f"E = {E:.0f} N/mm²")
        print(f"Dx = {Dx:.4f} mm")
        print(f"rs = |Dx| = {rs:.4f} mm")
        print(f"k = {k_mNm:.2f} mN·m")
        print(f"rs² = {rs ** 2:.2f} mm²")
        print(f"k × rs² = {performance:.2f} mN·m·mm²")
        print(f"Reference performance (k × rs²)_ref = {self.performance_ref:.2f} mN·m·mm²")
        print(f"Ratio (k × rs²)/(k × rs²)_ref = {perf_ratio:.6f}")
        print(f"Within acceptable range? {self.is_performance_ratio_valid(performance)}")
        print()
        print("PHYSICAL INTERPRETATION:")
        print(f"- To maintain equilibrium, k × rs² must stay close to {self.performance_ref:.2f}")
        print(f"- If rs decreases, k must increase proportionally to 1/rs²")
        print(f"- If rs increases, k can decrease proportionally to 1/rs²")
        print(f"Within acceptable range? {self.is_performance_ratio_valid(performance)}")

    def generate_all_configs_perf_filtered(self, config_type: ConfigurationType, fixed_value: float) -> List[Dict]:
        """Generate all configurations that respect perf_ratio criterion (without percentile filtering)"""
        valid_configurations = []

        if config_type == ConfigurationType.L_CONSTANT:
            # L fixed, vary h and w
            for h in self.h_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, h, fixed_value)
                    if self.is_performance_ratio_valid(perf):  # Filter by perf_ratio only
                        config = self._create_config_dict(w, h, fixed_value, perf, config_type)
                        valid_configurations.append(config)

        elif config_type == ConfigurationType.H_CONSTANT:
            # h fixed, vary L and w
            for L in self.L_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, fixed_value, L)
                    if self.is_performance_ratio_valid(perf):  # Filter by perf_ratio only
                        config = self._create_config_dict(w, fixed_value, L, perf, config_type)
                        valid_configurations.append(config)

        elif config_type == ConfigurationType.W_CONSTANT:
            # w fixed, vary L and h
            for L in self.L_range:
                for h in self.h_range:
                    perf = self._calculate_performance(fixed_value, h, L)
                    if self.is_performance_ratio_valid(perf):  # Filter by perf_ratio only
                        config = self._create_config_dict(fixed_value, h, L, perf, config_type)
                        valid_configurations.append(config)

        return valid_configurations

    def _get_global_minmax_configs(self, all_configs: List[Dict], config_type: ConfigurationType) -> Dict:
        """Extract global min/max configurations according to type (without perf_ratio filtering)"""
        if not all_configs:
            return {}

        minmax_configs = {}

        if config_type == ConfigurationType.L_CONSTANT:
            # For L constant, look for global h min and h max
            h_values = [config['thickness'] for config in all_configs]
            h_min = min(h_values)
            h_max = max(h_values)

            # Find corresponding configurations (take first occurrence)
            h_min_config = next(config for config in all_configs if config['thickness'] == h_min)
            h_max_config = next(config for config in all_configs if config['thickness'] == h_max)

            minmax_configs = {
                'h_min': h_min_config,
                'h_max': h_max_config
            }

        elif config_type == ConfigurationType.H_CONSTANT:
            # For h constant, look for global L min and L max
            L_values = [config['length'] for config in all_configs]
            L_min = min(L_values)
            L_max = max(L_values)

            # Find corresponding configurations
            L_min_config = next(config for config in all_configs if config['length'] == L_min)
            L_max_config = next(config for config in all_configs if config['length'] == L_max)

            minmax_configs = {
                'L_min': L_min_config,
                'L_max': L_max_config
            }

        elif config_type == ConfigurationType.W_CONSTANT:
            # For w constant, look for global L min and L max
            L_values = [config['length'] for config in all_configs]
            L_min = min(L_values)
            L_max = max(L_values)

            # Find corresponding configurations
            L_min_config = next(config for config in all_configs if config['length'] == L_min)
            L_max_config = next(config for config in all_configs if config['length'] == L_max)

            minmax_configs = {
                'L_min': L_min_config,
                'L_max': L_max_config
            }

        return minmax_configs

    def print_global_extremes_fixed_dx(self, config_type, fixed_value, title_base):
        """Print global extreme values for fixed Dx"""
        # Generate all configurations without filtering
        all_configs = self.generate_all_configs_perf_filtered(config_type, fixed_value)

        if not all_configs:
            print(f"\n=== {title_base} (Min / Max Values of unfiltered values) ===")
            print("No configuration found.")
            return

        # Get min/max configurations
        minmax_configs = self._get_global_minmax_configs(all_configs, config_type)

        if not minmax_configs:
            return

        print(f"\n=== {title_base} (Min / Max Values of unfiltered values) ===")
        print("Info\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\trs(mm)\tperf_ratio\tError")
        print("-" * 85)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            print(f"{label}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['rs']:.2f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    def print_global_extremes_variable_dx(self, config_type, fixed_value, title_base):
        """Print global extreme values for variable Dx"""
        # Generate all configurations without filtering
        all_configs = self.generate_all_configs_perf_filtered(config_type, fixed_value)

        if not all_configs:
            print(f"\n=== {title_base} (Min / Max Values of unfiltered values) ===")
            print("No configuration found.")
            return

        # Get min/max configurations
        minmax_configs = self._get_global_minmax_configs(all_configs, config_type)

        if not minmax_configs:
            return

        print(f"\n=== {title_base} (Min / Max Values of unfiltered values) ===")
        print("Info\tL(mm)\t\th(mm)\tw(mm)\tDx(mm)\trs(mm)\tk(mN.m)\trs^2(mm^2)\tk*rs^2\tRatio\tError")
        print("-" * 120)

        for key, config in minmax_configs.items():
            label = key.replace('_', ' ')
            rs_squared = config['rs'] ** 2
            print(f"{label}\t{config['length']:.2f}\t\t{config['thickness']:.3f}\t"
                  f"{config['width']:.1f}\t{config['Dx']:.2f}\t{config['rs']:.2f}\t"
                  f"{config['k_mNm']:.1f}\t\t{rs_squared:.1f}\t\t{config['k_times_rs_squared']:.1f}\t"
                  f"{config['perf_ratio']:.4f}\t{config['perf_error_from_optimal']:.6f}")

    def compare_modes_detailed_with_global_extremes(self):
        """Compare both modes with detailed tables + global extreme values"""
        print(f"\n{'=' * 100}")
        print("DETAILED COMPARISON OF FIXED Dx vs VARIABLE Dx MODES (with global extremes)")
        print(f"{'=' * 100}")

        # Generate configurations for fixed Dx
        print("\n" + "=" * 25 + " FIXED Dx " + "=" * 25)
        self.Dx_fixe = True
        configs_fixed = self.find_all_configurations()

        # Existing filtered tables
        self.print_configurations_fixed_dx(
            configs_fixed['L_constant_optimal'],
            f"L CONSTANT = {self.length_ref:.2f}mm - {self.material} (FIXED Dx)"
        )

        # NEW: Global extremes tables
        self.print_global_extremes_fixed_dx(
            ConfigurationType.L_CONSTANT,
            self.length_ref,
            f"L CONSTANT = {self.length_ref:.2f}mm - {self.material}"
        )

        self.print_configurations_fixed_dx(
            configs_fixed['h_constant_optimal'],
            f"h CONSTANT = {self.thickness_ref:.3f}mm - {self.material} (FIXED Dx)"
        )

        # NEW: Global extremes tables
        self.print_global_extremes_fixed_dx(
            ConfigurationType.H_CONSTANT,
            self.thickness_ref,
            f"h CONSTANT = {self.thickness_ref:.3f}mm - {self.material}"
        )

        self.print_configurations_fixed_dx(
            configs_fixed['w_constant_optimal'],
            f"w CONSTANT = {self.width_ref:.1f}mm - {self.material} (FIXED Dx)"
        )

        # NEW: Global extremes tables
        self.print_global_extremes_fixed_dx(
            ConfigurationType.W_CONSTANT,
            self.width_ref,
            f"w CONSTANT = {self.width_ref:.1f}mm - {self.material}"
        )

        # Generate configurations for variable Dx
        print("\n" + "=" * 25 + " VARIABLE Dx " + "=" * 25)
        self.Dx_fixe = False
        # Recalculate reference for variable mode
        self.performance_ref = self._calculate_performance(self.width_ref, self.thickness_ref, self.length_ref)
        configs_variable = self.find_all_configurations()

        # Existing filtered tables
        self.print_configurations_variable_dx(
            configs_variable['L_constant_optimal'],
            f"L CONSTANT = {self.length_ref:.2f}mm - {self.material} (VARIABLE Dx)"
        )

        # NEW: Global extremes tables
        self.print_global_extremes_variable_dx(
            ConfigurationType.L_CONSTANT,
            self.length_ref,
            f"L CONSTANT = {self.length_ref:.2f}mm - {self.material}"
        )

        self.print_configurations_variable_dx(
            configs_variable['h_constant_optimal'],
            f"h CONSTANT = {self.thickness_ref:.3f}mm - {self.material} (VARIABLE Dx)"
        )

        # NEW: Global extremes tables
        self.print_global_extremes_variable_dx(
            ConfigurationType.H_CONSTANT,
            self.thickness_ref,
            f"h CONSTANT = {self.thickness_ref:.3f}mm - {self.material}"
        )

        self.print_configurations_variable_dx(
            configs_variable['w_constant_optimal'],
            f"w CONSTANT = {self.width_ref:.1f}mm - {self.material} (VARIABLE Dx)"
        )

        # NEW: Global extremes tables
        self.print_global_extremes_variable_dx(
            ConfigurationType.W_CONSTANT,
            self.width_ref,
            f"w CONSTANT = {self.width_ref:.1f}mm - {self.material}"
        )

    def find_extreme_configurations(self) -> Dict[str, Dict]:
        """Trouve les configurations extrêmes pour chaque paramètre sans fixer les autres"""
        results = {
            'w_max': None,
            'w_min': None,
            'L_max': None,
            'L_min': None,
            'h_max': None,
            'h_min': None,
        }

        for L in self.L_range:
            for h in self.h_range:
                for w in self.w_range:
                    perf = self._calculate_performance(w, h, L)
                    if not self.is_performance_ratio_valid(perf):
                        continue
                    config = self._create_config_dict(w, h, L, perf, ConfigurationType.W_CONSTANT)

                    # Update extreme values
                    if results['w_max'] is None or w > results['w_max']['width']:
                        results['w_max'] = config
                    if results['w_min'] is None or w < results['w_min']['width']:
                        results['w_min'] = config
                    if results['L_max'] is None or L > results['L_max']['length']:
                        results['L_max'] = config
                    if results['L_min'] is None or L < results['L_min']['length']:
                        results['L_min'] = config
                    if results['h_max'] is None or h > results['h_max']['thickness']:
                        results['h_max'] = config
                    if results['h_min'] is None or h < results['h_min']['thickness']:
                        results['h_min'] = config

        return results


# Then modify the main() function:
def main():
    """Main function with mode comparison"""

    # Analysis with configurable material
    blade_config = ModularBladeConfig(material=MATERIAL_CONFIG, Dx_fixed=False)

    # Verification with reference configuration
    print("VERIFICATION VARIABLE MODE:")
    blade_config.verify_calculation(105.25, 0.24, 45.0, MATERIAL_CONFIG)

    blade_config.Dx_fixe = True
    blade_config.performance_ref = blade_config._calculate_performance(
        blade_config.width_ref, blade_config.thickness_ref, blade_config.length_ref
     )
    print("\nVERIFICATION FIXED MODE:")
    blade_config.verify_calculation(105.25, 0.24, 45.0, MATERIAL_CONFIG)

    # Detailed comparison of modes
    blade_config.compare_modes_detailed_with_global_extremes()

    return blade_config

if __name__ == "__main__":
    config = main()

    print("\n=== EXTREME CONFIGURATIONS (All parameters free) ===")
    extremes = config.find_extreme_configurations()

    for key, conf in extremes.items():
        print(f"\n--- {key.upper()} ---")
        print(f"L = {conf['length']:.2f} mm")
        print(f"h = {conf['thickness']:.3f} mm")
        print(f"w = {conf['width']:.2f} mm")
        print(f"Dx = {conf['Dx']:.2f} mm")
        print(f"perf_ratio = {conf['perf_ratio']:.6f}")
        print(f"k × rs² = {conf['performance']:.2f} mN·mm")
