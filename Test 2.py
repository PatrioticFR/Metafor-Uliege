# Nouvelle version du script avec calibration plus réaliste basée sur les données utilisateur

import math
import numpy as np
from enum import Enum
from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuration pour les 2 lames
DUAL_BLADE_CONFIG = True

class ConfigurationTypeDual(Enum):
    L_BECU_CONSTANT = "L_BeCu_constant"

class VariationMode(Enum):
    OPTIMAL = "optimal"
    HIGH_LOW = "high_low"
    LOW_HIGH = "low_high"

def calculate_ideal_perf_ratio_range_dual():
    # Basé sur les données expérimentales fournies par l'utilisateur (stabilisation entre -0.5 mm et 1.0 mm)
    # Données:
    # Contraintes finales (MPa) et déplacements (mm)
    sigma = np.array([457.745, 478.197, 505.432, 569.417])
    disp = np.array([-0.651699, -2.37118, -4.62793, -9.95165])

    # Ajustement linéaire disp = a * sigma + b
    coeffs = np.polyfit(sigma, disp, 1)
    a, b = coeffs

    # Objectif: trouver les contraintes qui donnent -0.5 mm et 1 mm
    target_min = -0.5
    target_max = 1.0
    sigma_min = (target_min - b) / a
    sigma_max = (target_max - b) / a

    # On convertit en perf_ratio approximatif (valeurs empiriques)
    perf_min = sigma_min / 457.745  # échelle relative
    perf_max = sigma_max / 457.745

    print("=== Calibration à partir des données utilisateur ===")
    print(f"Fit: disp = {a:.4f} * sigma + {b:.4f}")
    print(f"Objectif de stabilisation entre {target_min} et {target_max} mm")
    print(f"=> Contraintes cibles entre {sigma_min:.2f} MPa et {sigma_max:.2f} MPa")
    print(f"=> Approximation perf_ratio entre {perf_min:.6f} et {perf_max:.6f}\n")

    return min(perf_min, perf_max), max(perf_min, perf_max)

class DualBladeConfig:
    def __init__(self):
        self.thickness_ref_becu = 0.24
        self.length_ref_becu = 105.25
        self.width_ref_becu = 30.0

        self.thickness_ref_invar = 0.24
        self.length_ref_invar = 103.82
        self.width_ref_invar = 20.0

        self.enc_ref = 57.32
        self.decalage = 0.25

        self.material_properties = {
            'BE_CU': 131e3,
            'INVAR': 141e3,
        }

        self.H = 3.875
        self.D = 39.99
        self.d = 13.96
        self.l = 79.2
        self.r = 7.0
        self.R_rod = self.H
        self.rho = 7.85e-6
        self.depth = 63.0

        self.k_flex = 44.3
        self.r_s = 65.22

        self.w_becu_range = np.arange(20, 40.1, 1.0)
        self.h_becu_range = [0.24]
        self.w_invar_range = np.arange(20, 40.1, 1.0)
        self.h_invar_range = [0.24, 0.22]
        self.L_invar_range = np.arange(100.0, 105.0, 0.5)

        self.perf_ratio_min, self.perf_ratio_max = calculate_ideal_perf_ratio_range_dual()

        self.I_system = self._calculate_system_inertia()
        self.performance_ref = self._calculate_dual_performance(
            self.width_ref_becu, self.thickness_ref_becu, self.length_ref_becu,
            self.width_ref_invar, self.thickness_ref_invar, self.length_ref_invar
        )

    def _calculate_system_inertia(self):
        h_rod_segment = self.l - self.r - self.d
        I_barre_rod_g = self.rho * self.depth * h_rod_segment * (self.H ** 3 / 3 + self.H * (h_rod_segment / 2) ** 2)
        I_barre_mass = self.rho * self.depth * self.d * (self.D ** 3 / 3 + self.D * (h_rod_segment + self.d / 2) ** 2)
        I_barre_rod_d = self.rho * self.depth * self.r * (self.R_rod ** 3 / 3 + self.R_rod * (h_rod_segment + self.d + self.r / 2) ** 2)
        return I_barre_rod_g + I_barre_mass + I_barre_rod_d

    def _calculate_Dx(self, length):
        return -2 * (length / math.pi)

    def _calculate_k_single_blade(self, E, width, thickness, length):
        return (math.pi ** 2 / 6) * E * width * thickness ** 3 / length ** 3

    def _calculate_dual_performance(self, w_becu, h_becu, L_becu, w_invar, h_invar, L_invar):
        E_becu = self.material_properties['BE_CU']
        E_invar = self.material_properties['INVAR']
        k_becu = self._calculate_k_single_blade(E_becu, w_becu, h_becu, L_becu)
        k_invar = self._calculate_k_single_blade(E_invar, w_invar, h_invar, L_invar)
        k_total = k_becu + k_invar
        rs = abs(self._calculate_Dx((L_becu + L_invar)/2))
        return k_total * 1e3 * rs**2

    def is_performance_ratio_valid(self, performance):
        perf_ratio = performance / self.performance_ref
        return self.perf_ratio_min <= perf_ratio <= self.perf_ratio_max

    def verify_dual_calculation(self, w_becu, h_becu, L_becu, w_invar, h_invar, L_invar):
        print("\n=== VÉRIFICATION CONFIGURATION ===")
        perf = self._calculate_dual_performance(w_becu, h_becu, L_becu, w_invar, h_invar, L_invar)
        ratio = perf / self.performance_ref
        print(f"Performance (k\u00d7rs^2): {perf:.2f} mN·mm")
        print(f"Ratio vs référence: {ratio:.4f} ({'OK' if self.is_performance_ratio_valid(perf) else 'HORS PLAGE'})")

if __name__ == "__main__":
    config = DualBladeConfig()
    config.verify_dual_calculation(
        30.0, 0.24, 105.25,  # BeCu
        20.0, 0.24, 103.82   # Invar
    )
