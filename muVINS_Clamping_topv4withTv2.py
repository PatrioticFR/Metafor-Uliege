# Ultra-high performance inertial sensor for gravitational waves detection

# Study of the muVINS with configurable blade parameters , thermal settings, time step and others
# To change parameters, use the def __init__ in the BladeConfig and the SimulationConfig

# Adrien Pierrat based on the code of Morgane Zeoli


# -*- coding: Windows CP1252 -*-

import multiprocessing
print(f"Number of cores available : {multiprocessing.cpu_count()}")

from wrap import *
from wrap.mtFrequencyAnalysisw import *
import math

# enable full parallelism
StrVectorBase.useTBB(True)
StrMatrixBase.useTBB(True)
ContactInteraction.useTBB(True)


# =============================================================================
# Optimized configuration for a 0 to 1mm stabilization
# =============================================================================

import numpy as np
from math import sqrt, pi, cos, sin, atan2


class BladeConfig:
    """Configuration parameters for the blade with systematic compensation"""

    def __init__(self):
        # Reference blade geometry (from TFE baseline)
        self.thickness_ref = 0.24  # reference thickness (h)
        self.length_ref = 105.25  # reference length (L)
        self.width_ref = 45.0  # reference width

        # Current blade geometry
        self.thickness = 0.26  # current thickness (e/h)
        self.length = 105.25 # current length (L)
        self.width = 50.0 # current width

        # Material selection: 'BE_CU', 'INVAR', 'INVAR_CW_HARD'
        self.material = 'BE_CU'

        # Mesh parameters for blade - use your existing values
        self.elements_thickness = 8  # elements through thickness (ne)
        self.elements_length_factor = 20  # nL = L * this factors

        # Reference equilibrium parameters
        self.enc_ref = 57.32

        self.Dx_fixe = False  #Set to True for Dx = -67.5227 or -2 * R - 0.52 (experimental value) and False for Dx to be calculated

    def get_elements_length(self):
        return int(self.length * self.elements_length_factor)

    def get_elements_thickness(self):
        return self.elements_thickness

    def calculate_semicircle_radius(self):
        """Assume blade bends into a perfect semicircle: arc length L = pi * R"""
        return self.length / pi

    def get_physical_clamping_parameters(self):
        """
        Compute clamping parameters for ideal semicircular blade bending.
        enc is scaled with blade length; Dx ensures arc ends at 2R.
        Dy is kept zero unless asymmetry is introduced.
        """
        R = self.calculate_semicircle_radius()
        #enc_opt = self.enc_ref * (self.length / self.length_ref)
        enc_opt = self.enc_ref

        if self.Dx_fixe:
            Dx_opt = -2 * R -0.52  # Use fixed Dx value
            # Dx_opt = -67.5227  # (experimental value)
            print(f"INFO: Dx_fixe is True. Calculating Dx = -2 * R - 0.52 = {Dx_opt:.4f} (L={self.length:.2f} mm).")
        else:
            Dx_opt = -2 * R  # Calculate Dx for a semicircle
            print(f"INFO: Dx_fixe is False. Calculating Dx = -2 * R = {Dx_opt:.4f} (L={self.length:.2f} mm).")

        Dy_opt = -2 * R
        return Dx_opt, Dy_opt, enc_opt

    def get_compensated_clamping_position(self):
        return self.get_physical_clamping_parameters()[:2]

    def get_adjusted_enc_position(self):
        return self.get_physical_clamping_parameters()[2]

    def validate_parameters(self):
        """
        Validate that the blade parameters are within acceptable ranges
        """
        warnings = []

        # Check thickness limits
        if self.thickness < 0.1:
            warnings.append(f"Thickness {self.thickness} mm may be too small for reliable mesh")
        if self.thickness > 1.0:
            warnings.append(f"Thickness {self.thickness} mm may violate thin blade assumptions")

        # Check length limits
        if self.length < 50.0:
            warnings.append(f"Length {self.length} mm may result in very high resonance frequency")
        if self.length > 200.0:
            warnings.append(f"Length {self.length} mm may cause excessive parasitic resonances")

        # Check aspect ratio
        aspect_ratio = self.length / self.thickness
        if aspect_ratio < 100:
            warnings.append(f"Aspect ratio {aspect_ratio:.1f} may violate beam theory assumptions")
        if aspect_ratio > 1000:
            warnings.append(f"Aspect ratio {aspect_ratio:.1f} may cause numerical instabilities")

        # Check semicircle geometry feasibility
        R = self.calculate_semicircle_radius()
        if R < 10.0:
            warnings.append(f"Semicircle radius {R:.1f} mm may be too small for stable curvature.")

        return warnings

    def estimate_resonance_frequency(self):
        """
        Estimate the first resonance frequency using corrected TFE formulation
        """
        # Select material Young modulus (in MPa)
        if self.material == 'BE_CU':
            E = 131e3
        elif self.material == 'INVAR':
            E = 141e3
        else:  # STEEL
            E = 210e3

        # Geometry
        h_blade = self.thickness  # mm
        w = self.width  # mm
        L = self.length  # mm

        # Fixed geometric data for rod/mass system
        H = 3.875  # rod vertical thickness (mm)
        D = 39.99  # mass block height (mm)
        d = 13.96  # mass block length (mm)
        l = 79.2  # total rod length (mm)
        r = 7.0  # rod right segment (mm)
        R_rod = H  # rod right width (renamed from R to R_rod to avoid conflict with semicircle radius)
        rho = 7.85e-6  # steel density (kg/mm³)
        depth = 63.0  # structure width/depth (mm)

        # Recompute h (rod left segment)
        h_rod_segment = l - r - d  # Renamed h to h_rod_segment

        # Method of bars to compute I (kg·mm²)
        I_barre_rod_g = rho * depth * h_rod_segment * (H ** 3 / 3 + H * (h_rod_segment / 2) ** 2)
        I_barre_mass = rho * depth * d * (D ** 3 / 3 + D * (h_rod_segment + d / 2) ** 2)
        I_barre_rod_d = rho * depth * r * (R_rod ** 3 / 3 + R_rod * (h_rod_segment + d + r / 2) ** 2)

        # Total inertia
        I_system = I_barre_rod_g + I_barre_mass + I_barre_rod_d  # kg·mm²

        # Compute kth [N·mm]
        kth = (pi ** 2 / 6) * E * w * h_blade ** 3 / L ** 3  # N·mm # Corrected L to L from L_eff
        kth_mNm = kth * 1e3  # convert to mN·m

        # Fixed flexure stiffness and lever arm
        k_flex = 44.3  # mN·m
        r_s = 65.22  # mm

        # Frequency estimation [Hz]
        numerator = kth_mNm * r_s ** 2 + k_flex
        f_est = (1 / (2 * pi)) * sqrt(numerator / I_system)

        print(f"Blade stiffness kth = {kth_mNm:.3f} mN·m")
        print(f"System inertia I = {I_system:.3f} kg·mm²")

        return f_est

    def print_optimization_details(self):
        Dx_opt, Dy_opt, enc_opt = self.get_physical_clamping_parameters()
        R = self.calculate_semicircle_radius()

        print(f"\n=== GEOMETRY-BASED CLAMPING CONFIGURATION ===")
        print(f"Blade: L={self.length:.2f} mm, h={self.thickness:.2f} mm, w={self.width:.1f} mm")
        print(f"Semicircle radius: R={R:.2f} mm")
        print(f"Clamping position: enc = {enc_opt:.2f} mm")
        print(f"  -> Dx = {Dx_opt:.4f} mm")
        print(f"  -> Dy = {Dy_opt:.4f} mm")
        print(f"Reference ratios:")
        print(
            f"  L ratio = {self.length / self.length_ref:.3f}, h ratio = {self.thickness / self.thickness_ref:.3f}, w ratio = {self.width / self.width_ref:.3f}")
        print(f"Estimated f0 = {self.estimate_resonance_frequency():.2f} Hz")
        print(f"============================================\n")


class SimulationConfig:
    """General simulation parameters with optimized clamping"""

    def __init__(self, blade_config):
        # Time parameters
        self.final_time = 30.0
        self.loading_time = 10.0
        self.stabilization_time = 12.0

        # Temperature parameters
        self.enable_thermal = True
        self.temp_initial_kelvin = 273.15 + 10.0
        self.temp_final_kelvin = 273.15 + 50.0
        self.temp_start_time = 20.0
        self.temp_end_time = 35.0

        # Plasticity plots and data
        self.enable_plasticityData = False  # If True then all the plasticity data will be calculated in the simulation.
        # Else only the elastic data will be calculated. Improve the simulation time if put to False
        # /!\ Advice put True for Invar and False for Be-Cu.

        # Time step control
        self.adaptive_timestep = False  # Set to True for adaptive, False for fixed

        # Fixed time step parameters (used when adaptive_timestep = False)
        self.fixed_timestep_size = 0.005  # Fixed time step size
        self.fixed_initial_timestep = 0.01  # Initial time step for fixed mode

        self.ToporBottom = True # Set to True for bottom clamp, False for top clamp

        # Optimized clamping parameters
        self.Dx, self.Dy = blade_config.get_compensated_clamping_position()
        self.Dx1 = 0.0
        self.angleClamp = 0.0
        self.enc = blade_config.get_adjusted_enc_position()


def setup_optimized_blade_system(user_overrides=None):
    blade_config = BladeConfig()

    if user_overrides:
        for key, value in user_overrides.items():
            if hasattr(blade_config, key):
                setattr(blade_config, key, value)

    sim_config = SimulationConfig(blade_config)
    warnings = blade_config.validate_parameters()
    blade_config.print_optimization_details()

    print("=== FINAL CONFIGURATION SUMMARY ===")
    print(f"Blade: L={blade_config.length} mm, h={blade_config.thickness} mm, w={blade_config.width} mm")
    print(f"Material: {blade_config.material}")
    print(f"Clamping: Dx = {sim_config.Dx:.2f} mm, Dy = {sim_config.Dy:.2f} mm, enc = {sim_config.enc:.2f} mm")
    print(f"Mesh: {blade_config.get_elements_length()} x {blade_config.get_elements_thickness()} elements")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f" - {w}")
    print("====================================\n")

    return blade_config, sim_config, warnings


def setup_temperature_dependent_properties(material):
    """Return temperature-dependent E and CTE functions for the given material."""

    if material == 'BE_CU':
        fctE = PieceWiseLinearFunction()
        fctE.setData(283.15, 131e3)  # 10C (reference)
        fctE.setData(290.15, 130.8e3)  # 17C
        fctE.setData(293.15, 130.7e3)  # 20C
        fctE.setData(300.15, 130.4e3)  # 27C
        fctE.setData(310.15, 130.0e3)  # 37C
        fctE.setData(320.15, 129.5e3)  # 47C
        fctE.setData(323.15, 129.3e3)  # 50C

        fctCTE = PieceWiseLinearFunction()
        fctCTE.setData(283.15, 17.0e-6)
        fctCTE.setData(293.15, 17.2e-6)
        fctCTE.setData(300.15, 17.4e-6)
        fctCTE.setData(310.15, 17.6e-6)
        fctCTE.setData(320.15, 17.8e-6)
        fctCTE.setData(323.15, 17.9e-6)

    elif material == 'INVAR':
        fctE = PieceWiseLinearFunction()
        fctE.setData(283.15, 141e3)
        fctE.setData(293.15, 140.8e3)
        fctE.setData(300.15, 140.7e3)
        fctE.setData(310.15, 140.6e3)
        fctE.setData(320.15, 140.5e3)
        fctE.setData(323.15, 140.4e3)

        fctCTE = PieceWiseLinearFunction()
        fctCTE.setData(283.15, 1.1e-6)
        fctCTE.setData(293.15, 1.2e-6)
        fctCTE.setData(300.15, 1.3e-6)
        fctCTE.setData(310.15, 1.4e-6)
        fctCTE.setData(320.15, 1.5e-6)
        fctCTE.setData(323.15, 1.6e-6)

    elif material == 'INVAR_CW_HARD':
        fctE = PieceWiseLinearFunction()
        fctE.setData(-260 + 273.15, 126e3)
        fctE.setData(-187 + 273.15, 129e3)
        fctE.setData(-114 + 273.15, 133e3)
        fctE.setData(-40.2 + 273.15, 139e3)
        fctE.setData(33.2 + 273.15, 144e3)
        fctE.setData(107 + 273.15, 149e3)
        fctE.setData(180 + 273.15, 153e3)
        fctE.setData(253 + 273.15, 156e3)
        fctE.setData(327 + 273.15, 157e3)
        fctE.setData(400 + 273.15, 155e3)

        fctCTE = PieceWiseLinearFunction()
        fctCTE.setData(-260 + 273.15, 1.41e-6)
        fctCTE.setData(-221 + 273.15, 1.64e-6)
        fctCTE.setData(-182 + 273.15, 1.90e-6)
        fctCTE.setData(-143 + 273.15, 1.87e-6)
        fctCTE.setData(-104 + 273.15, 1.67e-6)
        fctCTE.setData(-65.4 + 273.15, 1.44e-6)
        fctCTE.setData(-26.5 + 273.15, 1.27e-6)
        fctCTE.setData(12.5 + 273.15, 1.20e-6)
        fctCTE.setData(51.4 + 273.15, 1.26e-6)
        fctCTE.setData(90.4 + 273.15, 1.44e-6)
        fctCTE.setData(129 + 273.15, 1.76e-6)
        fctCTE.setData(168 + 273.15, 2.21e-6)
        fctCTE.setData(207 + 273.15, 2.86e-6)
        fctCTE.setData(246 + 273.15, 3.81e-6)
        fctCTE.setData(285 + 273.15, 5.5e-6)
        fctCTE.setData(324 + 273.15, 7.08e-6)
        fctCTE.setData(363 + 273.15, 8.16e-6)
        fctCTE.setData(402 + 273.15, 8.87e-6)
        fctCTE.setData(441 + 273.15, 9.48e-6)
        fctCTE.setData(480 + 273.15, 10.1e-6)
    else:
        raise ValueError(f"Unsupported material: {material}")

    return fctE, fctCTE



def getMetafor(d={}):
    """
    Optimized main function with physics-based compensation
    """

    # Initialize configurations with physics-based optimization
    blade_config, sim_config, warnings = setup_optimized_blade_system(d)

    # Apply user overrides
    p = {'postpro': False}
    p.update(d)

    # Override parameters if provided
    if 'blade_material' in d:
        blade_config.material = d['blade_material']
    if 'blade_thickness' in d:
        blade_config.thickness = d['blade_thickness']
    if 'blade_length' in d:
        blade_config.length = d['blade_length']
    if 'blade_width' in d:
        blade_config.width = d['blade_width']
    if 'final_time' in d:
        sim_config.final_time = d['final_time']
    if 'loading_time' in d:
        sim_config.loading_time = d['loading_time']
    if 'enable_thermal' in d:
        sim_config.enable_thermal = d['enable_thermal']

    print(f"=== OPTIMIZED SIMULATION CONFIGURATION ===")
    print(f"Blade Material: Beryllium-Copper (Temperature Dependent)")
    print(
        f"Blade Dimensions: L={blade_config.length:.2f}mm, e={blade_config.thickness:.2f}mm, w={blade_config.width:.1f}mm")
    print(f"Thermal effects: {'Enabled' if sim_config.enable_thermal else 'Disabled'}")
    print(f"=== OPTIMIZED SEQUENCE ===")
    print(f"Phase 1 (0-{sim_config.loading_time}s): Mechanical loading & blade bending")
    print(
        f"Phase 2 ({sim_config.stabilization_time}s): System stabilization at {sim_config.temp_initial_kelvin - 273.15:.0f}C")
    print(
        f"Phase 3 ({sim_config.temp_start_time}-{sim_config.temp_end_time}s): Thermal loading {sim_config.temp_initial_kelvin - 273.15:.0f}C -> {sim_config.temp_final_kelvin - 273.15:.0f}C")
    print(f"Expected effect: Thermal expansion will displace the mass")  # MODIFIED MESSAGE
    print(f"===================================")

    # Continue with Metafor setup using optimized parameters
    metafor = Metafor()
    domain = metafor.getDomain()

    # Use optimized configuration parameters
    e = blade_config.thickness
    L = blade_config.length
    Dx = sim_config.Dx  # Physics-based Dx
    Dy = sim_config.Dy  # Physics-based Dy
    enc = sim_config.enc  # Physics-based enc

    T = sim_config.final_time
    T_load = sim_config.loading_time
    Dx1 = sim_config.Dx1
    angleClamp = sim_config.angleClamp

    # Fixed rod and mass parameters
    l = 79.2  # total length
    H = 3.875  # thickness
    r = 7  # distance between mass and rod end
    R = H

    # mass
    D = 39.99  # height
    d = 13.96  # length
    y = D / 2  # the mass is centered on the rod
    h = l - r - d

    # mesh parameters
    ne = blade_config.elements_thickness
    nL = blade_config.get_elements_length()
    nd = 10
    nr = 1
    n56 = 2  # rod vertical (/2)
    n7 = 5  # rod horizontal
    n9 = 1  # rod horiz 2
    n14 = 3  # mass vertical 1
    n15 = 17  # mass vertical 2

    # Geometry
    geometry = domain.getGeometry()
    geometry.setDimPlaneStrain(1.0)

    pointset = geometry.getPointSet()

    # Lame ressort
    p1 = pointset.define(1, enc + L, e / 2)
    p2 = pointset.define(2, enc + L, -e / 2)

    # Tige
    p5 = pointset.define(5, 0.0, H / 2)
    p6 = pointset.define(6, 0.0, -H / 2)
    p7 = pointset.define(7, h, -H / 2)
    p8 = pointset.define(8, h, H / 2)

    # Masse
    p9 = pointset.define(9, h, D - y)
    p10 = pointset.define(10, h, -y)
    p11 = pointset.define(11, h + d, -y)
    p12 = pointset.define(12, h + d, D - y)

    # Bout de tige
    p13 = pointset.define(13, h + d, R / 2)
    p14 = pointset.define(14, h + d, -R / 2)
    p15 = pointset.define(15, h + d + r, -R / 2)
    p16 = pointset.define(16, h + d + r, R / 2)

    # Compatibilité géométrique
    p17 = pointset.define(17, 0.0, 0.0)
    p171 = pointset.define(171, 0.0, e / 2)
    p172 = pointset.define(172, 0.0, -e / 2)
    p181 = pointset.define(181, h, e / 2)
    p182 = pointset.define(182, h, -e / 2)
    p191 = pointset.define(191, h + d, e / 2)
    p192 = pointset.define(192, h + d, -e / 2)
    p20 = pointset.define(20, h + d + r, 0)
    p201 = pointset.define(201, h + d + r, e / 2)
    p202 = pointset.define(202, h + d + r, -e / 2)

    # Plan médian
    p21 = pointset.define(21, enc, H / 2)
    p22 = pointset.define(22, enc, -H / 2)

    # Sol
    p25 = pointset.define(25, h + d + r, -y)
    p26 = pointset.define(26, 0, -y)

    # Point de serrage opposé
    p271 = pointset.define(271, Dx + enc, 0.0)  # Left clamping point
    p272 = pointset.define(272, enc, Dy)  # Above clamping point
    p273 = pointset.define(273, enc, -Dy)  # Under clamping point

    # Base lame
    p28 = pointset.define(28, enc, e / 2)
    p29 = pointset.define(29, enc, -e / 2)

    # Ressort de charnière
    p30 = pointset.define(30, 0.0, -H / 2)

    curveset = geometry.getCurveSet()

    # Lame ressort
    c1 = curveset.add(Line(1, p1, p2))  # extrémité droite lame
    c2 = curveset.add(Line(2, p29, p2))  # partie inférieure lame
    c3 = curveset.add(Line(3, p28, p1))  # partie supérieure lame
    c4 = curveset.add(Line(4, p28, p29))  # extrémité gauche lame

    # Tige
    c51 = curveset.add(Line(51, p5, p171))  # extrémité gauche supérieure
    c52 = curveset.add(Line(52, p171, p172))  # extrémité gauche milieu
    c6 = curveset.add(Line(6, p172, p6))  # extrémité gauche inférieure
    c71 = curveset.add(Line(71, p6, p22))  # partie inférieure gauche
    c72 = curveset.add(Line(72, p22, p7))  # partie inférieure droite
    c101 = curveset.add(Line(101, p7, p182))  # extrémité droite inférieure
    c102 = curveset.add(Line(102, p182, p181))  # extrémité droite milieu
    c11 = curveset.add(Line(11, p181, p8))  # extrémité droite supérieure
    c121 = curveset.add(Line(121, p8, p21))  # partie supérieure droite
    c122 = curveset.add(Line(122, p21, p5))  # partie supérieure gauche

    # Masse
    c14 = curveset.add(Line(14, p9, p8))  # extrémité gauche supérieure
    c15 = curveset.add(Line(15, p7, p10))  # extrémité gauche inférieure
    c16 = curveset.add(Line(16, p10, p11))  # partie inférieure
    c17 = curveset.add(Line(17, p11, p14))  # extrémité droite inférieure
    c181 = curveset.add(Line(181, p14, p192))  # extrémité droite milieu bas
    c182 = curveset.add(Line(182, p192, p191))  # extrémité droite milieu
    c19 = curveset.add(Line(19, p191, p13))  # extrémité droite milieu haut
    c20 = curveset.add(Line(20, p13, p12))  # extrémité droite supérieure
    c21 = curveset.add(Line(21, p12, p9))  # partie supérieure

    # Bout de tige
    c22 = curveset.add(Line(22, p14, p15))  # partie inférieure
    c231 = curveset.add(Line(231, p15, p202))  # extrémité droite milieu bas
    c232 = curveset.add(Line(232, p202, p201))  # extrémité droite milieu
    c24 = curveset.add(Line(24, p201, p16))  # extrémité droite milieu haut
    c25 = curveset.add(Line(25, p16, p13))  # partie supérieure

    # Sol
    c26 = curveset.add(Line(26, p25, p26))

    # Plan médian sup (tige gauche)
    c28 = curveset.add(Line(28, p28, p181))

    # Plan médian inf (tige droite)
    c30 = curveset.add(Line(30, p29, p182))

    # Plan coupe (vertical)
    c31 = curveset.add(Line(31, p21, p28))
    c32 = curveset.add(Line(32, p29, p22))

    wireset = geometry.getWireSet()

    # Lame ressort
    w1 = wireset.add(Wire(1, [c4, c2, c1, c3]))

    # Tige partie gauche
    w2 = wireset.add(Wire(2, [c51, c52, c6, c71, c32, c4, c31, c122]))

    # Tige partie droite
    w3 = wireset.add(Wire(3, [c31, c4, c32, c72, c101, c102, c11, c121]))

    # Masse
    w4 = wireset.add(Wire(4, [c14, c11, c102, c101, c15, c16, c17, c181, c182, c19, c20, c21]))

    # Bout de tige
    w5 = wireset.add(Wire(5, [c19, c182, c181, c22, c231, c232, c24, c25]))

    # Sol
    w6 = wireset.add(Wire(6, [c26]))

    sideset = geometry.getSideSet()
    s1 = sideset.add(Side(1, [w1]))
    s2 = sideset.add(Side(2, [w2]))
    s3 = sideset.add(Side(3, [w3]))
    s4 = sideset.add(Side(4, [w4]))
    s5 = sideset.add(Side(5, [w5]))
    s6 = sideset.add(Side(6, [w6]))

    if 0:
        win = VizWin()
        win.add(pointset)
        win.add(curveset)
        win.open()
        input()

    # Maillage 1D corrigé
    prog = 5

    # Lame ressort
    SimpleMesher1D(c1).execute(ne)  # extrémité droite lame
    SimpleMesher1D(c2).execute(nL)  # partie inférieure lame
    SimpleMesher1D(c3).execute(nL)  # partie supérieure lame
    SimpleMesher1D(c4).execute(ne)  # extrémité gauche lame

    # Tige
    SimpleMesher1D(c51).execute(n56)  # extrémité gauche supérieure
    SimpleMesher1D(c52).execute(ne)  # extrémité gauche milieu (même que c4)
    SimpleMesher1D(c6).execute(n56)  # extrémité gauche inférieure
    SimpleMesher1D(c71).execute(n7)  # partie inférieure gauche
    SimpleMesher1D(c72).execute(n9)  # partie inférieure droite
    SimpleMesher1D(c101).execute(n56)  # extrémité droite inférieure
    SimpleMesher1D(c102).execute(ne)  # extrémité droite milieu (même que c52)
    SimpleMesher1D(c11).execute(n56)  # extrémité droite supérieure
    SimpleMesher1D(c121).execute(n9)  # partie supérieure droite
    SimpleMesher1D(c122).execute(n7)  # partie supérieure gauche

    # Masse
    SimpleMesher1D(c14).execute(n14)  # extrémité gauche supérieure
    SimpleMesher1D(c15).execute(n15, 1 / prog)  # extrémité gauche inférieure
    SimpleMesher1D(c16).execute(nd)  # partie inférieure
    SimpleMesher1D(c17).execute(n15, prog)  # extrémité droite inférieure
    SimpleMesher1D(c181).execute(n56)  # extrémité droite milieu bas
    SimpleMesher1D(c182).execute(ne)  # extrémité droite milieu (même que c102)
    SimpleMesher1D(c19).execute(n56)  # extrémité droite milieu haut
    SimpleMesher1D(c20).execute(n14)  # extrémité droite supérieure
    SimpleMesher1D(c21).execute(nd)  # partie supérieure

    # Bout de tige
    SimpleMesher1D(c22).execute(nr)  # partie inférieure
    SimpleMesher1D(c231).execute(n56)  # extrémité droite milieu bas
    SimpleMesher1D(c232).execute(ne)  # extrémité droite milieu (même que c182)
    SimpleMesher1D(c24).execute(n56)  # extrémité droite milieu haut
    SimpleMesher1D(c25).execute(nr)  # partie supérieure

    # Sol
    # SimpleMesher1D(c26).execute(1)  # sol  /!\ Pas de maillage au niveau du sol

    # Plans de coupe
    # SimpleMesher1D(c28).execute(nL)  # plan médian sup (même que c3)
    # SimpleMesher1D(c30).execute(nL)  # plan médian inf (même que c2)
    SimpleMesher1D(c31).execute(n56)  # plan coupe vertical sup
    SimpleMesher1D(c32).execute(n56)  # plan coupe vertical inf

    # Maillage 2D corrigé
    TransfiniteMesher2D(s1).execute(True)
    TransfiniteMesher2D(s2).execute2(((51, 52, 6), 71, (32, 4, 31), 122))
    TransfiniteMesher2D(s3).execute2(((31, 4, 32), 72, (101, 102, 11), 121))
    TransfiniteMesher2D(s4).execute2(((14, 11, 102, 101, 15), 16, (17, 181, 182, 19, 20), 21))
    TransfiniteMesher2D(s5).execute2(((19, 182, 181), 22, (231, 232, 24), 25))

    if 0:
        win = VizWin()
        win.add(geometry.getMesh().getPointSet())
        win.add(geometry.getMesh().getCurveSet())
        win.add(curveset)
        win.open()
        input()

    # OPTIMIZED MATERIALS - Temperature dependent Be-Cu
    materials = domain.getMaterialSet()
    laws = domain.getMaterialLawSet()

    # Elastoplastic behavior laws
    # Elastoplastic behavior laws according to the chosen material
    if blade_config.material == 'BE_CU':
        laws.define(1, LinearIsotropicHardening)
        laws(1).put(IH_SIGEL, 1000.0)  # pproximate elastic limit for BE-CU (~1000 MPa)
        laws(1).put(IH_H, 1000.0)  # Hardening modulus BE-CU (moderate value)
        yield_num = 1
    elif blade_config.material == 'INVAR':
        laws.define(2, LinearIsotropicHardening)
        laws(2).put(IH_SIGEL, 250.0)  # Approximate elastic limit for Invar (~250 MPa): Elastic Limit: 240-725 MPa https://www.azom.com/properties.aspx?ArticleID=515
        laws(2).put(IH_H, 600.0)  # Hardening modulus Invar (moderate value) : h = 500-800 MPa
        yield_num = 2
    elif blade_config.material == 'INVAR_CW_HARD':
        laws.define(5, LinearIsotropicHardening)
        fctSigY = PieceWiseLinearFunction()
        fctSigY.setData(-260 + 273.15, 1090.0)
        fctSigY.setData(-229 + 273.15, 1000.0)
        fctSigY.setData(-198 + 273.15, 924.0)
        fctSigY.setData(-167 + 273.15, 860.0)
        fctSigY.setData(-136 + 273.15, 805.0)
        fctSigY.setData(-105 + 273.15, 759.0)
        fctSigY.setData(-73.5 + 273.15, 720.0)
        fctSigY.setData(-42.4 + 273.15, 686.0)
        fctSigY.setData(-11.3 + 273.15, 655.0)
        fctSigY.setData(19.9 + 273.15, 627.0)
        fctSigY.setData(25.0 + 273.15, 622.0)

        laws(5).put(IH_H, 700.0)
        laws(5).put(IH_SIGEL, fctSigY.evaluate(283.15))
        yield_num = 5
    else:
        raise ValueError(f"Unsupported material: {blade_config.material}")

    laws.define(3, LinearIsotropicHardening)
    laws(3).put(IH_SIGEL, 400.0)  # Elastic limit for steel
    laws(3).put(IH_H, 1000.0)  # Hardening modulus

    # Setup temperature-dependent functions
    fctE, fctCTE = setup_temperature_dependent_properties(blade_config.material)

    if sim_config.enable_thermal:
        # to use thermal properties
        #https://material-properties.org/beryllium-copper-density-strength-hardness-melting-point/#google_vignette
        materials.define(1, TmEvpIsoHHypoMaterial)
        if blade_config.material == 'BE_CU':
            materials(1).put(MASS_DENSITY, 8.36e-9)
            materials(1).put(POISSON_RATIO, 0.285)
            materials(1).put(CONDUCTIVITY, 105.0)
            materials(1).put(HEAT_CAPACITY, 420.e6)
            materials(1).put(ELASTIC_MODULUS, 1.0)
            materials(1).put(THERM_EXPANSION, 1.0)
        elif blade_config.material == 'INVAR':
            materials(1).put(MASS_DENSITY, 8.1e-9)
            materials(1).put(POISSON_RATIO, 0.29)
            materials(1).put(CONDUCTIVITY, 10.0)
            materials(1).put(HEAT_CAPACITY, 500.e6)
            materials(1).put(ELASTIC_MODULUS, 1.0)
            materials(1).put(THERM_EXPANSION, 1.0)
        elif blade_config.material == 'INVAR_CW_HARD':
            materials(1).put(MASS_DENSITY, 8.15e-9)  # kg/mm^3
            materials(1).put(POISSON_RATIO, 0.29)
            materials(1).put(CONDUCTIVITY, 13.9)
            materials(1).put(HEAT_CAPACITY, 479e6)  # µJ/kg.K
            materials(1).put(ELASTIC_MODULUS, 1.0)
            materials(1).put(THERM_EXPANSION, 1.0)

        materials(1).depend(ELASTIC_MODULUS, fctE, Field1D(TO, RE))
        materials(1).depend(THERM_EXPANSION, fctCTE, Field1D(TO, RE))
        materials(1).put(DISSIP_TE, 0.0)
        materials(1).put(DISSIP_TQ, 0.0)
        materials(1).put(YIELD_NUM, yield_num)
        # Note: No YIELD_NUM and TmElastHypoMaterial if elastic only: Use TmEvpIsoHHypoMaterial if plastic

        # Structure material - Use TmElastHypoMaterial also for coherence
        # http://metafor.ltas.ulg.ac.be/dokuwiki/doc/user/elements/volumes/iso_hypo_materials
        materials.define(2, TmEvpIsoHHypoMaterial)  # Change here
        materials(2).put(MASS_DENSITY, 8.0415e-9)
        materials(2).put(ELASTIC_MODULUS, 210e3)
        materials(2).put(POISSON_RATIO, 0.3)
        materials(2).put(THERM_EXPANSION, 0.0)  # No thermal expansion of the structure
        materials(2).put(CONDUCTIVITY, 50)
        materials(2).put(HEAT_CAPACITY, 500.e6)
        materials(2).put(DISSIP_TE, 0.0)
        materials(2).put(DISSIP_TQ, 0.0)
        materials(2).put(YIELD_NUM, 3)
        # Note: No YIELD_NUM and TmElastHypoMaterial if elastic only

    else:
        # Mechanical only - Use ElastHypoMaterial if simple elastic (and remove Yield_num)
        materials.define(1, EvpIsoHHypoMaterial)
        if blade_config.material == 'BE_CU':
            materials(1).put(MASS_DENSITY, 8.36e-9)
            materials(1).put(POISSON_RATIO, 0.285)
            materials(1).put(ELASTIC_MODULUS, 131e3)  # Value
        elif blade_config.material == 'INVAR':
            materials(1).put(MASS_DENSITY, 8.1e-9)
            materials(1).put(POISSON_RATIO, 0.29)
            materials(1).put(ELASTIC_MODULUS, 141e3)  # Value
        elif blade_config.material == 'INVAR_CW_HARD':
            materials(1).put(MASS_DENSITY, 8.15e-9)
            materials(1).put(POISSON_RATIO, 0.29)
            materials(1).put(ELASTIC_MODULUS, 141e3)

        materials(1).put(YIELD_NUM, yield_num)

        # Steel structure
        materials.define(2, EvpIsoHHypoMaterial)
        materials(2).put(MASS_DENSITY, 8.0415e-9)
        materials(2).put(ELASTIC_MODULUS, 210e3)
        materials(2).put(POISSON_RATIO, 0.3)
        materials(2).put(YIELD_NUM, 3)

    # Spring material
    materials.define(4, ConstantSpringMaterial)
    materials(4).put(SPRING_FK, 11.7211) # Rotational stiffness N/mm

    # OPTIMIZED GRAVITY FUNCTION - Smoother loading
    fctG = PieceWiseLinearFunction()
    fctG.setData(0.0, 0.0)  # Start with no gravity for better convergence
    fctG.setData(T_load / 20, 1.0)  # Apply gravity quickly but smoothly
    fctG.setData(T_load, 1.0)
    fctG.setData(T, 1.0)

    # Element properties
    if sim_config.enable_thermal:
        prp1 = ElementProperties(TmVolume2DElement)  # Blade
        prp2 = ElementProperties(TmVolume2DElement)  # Structure
    else:
        prp1 = ElementProperties(Volume2DElement)
        prp2 = ElementProperties(Volume2DElement)

    # Blade properties
    prp1.put(MATERIAL, 1)
    prp1.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)
    prp1.put(THICKNESS, blade_config.width)

    # Structure properties
    prp2.put(MATERIAL, 2)
    prp2.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)
    prp2.put(THICKNESS, 63.0)
    prp2.put(GRAVITY_Y, -9.81e3)
    prp2.depend(GRAVITY_Y, fctG, Field1D(TM))


    # Apply properties
    app = FieldApplicator(1)
    app.push(s1)
    app.addProperty(prp1)
    domain.getInteractionSet().add(app)

    app2 = FieldApplicator(2)
    app2.push(s2)
    app2.push(s3)
    app2.push(s4)
    app2.push(s5)
    app2.addProperty(prp2)
    domain.getInteractionSet().add(app2)

    # Spring setup (unchanged)
    groupset = geometry.getGroupSet()
    groupset.add(Group(1));
    groupset(1).addMeshPoint(p6)
    groupset.add(Group(2));
    groupset(2).addMeshPoint(p30)
    springMesher = CellLineMesher(groupset(1), groupset(2))
    springMesher.execute()

    prp4 = ElementProperties(Spring2DElement)
    prp4.put(MATERIAL, 4)
    prp4.put(SPRING_CLI, 0)
    prp4.put(STIFFMETHOD, STIFF_ANALYTIC)
    app4 = FieldApplicator(4)
    app4.push(groupset(1))
    app4.addProperty(prp4)
    domain.getInteractionSet().add(app4)

    # Boundary conditions
    domain.getLoadingSet().define(p17, Field1D(TX, RE), 0.0)
    domain.getLoadingSet().define(p17, Field1D(TY, RE), 0.0)
    domain.getLoadingSet().define(c26, Field1D(TX, RE), 0.0)
    domain.getLoadingSet().define(c26, Field1D(TY, RE), 0.0)
    domain.getLoadingSet().define(p30, Field1D(TY, RE), 0.0)
    domain.getLoadingSet().define(p30, Field1D(TX, RE), 0.0)

    # Rotation setup
    pa1 = pointset.define(23, enc + L / 2, 0.0)
    pa2 = pointset.define(24, enc + L / 2, 0.0, 1.0)
    axe1 = Axe(pa1, pa1)
    axe1.setSymZ1(1.0)

    if sim_config.ToporBottom:
        pa3 = pointset.define(31, Dx1 + enc, Dy, 0.0)
        pa4 = pointset.define(32, Dx1 + enc, Dy, 1.0)
        axe2 = Axe(pa3, pa4)
    else:
        pa3 = pointset.define(31, Dx1 + enc, -Dy, 0.0)
        pa4 = pointset.define(32, Dx1 + enc, -Dy, 1.0)
        axe2 = Axe(pa3, pa4)


    # OPTIMIZED LOADING FUNCTIONS - Smoother and faster convergence
    fctX = PieceWiseLinearFunction()
    fctX.setData(0.0, 0.0)
    fctX.setData(T_load / 8, 0.0)
    fctX.setData(T_load / 2, 1.0)
    fctX.setData(3 * T_load / 4, 1.0)
    fctX.setData(T_load, 1.0)
    fctX.setData(T, 1.0)

    fctY = PieceWiseLinearFunction()
    fctY.setData(0.0, 0.0)
    fctY.setData(T_load / 8, 0.0)
    fctY.setData(T_load / 2, 1.0)
    fctY.setData(3 * T_load / 4, 1.0)
    fctY.setData(T_load, 1.0)
    fctY.setData(T, 1.0)

    domain.getLoadingSet().define(pa1, Field1D(TX, RE), 0.0, fctX)
    domain.getLoadingSet().define(pa2, Field1D(TX, RE), 0.0, fctX)

    if sim_config.ToporBottom:
        domain.getLoadingSet().define(pa1, Field1D(TY, RE), Dy, fctY)
        domain.getLoadingSet().define(pa2, Field1D(TY, RE), Dy, fctY)

    else:
        domain.getLoadingSet().define(pa1, Field1D(TY, RE), -Dy, fctY)
        domain.getLoadingSet().define(pa2, Field1D(TY, RE), -Dy, fctY)

    # OPTIMIZED ROTATION FUNCTIONS - Much smoother for better convergence
    fctR = PieceWiseLinearFunction()
    fctR.setData(0.0, 0.0)
    fctR.setData(T_load / 10, 0.0)
    fctR.setData(T_load / 2, 1.0)
    fctR.setData(3 * T_load / 4, 1.0)
    fctR.setData(T_load, 1.0)
    fctR.setData(T, 1.0)

    fctR2 = PieceWiseLinearFunction()
    fctR2.setData(0.0, 0.0)
    fctR2.setData(T_load / 10, 0.0)
    fctR2.setData(T_load / 2, 0.0)
    fctR2.setData(3 * T_load / 4, 0.0)
    fctR2.setData(T_load, 1.0)
    fctR2.setData(T, 1.0)

    if sim_config.ToporBottom:
        domain.getLoadingSet().defineRot2(c1, Field3D(TXTYTZ, RE), axe2, angleClamp, fctR2, axe1, - 180, fctR, False)
    else:
        domain.getLoadingSet().defineRot2(c1, Field3D(TXTYTZ, RE), axe2, angleClamp, fctR2, axe1,  180, fctR, False)


    # OPTIMIZED THERMAL LOADING
    if sim_config.enable_thermal:
        print(f"Configuring OPTIMIZED thermal sequence:")
        print(f"  - Temperature-dependent material properties for Be-Cu")
        print(
            f"  - Temperature increase: {sim_config.temp_initial_kelvin - 273.15:.0f}C ->  {sim_config.temp_final_kelvin - 273.15:.0f}C")

        # Initial temperature conditions - IMPORTANT: Both absolute and relative
        Tabs = sim_config.temp_initial_kelvin  # Use initial temp as absolute reference

        initcondset = metafor.getDomain().getInitialConditionSet()
        # Set initial conditions for all sides
        for side_num in [1, 2, 3, 4, 5]:
            side = sideset(side_num)
            initcondset.define(side, Field1D(TO, AB), Tabs)  # Absolute reference
            initcondset.define(side, Field1D(TO, RE), 0.0)  # Start at 0 relative to reference

            # CORRECTED thermal loading function
        fctT = PieceWiseLinearFunction()
        fctT.setData(0.0, 0.0)  # Start at reference (20C)
        fctT.setData(sim_config.loading_time, 0.0)  # No thermal change during mechanical loading
        fctT.setData(sim_config.temp_start_time, 0.0)  # Start thermal loading
        # CRITICAL: Relative temperature change from reference
        temp_delta = sim_config.temp_final_kelvin - sim_config.temp_initial_kelvin
        fctT.setData(sim_config.temp_end_time, temp_delta)  # +30K relative change
        fctT.setData(sim_config.final_time, temp_delta)  # Maintain final state

        # Apply thermal loading
        for side_num in [1, 2, 3, 4, 5]:
            side = sideset(side_num)
            domain.getLoadingSet().define(side, Field1D(TO, RE), 1.0, fctT)

    # Ground displacement - Trigger floor movement during peak mass height ----------------------------------------------------------------------Modified compared to V10
    fctSol = PieceWiseLinearFunction()
    fctSol.setData(0.0, 0.0)
    fctSol.setData(3.0, 0.0)  # Ground stays fixed until 3s
    fctSol.setData(4.0, 1.0)  # Ground moves down between 3s and 4s
    fctSol.setData(T, 1.0)  # Ground remains down for the rest of the simulation

    if sim_config.ToporBottom:
        DSol = -20  # Downward displacement in mm (put to 20mm to avoid contact with the mass)
    else:
        DSol = -10  # Downward displacement in mm

    domain.getLoadingSet().define(c26, Field1D(TY, RE), DSol, fctSol)

    # Contact
    materials.define(3, FrictionlessContactMaterial)
    materials(3).put(PEN_NORMALE, 1e2)
    materials(3).put(PROF_CONT, 1.0)

    prp3 = ElementProperties(Contact2DElement)
    prp3.put(MATERIAL, 3)
    prp3.put(AREAINCONTACT, AIC_ONCE)

    ci = RdContactInteraction(3)
    ci.setTool(curveset(26))
    ci.push(curveset(16))
    ci.addProperty(prp3)
    domain.getInteractionSet().add(ci)

    # OPTIMIZED TIME INTEGRATION - Key for performance improvement
    if sim_config.enable_thermal:
        # Mechanical time integration - OPTIMIZED for stability
        tiMech = AlphaGeneralizedTimeIntegration(metafor)  #----------------------------------------------Line change (quasistatic -> ALPHA)
        # Note: convergence control via iteration handlers rather than setMaxNumberOfLoadIncrements

        # Thermal time integration - OPTIMIZED parameters
        tiTher = TrapezoidalThermalTimeIntegration(metafor)
        tiTher.setTheta(1.0)  # Fully implicit for stability with large temperature changes

        # Staggered integration - OPTIMIZED coupling
        ti = StaggeredTmTimeIntegration(metafor)
        ti.setIsAdiabatic(False)
        ti.setWithStressReevaluation(False)  # Re-evaluate stress after thermal at the expense of time
        # True = stronger but slower coupling
        # False = weaker but faster coupling
        ti.setMechanicalTimeIntegration(tiMech)
        ti.setThermalTimeIntegration(tiTher)
        metafor.setTimeIntegration(ti)


        # Thermal iteration manager - OPTIMIZED for convergence
        tim = metafor.getThermalIterationManager()
        tim.setResidualComputationMethod(Method4ResidualComputation())
        tim.setMaxNbOfIterations(15)  # Increased for thermal stability
        tim.setResidualTolerance(1e-5)

    else:
        # Pure mechanical - faster integration
        ti = AlphaGeneralizedTimeIntegration(metafor)
        metafor.setTimeIntegration(ti)

    # TIME STEP MANAGEMENT - ADAPTIVE OR FIXED
    tsm = metafor.getTimeStepManager()

    if sim_config.adaptive_timestep:
        print("INFO: Using ADAPTIVE time stepping.")
        tsm.setInitialTime(0.0, 0.01)  # Smaller initial step for better convergence

        # ADAPTIVE TIME STEPPING based on simulation phases - ENHANCED
        if sim_config.enable_thermal:
            # Phase 1: Mechanical loading (0 to T_load) - progressive refinement
            tsm.setNextTime(sim_config.loading_time * 0.25, 6, 0.01)  # Early loading Up to 2.5s
            tsm.setNextTime(sim_config.loading_time * 0.5, 6, 0.005)  # Mid loading Up to 5s
            tsm.setNextTime(sim_config.loading_time * 0.75, 12, 0.005)  # Late loading Up to 7.5s
            tsm.setNextTime(sim_config.loading_time, 13, 0.005)  # Final mechanical phase Up to 10s

            # Phase 2: Stabilization (T_load to temp_start_time) - MUCH SMALLER STEPS
            tsm.setNextTime(sim_config.loading_time + 1.0, 5, 0.01)  # First second after loading Up to 11s
            tsm.setNextTime(sim_config.loading_time + 2.5, 7, 0.02)  # Intermediate stabilization Up to 12.5s
            tsm.setNextTime(sim_config.temp_start_time - 2.0, 5, 0.05)  # Pre-thermal Up to 18s
            tsm.setNextTime(sim_config.temp_start_time, 2, 0.05)  # Just before thermal Up to 20s

            # Phase 3: Thermal loading (temp_start_time to temp_end_time) - adaptive steps
            thermal_duration = sim_config.temp_end_time - sim_config.temp_start_time
            tsm.setNextTime(sim_config.temp_start_time + thermal_duration * 0.1, 3, 0.05)  # Thermal start Up to 21.5s
            tsm.setNextTime(sim_config.temp_start_time + thermal_duration * 0.5, 6, 0.05)  # Mid thermal Up to 27.5s
            tsm.setNextTime(sim_config.temp_end_time, 18, 0.02)  # End thermal - finest steps Up to 35s

            # Phase 4: Final phase - can use larger steps again
            if sim_config.temp_end_time < sim_config.final_time:
                tsm.setNextTime(sim_config.final_time, 5, 0.1)

        else:
            # Pure mechanical - simpler time stepping
            tsm.setNextTime(sim_config.loading_time * 0.5, 13, 0.01) # Up to 5s
            tsm.setNextTime(sim_config.loading_time, 25, 0.005) # Up to 10s
            tsm.setNextTime(sim_config.final_time, 20, 0.05) # Up to 30s

    else:
        print(f"INFO: Using FIXED time stepping with dt = {sim_config.fixed_timestep_size}.")
        tsm.setInitialTime(0.0, sim_config.fixed_initial_timestep)

        # Single fixed time step for entire simulation
        tsm.setNextTime(sim_config.final_time, 20, sim_config.fixed_timestep_size)

    # TOLERANCE MANAGEMENT - ADAPTIVE for different phases
    fct_TOL = PieceWiseLinearFunction()
    fct_TOL.setData(0.0, 1.0)  # Standard tolerance for mechanical
    fct_TOL.setData(sim_config.loading_time * 0.75, 1.0)  # Tighter end of loading
    fct_TOL.setData(sim_config.loading_time, 1.0)  # Strict for transition
    fct_TOL.setData(sim_config.loading_time + 1.0, 0.1)  # Tres strict stabilisation
    fct_TOL.setData(sim_config.loading_time + 2.5, 0.1)  # Maximum stricte 7.5-12s
    fct_TOL.setData(sim_config.temp_start_time - 1.0, 0.1)  # Relachement progressif
    fct_TOL.setData(sim_config.temp_start_time, 0.1)  # Standard pour thermal

    if sim_config.enable_thermal:
        # Stricter tolerance during thermal phase due to large temperature gradients
        fct_TOL.setData(sim_config.temp_start_time + (sim_config.temp_end_time - sim_config.temp_start_time) * 0.5,
                        0.2)  # Tightest mid-thermal
        fct_TOL.setData(sim_config.temp_end_time, 0.5)  # Relax after thermal
        fct_TOL.setData(sim_config.final_time, 1.0)  # Standard for final phase
    else:
        fct_TOL.setData(T, 1.0)

    # MECHANICAL ITERATION MANAGER - OPTIMIZED with compatibility management
    mim = metafor.getMechanicalIterationManager()
    mim.setResidualComputationMethod(Method4ResidualComputation()) #----------------------------------------------Line added
    # Stricter parameters for plasticity
    # ENHANCED CONVERGENCE CONTROL for critical phases
    fct_MaxIter = PieceWiseLinearFunction()
    fct_MaxIter.setData(0.0, 25.0)  # Standard for the start
    fct_MaxIter.setData(sim_config.loading_time * 0.75, 35.0) # More iterations until the end loading
    fct_MaxIter.setData(sim_config.loading_time, 40.0) # Max for transition
    fct_MaxIter.setData(sim_config.loading_time + 2.5, 35.0) # Critical stabilization
    fct_MaxIter.setData(sim_config.temp_start_time, 30.0) # Normal return before thermal
    fct_MaxIter.setData(sim_config.temp_end_time, 40.0) # Max pendant thermal
    fct_MaxIter.setData(sim_config.final_time, 25.0)  # Standard until the end


    print("Using fixed iteration control")
    mim.setMaxNbOfIterations(20)  # Fallback fixe plus eleve
    mim.setResidualTolerance(1e-4, fct_TOL)
    mim.setPredictorComputationMethod(EXTRAPOLATION_MRUA)  # Better prediction


    # History curves (Save data in .txt files) - OPTIMIZED for thermal analysis
    interactionset = domain.getInteractionSet()
    if not p['postpro']:
        hcurves = metafor.getValuesManager()
        extractor_count = 0  # Total number of extractors effectively added

        # === Time and force monitoring ===
        hcurves.add(1, MiscValueExtractor(metafor, EXT_T), 'time');
        extractor_count += 1
        hcurves.add(2, NormalForceValueExtractor(ci), SumOperator(), 'ContactForceY');
        extractor_count += 1

        # === Displacements and external forces ===
        hcurves.add(3, DbNodalValueExtractor(p15, Field1D(TY, RE)), SumOperator(), 'displacement_rod_end_Y');
        extractor_count += 1
        hcurves.add(4, DbNodalValueExtractor(c3, Field1D(TY, GF1)), SumOperator(), 'forceYExtClampinPt');
        extractor_count += 1
        hcurves.add(5, DbNodalValueExtractor(c3, Field1D(TX, GF1)), SumOperator(), 'forceXExtClampinPt');
        extractor_count += 1
        hcurves.add(6, DbNodalValueExtractor(p17, Field1D(TY, GF1)), SumOperator(), 'forceYHinge');
        extractor_count += 1
        hcurves.add(7, DbNodalValueExtractor(p17, Field1D(TX, GF1)), SumOperator(), 'forceXHinge');
        extractor_count += 1

        # === Moment at clamping point ===
        ext0 = MomentValueExtractor(c3, pa3, TZ, GF1)
        hcurves.add(8, ext0, SumOperator(), 'MomentExtClampingPt');
        extractor_count += 1

        # === Von Mises stress in blade ===
        ext_becu = IFNodalValueExtractor(interactionset(1), IF_EVMS)
        hcurves.add(9, ext_becu, MaxOperator(), 'Max_VonMises_BeCu');
        extractor_count += 1

        # === Mass corner displacements (Y) ===
        hcurves.add(10, DbNodalValueExtractor(p10, Field1D(TY, RE)), SumOperator(), 'dispY_Bottom_left_mass');
        extractor_count += 1
        hcurves.add(11, DbNodalValueExtractor(p11, Field1D(TY, RE)), SumOperator(), 'dispY_Bottom_right_mass');
        extractor_count += 1
        hcurves.add(12, DbNodalValueExtractor(p12, Field1D(TY, RE)), SumOperator(), 'dispY_Top_right_mass');
        extractor_count += 1
        hcurves.add(13, DbNodalValueExtractor(p9, Field1D(TY, RE)), SumOperator(), 'dispY_Top_left_mass');
        extractor_count += 1

        # === Rod end X-displacement (thermal reference) ===
        hcurves.add(14, DbNodalValueExtractor(p15, Field1D(TX, RE)), SumOperator(), 'dispX_rod_end');
        extractor_count += 1

        # === Plasticity extractors ===
        if sim_config.enable_plasticityData:
            # --- Plastic strain monitoring ---
            hcurves.add(15, IFNodalValueExtractor(interactionset(1), IF_EPL), MaxOperator(),
                        'max_plastic_strain_blade');
            extractor_count += 1
            hcurves.add(16, IFNodalValueExtractor(interactionset(1), IF_EPL), MeanOperator(),
                        'mean_plastic_strain_blade');
            extractor_count += 1
            hcurves.add(17, IFNodalValueExtractor(interactionset(2), IF_EPL), MaxOperator(),
                        'max_plastic_strain_structure');
            extractor_count += 1

            # --- Plastic strain rate (DEPL assumed to track rate proxy here) ---
            hcurves.add(18, IFNodalValueExtractor(interactionset(1), IF_DEPL), MaxOperator(),
                        'max_plastic_strain_rate_blade');
            extractor_count += 1
            hcurves.add(19, IFNodalValueExtractor(interactionset(1), IF_DEPL), MeanOperator(),
                        'mean_plastic_strain_rate_blade');
            extractor_count += 1

            # --- Yield criterion and stress ---
            hcurves.add(20, IFNodalValueExtractor(interactionset(1), IF_CRITERION), MaxOperator(),
                        'max_yield_function_blade');
            extractor_count += 1
            hcurves.add(21, IFNodalValueExtractor(interactionset(1), IF_CRITERION), MeanOperator(),
                        'mean_yield_function_blade');
            extractor_count += 1
            hcurves.add(22, IFNodalValueExtractor(interactionset(1), IF_YIELD_STRESS), MaxOperator(),
                        'max_yield_stress_blade');
            extractor_count += 1
            hcurves.add(23, IFNodalValueExtractor(interactionset(1), IF_YIELD_STRESS), MeanOperator(),
                        'mean_yield_stress_blade');
            extractor_count += 1

            # --- Von Mises (mean) ---
            hcurves.add(24, IFNodalValueExtractor(interactionset(1), IF_EVMS), MeanOperator(), 'mean_VonMises_blade');
            extractor_count += 1

            # === Stress analysis ===
            # --- Cartesian stress components ---
            hcurves.add(25, IFNodalValueExtractor(interactionset(1), IF_SIG_XX), MaxOperator(), 'max_stress_xx_blade');
            extractor_count += 1
            hcurves.add(26, IFNodalValueExtractor(interactionset(1), IF_SIG_YY), MaxOperator(), 'max_stress_yy_blade');
            extractor_count += 1
            hcurves.add(27, IFNodalValueExtractor(interactionset(1), IF_SIG_XY), MaxOperator(), 'max_stress_xy_blade');
            extractor_count += 1

            # --- Principal stresses ---
            hcurves.add(28, IFNodalValueExtractor(interactionset(1), IF_SIG_1), MaxOperator(),
                        'max_principal_stress_1_blade');
            extractor_count += 1
            hcurves.add(29, IFNodalValueExtractor(interactionset(1), IF_SIG_2), MaxOperator(),
                        'max_principal_stress_2_blade');
            extractor_count += 1
            hcurves.add(30, IFNodalValueExtractor(interactionset(1), IF_SIG_3), MaxOperator(),
                        'max_principal_stress_3_blade');
            extractor_count += 1

            # --- Hydrostatic pressure ---
            hcurves.add(31, IFNodalValueExtractor(interactionset(1), IF_P), MaxOperator(), 'max_pressure_blade');
            extractor_count += 1
            hcurves.add(32, IFNodalValueExtractor(interactionset(1), IF_P), MeanOperator(), 'mean_pressure_blade');
            extractor_count += 1

            # --- Stress triaxiality ---
            hcurves.add(33, IFNodalValueExtractor(interactionset(1), IF_TRIAX), MaxOperator(), 'max_triaxiality_blade');
            extractor_count += 1
            hcurves.add(34, IFNodalValueExtractor(interactionset(1), IF_TRIAX), MeanOperator(),
                        'mean_triaxiality_blade');
            extractor_count += 1

        # === Thermal extractors ===
        if sim_config.enable_thermal:
            # --- Temperature monitoring ---
            hcurves.add(35, DbNodalValueExtractor(s1, Field1D(TO, RE)), MeanOperator(), 'temp_mean_blade_K');
            extractor_count += 1
            hcurves.add(36, DbNodalValueExtractor(s1, Field1D(TO, RE)), MaxOperator(), 'temp_max_blade_K');
            extractor_count += 1
            hcurves.add(37, DbNodalValueExtractor(s1, Field1D(TO, RE)), MinOperator(), 'temp_min_blade_K');
            extractor_count += 1

            # --- Thermal strain (material expansion) ---
            hcurves.add(38, IFNodalValueExtractor(interactionset(1), IF_THERMAL_STRAIN), MeanOperator(),
                        'thermal_strain_mean_blade');
            extractor_count += 1
            hcurves.add(39, IFNodalValueExtractor(interactionset(1), IF_THERMAL_STRAIN), MaxOperator(),
                        'thermal_strain_max_blade');
            extractor_count += 1

            # --- Thermal-induced displacements (mass top corner) ---
            hcurves.add(40, DbNodalValueExtractor(p12, Field1D(TX, RE)), SumOperator(), 'thermal_dispX_mass_top_right');
            extractor_count += 1
            hcurves.add(41, DbNodalValueExtractor(p12, Field1D(TY, RE)), SumOperator(), 'thermal_dispY_mass_top_right');
            extractor_count += 1

            # --- Blade tip temperature and displacement ---
            hcurves.add(42, DbNodalValueExtractor(p2, Field1D(TO, RE)), SumOperator(), 'temp_blade_tip_K');
            extractor_count += 1
            hcurves.add(43, DbNodalValueExtractor(p2, Field1D(TX, RE)), SumOperator(), 'dispX_blade_tip');
            extractor_count += 1
            hcurves.add(44, DbNodalValueExtractor(p2, Field1D(TY, RE)), SumOperator(), 'dispY_blade_tip');
            extractor_count += 1

        # === Final check: verify all extractors that were added ===
        for i in range(1, extractor_count + 1):
            metafor.getTestSuiteChecker().checkExtractor(i)

    # REAL-TIME PLOTTING - OPTIMIZED for thermal monitoring
    if not p['postpro']:
        try:
            # === Plot 1: Mass displacements (always shown) ===
            plot1 = DataCurveSet()
            plot1.add(VectorDataCurve(3, hcurves.getDataVector(1), hcurves.getDataVector(3), 'Rod End Y'))
            plot1.add(VectorDataCurve(10, hcurves.getDataVector(1), hcurves.getDataVector(10), 'Mass Bottom Left Y'))
            plot1.add(VectorDataCurve(11, hcurves.getDataVector(1), hcurves.getDataVector(11), 'Mass Bottom Right Y'))
            plot1.add(VectorDataCurve(12, hcurves.getDataVector(1), hcurves.getDataVector(12), 'Mass Top Right Y'))
            plot1.add(VectorDataCurve(13, hcurves.getDataVector(1), hcurves.getDataVector(13), 'Mass Top Left Y'))

            win1 = VizWin()
            win1.add(plot1)
            win1.setPlotTitle("Mass Position Evolution - Mechanical + Thermal Effects")
            win1.setPlotXLabel("Time [s]")
            win1.setPlotYLabel("Y Displacement [mm]")
            metafor.addObserver(win1)

            # === Plasticity-related plots ===
            if sim_config.enable_plasticityData:

                # --- Plot 2: Plastic strain evolution ---
                plot2 = DataCurveSet()
                plot2.add(
                    VectorDataCurve(15, hcurves.getDataVector(1), hcurves.getDataVector(15), 'Max Plastic Strain'))
                plot2.add(
                    VectorDataCurve(16, hcurves.getDataVector(1), hcurves.getDataVector(16), 'Mean Plastic Strain'))
                plot2.add(
                    VectorDataCurve(18, hcurves.getDataVector(1), hcurves.getDataVector(18), 'Max Plastic Strain Rate'))

                win2 = VizWin()
                win2.add(plot2)
                win2.setPlotTitle("Plastic Strain Evolution - Material Nonlinearity")
                win2.setPlotXLabel("Time [s]")
                win2.setPlotYLabel("Plastic Strain [-]")
                metafor.addObserver(win2)

                # --- Plot 3: Stress vs Yield comparison ---
                plot3 = DataCurveSet()
                plot3.add(VectorDataCurve(9, hcurves.getDataVector(1), hcurves.getDataVector(9), 'Max Von Mises'))
                plot3.add(VectorDataCurve(24, hcurves.getDataVector(1), hcurves.getDataVector(24), 'Mean Von Mises'))
                plot3.add(VectorDataCurve(23, hcurves.getDataVector(1), hcurves.getDataVector(23), 'Mean Yield Stress'))
                plot3.add(
                    VectorDataCurve(21, hcurves.getDataVector(1), hcurves.getDataVector(21), 'Mean Yield Function'))

                if blade_config.material == 'INVAR':
                    elastic_limit = 250.0  # MPa
                elif blade_config.material == 'INVAR_CW_HARD':
                    elastic_limit = 650.0  # MPa
                elif blade_config.material == 'BE_CU':
                    elastic_limit = 1000.0  # MPa

                win3 = VizWin()
                win3.add(plot3)
                win3.setPlotTitle(
                    f"Stress Evolution vs Yield ({blade_config.material} - Initial Elastic Limit: {elastic_limit} MPa)")
                win3.setPlotXLabel("Time [s]")
                win3.setPlotYLabel("Stress [MPa]")
                metafor.addObserver(win3)

                # --- Plot 4: Stress components evolution ---
                plot4 = DataCurveSet()
                plot4.add(VectorDataCurve(25, hcurves.getDataVector(1), hcurves.getDataVector(25), 'Max σ_xx'))
                plot4.add(VectorDataCurve(26, hcurves.getDataVector(1), hcurves.getDataVector(26), 'Max σ_yy'))
                plot4.add(VectorDataCurve(27, hcurves.getDataVector(1), hcurves.getDataVector(27), 'Max σ_xy'))
                plot4.add(
                    VectorDataCurve(28, hcurves.getDataVector(1), hcurves.getDataVector(28), 'Max σ_1 (Principal)'))

                win4 = VizWin()
                win4.add(plot4)
                win4.setPlotTitle("Stress Components Evolution")
                win4.setPlotXLabel("Time [s]")
                win4.setPlotYLabel("Stress [MPa]")
                metafor.addObserver(win4)

                # --- Plot 5: Pressure evolution ---
                plot5 = DataCurveSet()
                plot5.add(VectorDataCurve(31, hcurves.getDataVector(1), hcurves.getDataVector(31), 'Max Pressure'))
                plot5.add(VectorDataCurve(32, hcurves.getDataVector(1), hcurves.getDataVector(32), 'Mean Pressure'))

                win5 = VizWin()
                win5.add(plot5)
                win5.setPlotTitle("Pressure Evolution")
                win5.setPlotXLabel("Time [s]")
                win5.setPlotYLabel("Pressure [MPa]")
                metafor.addObserver(win5)

                # --- Plot 6: Triaxiality evolution ---
                plot6 = DataCurveSet()
                plot6.add(VectorDataCurve(33, hcurves.getDataVector(1), hcurves.getDataVector(33), 'Max Triaxiality'))
                plot6.add(VectorDataCurve(34, hcurves.getDataVector(1), hcurves.getDataVector(34), 'Mean Triaxiality'))

                win6 = VizWin()
                win6.add(plot6)
                win6.setPlotTitle("Triaxiality Evolution")
                win6.setPlotXLabel("Time [s]")
                win6.setPlotYLabel("Triaxiality [-]")
                metafor.addObserver(win6)

                # --- Plot 7: Plasticity onset detection ---
                plot7 = DataCurveSet()
                plot7.add(
                    VectorDataCurve(20, hcurves.getDataVector(1), hcurves.getDataVector(20), 'Max Yield Function'))
                plot7.add(
                    VectorDataCurve(21, hcurves.getDataVector(1), hcurves.getDataVector(21), 'Mean Yield Function'))
                plot7.add(VectorDataCurve(19, hcurves.getDataVector(1), hcurves.getDataVector(19),
                                          'Mean Plastic Strain Rate'))

                win7 = VizWin()
                win7.add(plot7)
                win7.setPlotTitle("Plasticity Onset Monitoring - Criterion Function")
                win7.setPlotXLabel("Time [s]")
                win7.setPlotYLabel("Criterion Value / Plastic Strain Rate [-]")
                metafor.addObserver(win7)

            # === Thermal-related plots ===
            if sim_config.enable_thermal:
                # --- Plot 8: Temperature evolution in blade ---
                plot8 = DataCurveSet()
                plot8.add(VectorDataCurve(35, hcurves.getDataVector(1), hcurves.getDataVector(35), 'Mean Temp Blade'))
                plot8.add(VectorDataCurve(36, hcurves.getDataVector(1), hcurves.getDataVector(36), 'Max Temp Blade'))
                plot8.add(VectorDataCurve(37, hcurves.getDataVector(1), hcurves.getDataVector(37), 'Min Temp Blade'))
                plot8.add(VectorDataCurve(42, hcurves.getDataVector(1), hcurves.getDataVector(42), 'Blade Tip Temp'))

                win8 = VizWin()
                win8.add(plot8)
                win8.setPlotTitle("Temperature Evolution - Be-Cu Blade (10°C → 50°C)")
                win8.setPlotXLabel("Time [s]")
                win8.setPlotYLabel("Temperature [K]")
                metafor.addObserver(win8)

                # --- Plot 9: Thermal-induced displacements ---
                plot9 = DataCurveSet()
                plot9.add(
                    VectorDataCurve(40, hcurves.getDataVector(1), hcurves.getDataVector(40), 'Mass Thermal Disp X'))
                plot9.add(
                    VectorDataCurve(41, hcurves.getDataVector(1), hcurves.getDataVector(41), 'Mass Thermal Disp Y'))
                plot9.add(VectorDataCurve(14, hcurves.getDataVector(1), hcurves.getDataVector(14), 'Rod End Disp X'))

                win9 = VizWin()
                win9.add(plot9)
                win9.setPlotTitle("Thermal-Induced Displacements - Sensor Response")
                win9.setPlotXLabel("Time [s]")
                win9.setPlotYLabel("Displacement [mm]")
                metafor.addObserver(win9)

                # --- Plot 10: Thermal strain evolution ---
                plot10 = DataCurveSet()
                plot10.add(
                    VectorDataCurve(38, hcurves.getDataVector(1), hcurves.getDataVector(38), 'Mean Thermal Strain'))
                plot10.add(
                    VectorDataCurve(39, hcurves.getDataVector(1), hcurves.getDataVector(39), 'Max Thermal Strain'))

                win10 = VizWin()
                win10.add(plot10)
                win10.setPlotTitle("Thermal Strain Evolution - Material Response")
                win10.setPlotXLabel("Time [s]")
                win10.setPlotYLabel("Thermal Strain [-]")
                metafor.addObserver(win10)

                # --- Plot 11: Rod end Y displacement vs temperature ---
                plot11 = DataCurveSet()
                plot11.add(
                    VectorDataCurve(14, hcurves.getDataVector(35), hcurves.getDataVector(3), 'Rod End Y vs Temp'))

                win11 = VizWin()
                win11.add(plot11)
                win11.setPlotTitle("Rod End Y Displacement vs Temperature")
                win11.setPlotXLabel("Mean Temperature [K]")
                win11.setPlotYLabel("Rod End Disp Y [mm]")
                metafor.addObserver(win11)

                # --- Plot 12: Thermomechanical coupling ---
                if sim_config.enable_plasticityData:
                    plot12 = DataCurveSet()
                    plot12.add(VectorDataCurve(15, hcurves.getDataVector(35), hcurves.getDataVector(15),
                                               'Max Plastic Strain vs Temperature'))
                    plot12.add(VectorDataCurve(9, hcurves.getDataVector(35), hcurves.getDataVector(9),
                                               'Von Mises vs Temperature'))

                    win12 = VizWin()
                    win12.add(plot12)
                    win12.setPlotTitle("Thermomechanical Coupling - Temperature Effects on Plasticity")
                    win12.setPlotXLabel("Mean Temperature [K]")
                    win12.setPlotYLabel("Plastic Strain [-] / Von Mises [MPa]")
                    metafor.addObserver(win12)

                # --- Plot 13: Blade tip displacement vs time ---
                plot13 = DataCurveSet()
                plot13.add(VectorDataCurve(43, hcurves.getDataVector(1), hcurves.getDataVector(43), 'Blade Tip Disp X'))
                plot13.add(VectorDataCurve(44, hcurves.getDataVector(1), hcurves.getDataVector(44), 'Blade Tip Disp Y'))

                win13 = VizWin()
                win13.add(plot13)
                win13.setPlotTitle("Blade Tip Displacement Evolution")
                win13.setPlotXLabel("Time [s]")
                win13.setPlotYLabel("Displacement [mm]")
                metafor.addObserver(win13)

        except NameError:
            print("Warning: Visualization not available")
            pass

    return metafor

# ----------------------------------------------------------------------------------
# OPTIMIZED Modal analysis with thermal effects consideration
from toolbox.utilities import *

def postpro():
    """
    Enhanced post-processing for thermal-mechanical analysis
    """
    import os.path
    # Working directory
    setDir('workspace/%s' % os.path.splitext(os.path.basename(__file__))[0])
    load(__name__)

    p = {}
    p['postpro'] = True
    metafor = instance(p)
    domain = metafor.getDomain()

    # This loads the last available state WITHOUT forcing continuous recording
    loader = fac.FacManager(metafor)
    loader.load()  # No parameters = loads last available state

    print('\n=== CORRECTED MODAL ANALYSIS AFTER THERMAL LOADING ===')

    # Enhanced frequency analysis on thermally loaded state
    curves = metafor.getValuesManager()

    lanczos = LanczosFrequencyAnalysisMethod(domain)
    lanczos.setNumberOfEigenValues(8)
    lanczos.setSpectralShifting(0.0)
    lanczos.setComputeEigenVectors(True)
    lanczos.setWriteMatrix2Matlab(True)

    # Remove unsupported methods - these don't exist in Metafor API:
    # lanczos.setTolerance(1e-12)        # <- This was causing the error
    # lanczos.setMaxIterations(1000)     # <- This might not exist either

    print('Computing eigenvalues on thermally deformed structure...')
    print('(Analysis performed on final thermal state)')

    try:
        lanczos.execute()
        lanczos.writeTSC()

        # Frequency extraction
        curves.add(11, FrequencyAnalysisValueExtractor(lanczos), 'freqs_thermal_corrected')

        print('\n--- EIGENVALUE RESULTS (Direct extraction) ---')
        eigenvalues = []
        frequencies_hz = []

        for i in range(min(8, lanczos.getNumberOfEigenValues())):
            eigenval = lanczos.getEigenValue(i)

            if eigenval > 0:
                # eigenval from Metafor is actually frequency in Hz, not eigenvalue
                frequency_hz = eigenval  # This is the frequency
                frequencies_hz.append(frequency_hz)

                # Calculate the TRUE eigenvalue: ω² = (2πf)²
                true_eigenvalue = (2 * math.pi * frequency_hz) ** 2
                eigenvalues.append(true_eigenvalue)

                print(f'Mode {i + 1}: Frequency = {frequency_hz:.4f} Hz, True Eigenvalue = {true_eigenvalue:.8e}')

                # Display
                lanczos.showEigenVector(i)

                if i == 0:
                    expected_freq = 2.91
                    error_percent = abs(frequency_hz - expected_freq) / expected_freq * 100
                    print(f'  Expected: ~{expected_freq:.2f} Hz, Error: {error_percent:.2f}%')

                    if error_percent > 20:
                        print(f'  *** WARNING: Large frequency deviation! ***')
                    elif error_percent > 5:
                        print(f'  *** CAUTION: Moderate frequency deviation ***')
                    else:
                        print(f'  *** GOOD: Frequency within expected range ***')
            else:
                print(f'Mode {i + 1}: Invalid eigenvalue = {eigenval:.6e}')

        # Save and verify results
        try:
            # Fill curves with current results
            curves.fillNow(metafor.getCurrentStepNo())
            curves.toAscii()
            curves.flush()

            # Read generated file for verification
            with open('freqs_thermal_corrected.ascii') as f:
                txt = f.readlines()
            freq_values = [float(v) for v in txt[0].strip().split()]

            print(f'\n--- FILE VERIFICATION ---')
            print(f'Saved frequencies = {[f"{v:.4f} Hz" for v in freq_values[:5]]}')
            # Calculate true eigenvalues from saved frequencies
            true_eigenvals = [(2 * math.pi * val) ** 2 for val in freq_values[:5] if val > 0]
            print(f'True eigenvalues = {[f"{v:.6e}" for v in true_eigenvals]}')

        except Exception as e:
            print(f"File operations failed (non-critical): {e}")

        # Summary
        if frequencies_hz:
            print(f'\n=== THERMAL EFFECT SUMMARY ===')
            print(f'First natural frequency: {frequencies_hz[0]:.4f} Hz')
            print(f'Expected baseline: ~2.91 Hz')

            if len(frequencies_hz) > 1:
                print(f'Higher modes: {[f"{f:.3f}" for f in frequencies_hz[1:5]]} Hz')

            print(f'Total modes analyzed: {len(frequencies_hz)}')

    except Exception as e:
        print(f'Modal analysis failed: {e}')
        print('Possible causes:')
        print('1. Thermal state not properly converged')
        print('2. Excessive deformation affecting stiffness matrix')
        print('3. Numerical conditioning issues')
        print('4. Missing math import (add: import math at the top)')

# Alternative simpler version (closer to your working code):
def postpro_simple():
    """
    Simplified version similar to your working code but with thermal state loading
    """
    import os.path
    import math  # Make sure math is imported

    setDir('workspace/%s' % os.path.splitext(os.path.basename(__file__))[0])
    load(__name__)

    p = {}
    p['postpro'] = True
    metafor = instance(p)
    domain = metafor.getDomain()

    # Load the last archive (thermal state)
    loader = fac.FacManager(metafor)
    loader.load()

    # Set up curves
    curves = metafor.getValuesManager()

    # Configure Lanczos analysis
    lanczos = LanczosFrequencyAnalysisMethod(domain)
    lanczos.setNumberOfEigenValues(8)  # More modes for better analysis
    lanczos.setSpectralShifting(0.0)
    lanczos.setComputeEigenVectors(True)
    lanczos.setWriteMatrix2Matlab(True)

    print('\n=== Modal Analysis on Thermal State ===')
    print('Computing Eigenvalues...')

    # Execute analysis
    lanczos.execute()
    lanczos.writeTSC()

    # Add to curves
    curves.add(11, FrequencyAnalysisValueExtractor(lanczos), 'freqs_thermal')

    # Fill and save
    curves.fillNow(metafor.getCurrentStepNo())
    curves.toAscii()
    curves.flush()

    # Display results
    print('\n--- Results ---')
    for i in range(min(8, lanczos.getNumberOfEigenValues())):
        eigenval = lanczos.getEigenValue(i)
        if eigenval > 0:
            frequency_hz = eigenval  # eigenval is actually frequency in Hz
            true_eigenvalue = (2 * math.pi * frequency_hz) ** 2  # Calculate true eigenvalue
            print(f'Mode {i + 1}: Frequency = {frequency_hz:.4f} Hz, True Eigenvalue = {true_eigenvalue:.6e}')

            if i == 0:  # First mode comparison
                expected = 2.91
                error = abs(frequency_hz - expected) / expected * 100
                print(f'  (Expected ~{expected} Hz, Error: {error:.1f}%)')
        else:
            print(f'Mode {i + 1}: Invalid eigenvalue = {eigenval:.6e}')

    # Read and verify saved file
    try:
        with open('freqs_thermal.ascii') as f:
            txt = f.readlines()
        frequencies = [float(v) for v in txt[0].strip().split()]
        true_eigenvalues = [(2 * math.pi * f) ** 2 for f in frequencies]
        print(f'\nSaved frequencies: {[f"{f:.4f} Hz" for f in frequencies[:5]]}')
        print(f'True eigenvalues: {[f"{ev:.6e}" for ev in true_eigenvalues[:5]]}')
    except Exception as e:
        print(f'Could not read output file: {e}')

# Visualization version (if you want to see mode shapes)
def postpro_with_viz():
    """
    Version with visualization like your original working code
    """
    import os.path
    import math

    setDir('workspace/%s' % os.path.splitext(os.path.basename(__file__))[0])
    load(__name__)

    p = {}
    p['postpro'] = True
    metafor = instance(p)
    domain = metafor.getDomain()

    # Load thermal state
    loader = fac.FacManager(metafor)
    loader.load()

    curves = metafor.getValuesManager()

    lanczos = LanczosFrequencyAnalysisMethod(domain)
    lanczos.setNumberOfEigenValues(8)
    lanczos.setSpectralShifting(0.0)
    lanczos.setComputeEigenVectors(True)
    lanczos.setWriteMatrix2Matlab(True)

    lanczos.execute()
    lanczos.writeTSC()
    curves.add(11, FrequencyAnalysisValueExtractor(lanczos), 'freqs_thermal_viz')

    curves.fillNow(metafor.getCurrentStepNo())
    curves.toAscii()
    curves.flush()

    # Visualization setup
    win = VizWin()
    for i in range(domain.getInteractionSet().size()):
        win.add(domain.getInteractionSet().getInteraction(i))

    # Show each mode
    for i in range(min(3, lanczos.getNumberOfEigenValues())):
        eigenval = lanczos.getEigenValue(i)
        if eigenval > 0:
            frequency_hz = eigenval  # eigenval is actually frequency in Hz
            true_eigenvalue = (2 * math.pi * frequency_hz) ** 2  # Calculate true eigenvalue
            lanczos.showEigenVector(i)
            win.update()
            print(f'Eigen Vector {i}, Frequency = {frequency_hz:.4f} Hz, True Eigenvalue = {true_eigenvalue:.6e}')
            input("Press enter to continue to next mode...")
        else:
            print(f'Skipping mode {i}: invalid eigenvalue = {eigenval:.6e}')

    # Final results
    with open('freqs_thermal_viz.ascii') as f:
        txt = f.readlines()
    frequencies = [float(v) for v in txt[0].strip().split()]
    true_eigenvalues = [(2 * math.pi * f) ** 2 for f in frequencies]
    print(f'\nFinal frequencies = {frequencies[:5]}')
    print(f'Final true eigenvalues = {true_eigenvalues[:5]}')

def postpro_initial():
    import os.path
    setDir('workspace/%s' % os.path.splitext(os.path.basename(__file__))[0])
    load(__name__)

    p = {}
    p['postpro'] = True
    metafor = instance(p)
    domain = metafor.getDomain()

    # set new curves
    curves = metafor.getValuesManager()

    lanczos = LanczosFrequencyAnalysisMethod(domain)
    lanczos.setNumberOfEigenValues(3)
    lanczos.setSpectralShifting(0.0)
    lanczos.setComputeEigenVectors(True)
    lanczos.setWriteMatrix2Matlab(True)

    # load the last archive
    loader = fac.FacManager(metafor)
    loader.load()

    lanczos.execute()
    lanczos.writeTSC()
    fExtr = FrequencyAnalysisValueExtractor(lanczos)
    curves.add(11, FrequencyAnalysisValueExtractor(lanczos), 'freqs')

    # extraction
    print('\nComputing Eigenvalues...')
    curves.fillNow(metafor.getCurrentStepNo())
    curves.toAscii()
    curves.flush()

    win = VizWin()
    for i in range(domain.getInteractionSet().size()):
        win.add(domain.getInteractionSet().getInteraction(i))
    for i in range(3):
        lanczos.showEigenVector(i)
        win.update()
        print('Eigen Vector ', i, 'EigenValue = ', lanczos.getEigenValue(i))
        input("press enter to continue")

    with open('freqs.ascii') as f:
        txt = f.readlines()

    print(f'eigenvalues = {[float(v) for v in txt[0].strip().split()]}')


def additional_diagnostics(blade_config):
    """Additional diagnostics to verify thermal condition with comprehensive analysis"""
    import os
    import numpy as np

    print('\n=== THERMAL STATE DIAGNOSTICS ===')

    # Verification of thermal output files (original code)
    thermal_files = {
        'temp_mean_blade_K.ascii': 'Mean blade temperature',
        'thermal_strain_max_blade.ascii': 'Maximum thermal strain',
        'thermal_dispX_mass_top_right.ascii': 'Thermal displacement X',
        'thermal_dispY_mass_top_right.ascii': 'Thermal displacement Y'
    }

    for filename, description in thermal_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    values = [float(line.strip()) for line in f if line.strip()]

                if values:
                    final_value = values[-1]
                    max_value = max(abs(v) for v in values)

                    if 'temp' in filename:
                        if 'K' in filename:
                            final_celsius = final_value - 273.15
                            print(f'{description}: {final_celsius:.1f}C (final)')
                        else:
                            print(f'{description}: {final_value:.6e} (final)')
                    else:
                        print(f'{description}: {final_value:.6f} mm (final), max: {max_value:.6f} mm')

            except Exception as e:
                print(f'Could not read {filename}: {e}')
        else:
            print(f'{description}: File not found')

    print('\n--- THERMAL LOADING VERIFICATION ---')
    print('Expected effects:')
    print('- Temperature change: 10C -> 50C (40C increase)')
    print('- Thermal strain: ~40C x 17e-6/C = 6.8e-4')
    print('- Frequency shift due to thermal expansion and stiffness change')

    # ENHANCED THERMAL ANALYSIS
    print('\n=== COMPREHENSIVE THERMAL ANALYSIS ===')

    # Read time data for correlation
    time_values = []
    try:
        with open('time.ascii', 'r') as f_time:
            time_values = [float(line.strip()) for line in f_time if line.strip()]
        print(f"Time steps analyzed: {len(time_values)}")
    except:
        print("Warning: Could not read time.ascii")

    # Von Mises stress analysis
    stress_files = {
        "Be-Cu Blade": 'Max_VonMises_BeCu.ascii',
    }

    print('\n--- STRESS ANALYSIS ---')
    for material, filename in stress_files.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                values = [float(line.strip()) for line in f if line.strip()]

            if values and time_values and len(time_values) == len(values):
                max_stress = max(values)
                max_stress_time = time_values[values.index(max_stress)]
                final_stress = values[-1]

                print(f"{material}:")
                print(f"  Maximum stress: {max_stress:.2f} MPa at t = {max_stress_time:.2f} s")
                print(f"  Final stress: {final_stress:.2f} MPa")
                print(f"  Stress evolution: {values[0]:.2f} → {final_stress:.2f} MPa")
        else:
            print(f"Warning: {filename} not found")

    # Temperature analysis
    temp_files = {
        "Mean Temperature": 'temp_mean_blade_K.ascii',
        "Max Temperature": 'temp_max_blade_K.ascii',
        "Min Temperature": 'temp_min_blade_K.ascii',
        "Blade Tip Temperature": 'temp_blade_tip_K.ascii'
    }

    print('\n--- TEMPERATURE ANALYSIS ---')
    for temp_type, filename in temp_files.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                values = [float(line.strip()) for line in f if line.strip()]

            if values:
                # Convert to Celsius
                initial_C = values[0] + 273.15
                final_C = values[-1] + 273.15
                delta_T = final_C - initial_C

                print(f"{temp_type}:")
                print(f"  Initial: {initial_C:.1f}°C ({values[0]:.1f}C)")
                print(f"  Final: {final_C:.1f}°C ({values[-1]:.1f}C)")
                print(f"  Change: ΔT = {delta_T:.1f}°C")

    # Thermal strain analysis
    strain_files = {
        "Mean Thermal Strain": 'thermal_strain_mean_blade.ascii',
        "Max Thermal Strain": 'thermal_strain_max_blade.ascii'
    }

    print('\n--- THERMAL STRAIN ANALYSIS ---')
    for strain_type, filename in strain_files.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                values = [float(line.strip()) for line in f if line.strip()]

            if values:
                initial_strain = values[0]
                final_strain = values[-1]
                max_strain = max(values)

                print(f"{strain_type}:")
                print(f"  Initial: {initial_strain:.2e}")
                print(f"  Final: {final_strain:.2e}")
                print(f"  Maximum: {max_strain:.2e}")
                print(f"  Change: Δε = {final_strain - initial_strain:.2e}")

    print('\n=== THERMAL LINEARITY ANALYSIS: ROD END Y (V2) ===')

    try:
        import numpy as np

        # Read temperature, displacement, and time data
        temp_data = []
        disp_data = []
        time_data = []

        # Read temperature data (relative to 10°C)
        with open('temp_mean_blade_K.ascii', 'r') as f:
            temp_data = [float(line.strip()) + 10.0 for line in f if line.strip()]

        # Read displacement data
        with open('displacement_rod_end_Y.ascii', 'r') as f:
            disp_data = [float(line.strip()) for line in f if line.strip()]

        # Read time data
        with open('time.ascii', 'r') as f:
            time_data = [float(line.strip()) for line in f if line.strip()]

        # Check lengths
        if len(temp_data) != len(disp_data) or len(temp_data) != len(time_data):
            raise ValueError("Mismatched lengths: temp, displacement, and time must be the same.")

        # Convert to numpy arrays
        temp_array = np.array(temp_data)
        disp_array = np.array(disp_data)
        time_array = np.array(time_data)

        # Filter values between 20s and 30s
        mask_time = (time_array >= 20.0) & (time_array <= 30.0)
        temp_array = temp_array[mask_time]
        disp_array = disp_array[mask_time]
        time_array = time_array[mask_time]

        if len(temp_array) == 0:
            raise ValueError("No data points found between 20s and 30s.")

        print(f"Data points (20s–30s): {len(temp_array)}")
        print(f"Temperature range: {temp_array[0]:.1f} to {temp_array[-1]:.1f} degC")
        print(f"Displacement range: {disp_array[0]:.6f} to {disp_array[-1]:.6f} mm")

        # Define temperature intervals (10 degC wide)
        temp_intervals = [(10, 20), (20, 30), (30, 40), (40, 50)]

        print("\n--- INTERVAL ANALYSIS (10 degC each) ---")
        slopes = []
        r_squared_values = []

        for temp_min, temp_max in temp_intervals:
            mask = (temp_array >= temp_min) & (temp_array <= temp_max)
            temp_interval = temp_array[mask]
            disp_interval = disp_array[mask]

            if len(temp_interval) > 1:
                coeffs = np.polyfit(temp_interval, disp_interval, 1)
                slope = coeffs[0]
                intercept = coeffs[1]

                disp_fit = np.polyval(coeffs, temp_interval)
                ss_res = np.sum((disp_interval - disp_fit) ** 2)
                ss_tot = np.sum((disp_interval - np.mean(disp_interval)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

                slopes.append(slope)
                r_squared_values.append(r_squared)

                print(f"Interval {temp_min} to {temp_max} degC:")
                print(f"  Data points: {len(temp_interval)}")
                print(f"  Slope: {slope:.8f} mm/degC")
                print(f"  R squared: {r_squared:.6f}")
                print(f"  Predicted displacement at {temp_min} degC: {slope * temp_min + intercept:.6f} mm")
                print(f"  Predicted displacement at {temp_max} degC: {slope * temp_max + intercept:.6f} mm")
                print(f"  Total variation over interval: {slope * 10:.6f} mm\n")
            else:
                print(f"Interval {temp_min} to {temp_max} degC: Not enough data points")

        # Global linearity analysis
        print("--- GLOBAL LINEARITY ANALYSIS ---")
        if len(slopes) > 1:
            slopes_array = np.array(slopes)
            mean_slope = np.mean(slopes_array)
            std_slope = np.std(slopes_array)
            cv_slope = (abs(std_slope) / abs(mean_slope)) * 100 if mean_slope != 0 else 0

            print(f"Mean slope: {mean_slope:.8f} mm/degC")
            print(f"Slope standard deviation: {std_slope:.8f} mm/degC")
            print(f"Coefficient of variation: {cv_slope:.2f}%")

            if cv_slope < 1.0:
                linearity_assessment = "EXCELLENT"
            elif cv_slope < 5.0:
                linearity_assessment = "GOOD"
            elif cv_slope < 10.0:
                linearity_assessment = "ACCEPTABLE"
            else:
                linearity_assessment = "POOR"

            print(f"Linearity assessment: {linearity_assessment}")

            print("\n--- SLOPE COMPARISON BY INTERVAL ---")
            for i, (interval, slope) in enumerate(zip(temp_intervals[:len(slopes)], slopes)):
                deviation = ((slope - mean_slope) / mean_slope) * 100 if mean_slope != 0 else 0
                print(f"{interval[0]}-{interval[1]} degC: {slope:.8f} mm/degC (deviation: {deviation:+.2f}%)")

            global_coeffs = np.polyfit(temp_array, disp_array, 1)
            global_slope = global_coeffs[0]
            global_intercept = global_coeffs[1]

            disp_global_fit = np.polyval(global_coeffs, temp_array)
            ss_res_global = np.sum((disp_array - disp_global_fit) ** 2)
            ss_tot_global = np.sum((disp_array - np.mean(disp_array)) ** 2)
            r_squared_global = 1 - (ss_res_global / ss_tot_global) if ss_tot_global != 0 else 1.0

            print(f"\n--- GLOBAL LINEAR REGRESSION ---")
            print(f"Global slope: {global_slope:.8f} mm/degC")
            print(f"Intercept: {global_intercept:.8f} mm")
            print(f"Global R squared: {r_squared_global:.6f}")
            print(f"Predicted total displacement (40 degC): {global_slope * 40:.6f} mm")

        else:
            print("Not enough intervals for comparative analysis")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Make sure 'temp_mean_blade_K.ascii', 'displacement_rod_end_Y.ascii', and 'time.ascii' exist")
    except Exception as e:
        print(f"Error during linearity analysis: {e}")


        # ========================================= PLASTICITY ANALYSIS =====================================================
    print('\n=== PLASTICITY ANALYSIS ===')

    # Define material properties for reference
    material_properties = {
        'BE_CU': {'elastic_limit': 1000.0, 'hardening': 1000.0},
        'INVAR': {'elastic_limit': 250.0, 'hardening': 600.0},
        'INVAR_CW_HARD': {'elastic_limit': 650.0, 'hardening': 700.0}
    }

    # Determine current material (you'll need to pass this or detect it)
    current_material = blade_config.material  # Change this based on your blade_config.material
    elastic_limit = material_properties[current_material]['elastic_limit']

    print(f"Material: {current_material}")
    print(f"Elastic limit: {elastic_limit} MPa")

    # Plasticity files to analyze
    plasticity_files = {
        'Max Von Mises Stress': 'Max_VonMises_BeCu.ascii',
        'Max Plastic Strain': 'max_plastic_strain_blade.ascii',
        'Mean Plastic Strain': 'mean_plastic_strain_blade.ascii',
        'Max Plastic Strain Rate': 'max_plastic_strain_rate_blade.ascii',
        'Max Yield Function': 'max_yield_function_blade.ascii',
        'Max Yield Stress': 'max_yield_stress_blade.ascii',
        'Max Triaxiality': 'max_triaxiality_blade.ascii',
        'Max Principal Stress 1': 'max_principal_stress_1_blade.ascii',
        'Max Principal Stress 2': 'max_principal_stress_2_blade.ascii',
        'Max Principal Stress 3': 'max_principal_stress_3_blade.ascii',
        'Max Pressure': 'max_pressure_blade.ascii'
    }

    print('\n--- PLASTICITY STATE ANALYSIS ---')
    plasticity_data = {}

    for param_name, filename in plasticity_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    values = [float(line.strip()) for line in f if line.strip()]

                if values:
                    initial_val = values[0]
                    final_val = values[-1]
                    max_val = max(values)
                    min_val = min(values)

                    plasticity_data[param_name] = {
                        'initial': initial_val,
                        'final': final_val,
                        'maximum': max_val,
                        'minimum': min_val,
                        'evolution': final_val - initial_val
                    }

                    print(f"{param_name}:")
                    if 'Stress' in param_name or 'Pressure' in param_name:
                        print(f"  Initial: {initial_val:.2f} MPa")
                        print(f"  Final: {final_val:.2f} MPa")
                        print(f"  Maximum: {max_val:.2f} MPa")
                        print(f"  Evolution: {final_val - initial_val:.2f} MPa")
                    elif 'Strain' in param_name:
                        print(f"  Initial: {initial_val:.2e}")
                        print(f"  Final: {final_val:.2e}")
                        print(f"  Maximum: {max_val:.2e}")
                        print(f"  Evolution: {final_val - initial_val:.2e}")
                    else:
                        print(f"  Initial: {initial_val:.6f}")
                        print(f"  Final: {final_val:.6f}")
                        print(f"  Maximum: {max_val:.6f}")
                        print(f"  Evolution: {final_val - initial_val:.6f}")

            except Exception as e:
                print(f"Could not read {filename}: {e}")
        else:
            print(f"{param_name}: File not found ({filename})")

    # Plasticity assessment
    print('\n--- PLASTICITY ASSESSMENT ---')

    if 'Max Von Mises Stress' in plasticity_data:
        max_stress = plasticity_data['Max Von Mises Stress']['maximum']
        final_stress = plasticity_data['Max Von Mises Stress']['final']

        print(f"Maximum Von Mises stress reached: {max_stress:.2f} MPa")
        print(f"Final Von Mises stress: {final_stress:.2f} MPa")
        print(f"Elastic limit: {elastic_limit:.2f} MPa")

        if max_stress > elastic_limit:
            overstress = max_stress - elastic_limit
            overstress_percent = (overstress / elastic_limit) * 100
            print(f"*** PLASTICITY DETECTED ***")
            print(f"Overstress: {overstress:.2f} MPa ({overstress_percent:.1f}% above elastic limit)")
        else:
            safety_factor = elastic_limit / max_stress
            print(f"*** ELASTIC BEHAVIOR ***")
            print(f"Safety factor: {safety_factor:.2f}")

    if 'Max Plastic Strain' in plasticity_data:
        max_plastic_strain = plasticity_data['Max Plastic Strain']['maximum']
        final_plastic_strain = plasticity_data['Max Plastic Strain']['final']

        if max_plastic_strain > 1e-8:
            print(f"*** PLASTIC DEFORMATION CONFIRMED ***")
            print(f"Maximum plastic strain: {max_plastic_strain:.2e}")
            print(f"Final plastic strain: {final_plastic_strain:.2e}")

            # Estimate permanent deformation
            if max_plastic_strain > 1e-6:
                print(f"*** SIGNIFICANT PLASTIC DEFORMATION ***")
                print(f"This level of plastic strain may cause permanent changes")
                print(f"to the sensor's mechanical properties and calibration")
        else:
            print(f"No significant plastic strain detected")

    if 'Max Yield Function' in plasticity_data:
        max_yield_function = plasticity_data['Max Yield Function']['maximum']
        final_yield_function = plasticity_data['Max Yield Function']['final']

        print(f"Maximum yield function value: {max_yield_function:.6f}")
        print(f"Final yield function value: {final_yield_function:.6f}")

        if max_yield_function > 1e-6:
            print(f"*** YIELD FUNCTION ACTIVE (f > 0) ***")
            print(f"Material is actively yielding during simulation")
        else:
            print(f"Yield function remains inactive (f </= 0)")

    # Stress state analysis
    if all(key in plasticity_data for key in
           ['Max Principal Stress 1', 'Max Principal Stress 2', 'Max Principal Stress 3']):
        print('\n--- STRESS STATE ANALYSIS ---')
        sig1_max = plasticity_data['Max Principal Stress 1']['maximum']
        sig2_max = plasticity_data['Max Principal Stress 2']['maximum']
        sig3_max = plasticity_data['Max Principal Stress 3']['maximum']

        print(f"Maximum principal stresses:")
        print(f"  Sigma1 = {sig1_max:.2f} MPa")
        print(f"  Sigma2 = {sig2_max:.2f} MPa")
        print(f"  Sigma3 = {sig3_max:.2f} MPa")

        # Determine stress state
        if abs(sig2_max) < 0.1 * abs(sig1_max) and abs(sig3_max) < 0.1 * abs(sig1_max):
            print("Stress state: Predominantly uniaxial")
        elif abs(sig3_max) < 0.1 * max(abs(sig1_max), abs(sig2_max)):
            print("Stress state: Predominantly plane stress")
        else:
            print("Stress state: Triaxial")

    if 'Max Triaxiality' in plasticity_data:
        max_triax = plasticity_data['Max Triaxiality']['maximum']
        print(f"Maximum stress triaxiality: {max_triax:.3f}")

        if max_triax > 0.33:
            print("High triaxiality - potential for void growth")
        elif max_triax < -0.33:
            print("Compressive triaxiality - shear-dominated deformation")
        else:
            print("Moderate triaxiality - balanced stress state")

    # Correlation with temperature
    print('\n--- PLASTICITY-TEMPERATURE CORRELATION ---')

    if time_values and 'Max Von Mises Stress' in plasticity_data:
        try:
            # Read stress evolution
            with open('Max_VonMises_BeCu.ascii', 'r') as f:
                stress_values = [float(line.strip()) for line in f if line.strip()]

            if len(stress_values) == len(time_values):
                # Find when plasticity starts (stress exceeds elastic limit)
                plastic_start_indices = [i for i, stress in enumerate(stress_values) if stress > elastic_limit]

                if plastic_start_indices:
                    plastic_start_time = time_values[plastic_start_indices[0]]
                    plastic_start_stress = stress_values[plastic_start_indices[0]]

                    print(f"Plasticity initiated at:")
                    print(f"  Time: {plastic_start_time:.2f} s")
                    print(f"  Stress: {plastic_start_stress:.2f} MPa")

                    # Correlate with temperature if available
                    if os.path.exists('temp_mean_blade_K.ascii'):
                        with open('temp_mean_blade_K.ascii', 'r') as f:
                            temp_values = [float(line.strip()) for line in f if line.strip()]

                        if len(temp_values) == len(time_values):
                            delta_T = temp_values[plastic_start_indices[0]]  # in Kelvin
                            plastic_start_temp_K = delta_T + 273.15 + 10.0
                            plastic_start_temp_C = delta_T + 10.0

                            print(f"  Temperature at plasticity onset: {plastic_start_temp_C:.1f}°C")
                            print(f"  Temperature rise from start: {delta_T:.1f}°C")
                else:
                    print("No plasticity detected based on stress threshold")

        except Exception as e:
            print(f"Could not correlate plasticity with temperature: {e}")

    print('\n--- PLASTICITY IMPACT ON SENSOR ---')
    if 'Max Plastic Strain' in plasticity_data and plasticity_data['Max Plastic Strain']['maximum'] > 1e-6:
        print("*** WARNING: SIGNIFICANT PLASTICITY DETECTED ***")
        print("Potential impacts on sensor performance:")
        print(" Permanent deformation may alter resonant frequencies")
        print(" Stress relaxation could affect long-term stability")
        print(" Calibration may drift due to permanent material changes")
        print(" Consider reducing temperature range or stress levels")
        print(" Material substitution (higher yield strength) may be needed")
    else:
        print(" Plasticity levels are minimal - sensor should maintain calibration")
        print(" Elastic behavior preserved - good for long-term stability")

    # Displacement analysis - KEY for sensor performance (code original conservé)
    disp_files = {
        "Rod End Y": 'displacement_rod_end_Y.ascii',
        "Rod End X": 'dispX_rod_end.ascii',
        "Mass Top Right X": 'thermal_dispX_mass_top_right.ascii',
        "Mass Top Right Y": 'thermal_dispY_mass_top_right.ascii',
        "Blade Tip X": 'dispX_blade_tip.ascii',
        "Blade Tip Y": 'dispY_blade_tip.ascii'
    }

    print('\n--- DISPLACEMENT ANALYSIS (SENSOR RESPONSE) ---')
    for disp_type, filename in disp_files.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                values = [float(line.strip()) for line in f if line.strip()]

            if values:
                initial_disp = values[0]
                final_disp = values[-1]
                max_disp = max(values)
                thermal_effect = final_disp - initial_disp

                print(f"{disp_type}:")
                print(f"  Initial: {initial_disp:.6f} mm")
                print(f"  Final: {final_disp:.6f} mm")
                print(f"  Maximum: {max_disp:.6f} mm")
                print(f"  Thermal effect: Δ = {thermal_effect:.6f} mm")

                if abs(thermal_effect) > 1e-6:
                    print(f"  *** SIGNIFICANT THERMAL DISPLACEMENT: {thermal_effect:.3f} mm ***")

    print('\n=== SENSOR PERFORMANCE SUMMARY ===')
    print(f"This simulation demonstrates the muVINS sensor response to:")
    print(f"1. Mechanical pre-loading (blade bending)")
    print(f"2. Large temperature change (10°C → 50°C)")
    print(f"3. Resulting thermal expansion/contraction effects")
    print(f"4. Mass displacement due to thermal-mechanical coupling")
    print(f"\nThe thermal effects on mass position are critical for")
    print(f"gravitational wave detection sensitivity.")
    print('=' * 60)


if __name__ == "__main__":

    config = BladeConfig()

    # Main analysis : Only one at a time
    postpro()
    #postpro_simple() # Initial way to obtain the modal analysis
    #postpro_with_viz() # Modal analysys with graphs
    #postpro_initial() #Post pro of morgan code

    additional_diagnostics(config)

