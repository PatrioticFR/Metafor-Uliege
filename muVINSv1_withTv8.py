# TFE: Towards the development of an ultra-high performance inertial sensor for
# gravitational waves detection

# Study of the muVINS with configurable blade parameters - OPTIMIZED VERSION
# Morgane Zeoli

# -*- coding: utf-8 -*-

from wrap import *
from wrap.mtFrequencyAnalysisw import *
import math

# enable full parallelism
StrVectorBase.useTBB()
StrMatrixBase.useTBB()
ContactInteraction.useTBB()


# =============================================================================
# CONFIGURATION PARAMETERS - EASY TO MODIFY
# =============================================================================

class BladeConfig:
    """Configuration parameters for the blade"""

    def __init__(self):
        # Blade geometry
        self.thickness = 0.24  # blade thickness (e)
        self.length = 105.25  # blade length (L)
        self.width = 45.0  # blade width (thickness in 3D)

        # Material selection: 'BE_CU', 'INVAR', 'STEEL'
        self.material = 'BE_CU'

        # Mesh parameters for blade
        self.elements_thickness = 8  # elements through thickness (ne)
        self.elements_length_factor = 20  # nL = L * this factor

    def get_elements_length(self):
        """Calculate number of elements along length"""
        return int(self.length * self.elements_length_factor)


class SimulationConfig:
    """General simulation parameters - OPTIMIZED"""

    def __init__(self):
        # Time parameters - OPTIMIZED sequence
        self.final_time = 30.0  # T - Total simulation time (30s)
        self.loading_time = 10.0  # T_load - Mechanical loading up to 10s
        self.stabilization_time = 12.0  # Time when system is stabilized

        # Temperature parameters - MODIFIED FOR 10C to 50C
        self.enable_thermal = False #True or False
        self.temp_initial_kelvin = 273.15 + 10.0  # 10C initial
        self.temp_final_kelvin = 273.15 + 50.0    # 50C final
        self.temp_start_time = 20.0  # Start of thermal change (modified)
        self.temp_end_time = 35.0    # End of thermal change (modified)

        # Clamping parameters
        self.Dx = -67.5227  # reference horizontal location
        self.Dx1 = 0.0  # horizontal shift
        self.Dy = 0.0  # vertical shift
        self.angleClamp = 0.0  # clamping angle


def setup_temperature_dependent_properties(material):
    """Return temperature-dependent E and CTE functions for the given material."""

    if material == 'BE_CU':
        fctE = PieceWiseLinearFunction()
        fctE.setData(283.15, 131e3)  # 10C
        fctE.setData(290.15, 130.8e3)  # 17C
        fctE.setData(293.15, 130.7e3)  # 20C (reference)
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

    else:
        raise ValueError(f"Unsupported material: {material}")

    return fctE, fctCTE


def getMetafor(d={}):
    """
    Optimized main function with better convergence
    """

    # Initialize configurations
    blade_config = BladeConfig()
    sim_config = SimulationConfig()

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

    metafor = Metafor()
    #optimize_archiving(metafor, sim_config)
    domain = metafor.getDomain()

    # Use configuration parameters
    e = blade_config.thickness
    L = blade_config.length
    T = sim_config.final_time
    T_load = sim_config.loading_time
    Dx = sim_config.Dx
    Dx1 = sim_config.Dx1
    Dy = sim_config.Dy
    angleClamp = sim_config.angleClamp

    # Fixed rod and mass parameters
    l = 79.2  # total length
    H = 3.875  # thickness
    r = 7  # distance between mass and rod end
    R = H
    enc = 57.32  # Leaf-spring clamping point location on the rod

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

    # Geometry (unchanged from original)
    geometry = domain.getGeometry()
    geometry.setDimPlaneStrain(1.0)

    pointset = geometry.getPointSet()

    # blade
    p1 = pointset.define(1, enc, H / 2)
    p2 = pointset.define(2, enc + e, H / 2)
    p3 = pointset.define(3, enc + e, L)
    p4 = pointset.define(4, enc, L)
    # rod
    p5 = pointset.define(5, 0.0, H / 2)
    p6 = pointset.define(6, 0.0, -H / 2)
    p7 = pointset.define(7, h, -H / 2)
    p8 = pointset.define(8, h, H / 2)
    # mass
    p9 = pointset.define(9, h, D - y)
    p10 = pointset.define(10, h, -y)
    p11 = pointset.define(11, h + d, -y)
    p12 = pointset.define(12, h + d, D - y)
    # end rod
    p13 = pointset.define(13, h + d, R / 2)
    p14 = pointset.define(14, h + d, -R / 2)
    p15 = pointset.define(15, h + d + r, -R / 2)
    p16 = pointset.define(16, h + d + r, R / 2)
    # Geometry compatibility
    p17 = pointset.define(17, 0.0, 0.0)
    p18 = pointset.define(18, h, 0.0)
    p19 = pointset.define(19, h + d, 0.0)
    p20 = pointset.define(20, h + d + r, 0.0)
    p21 = pointset.define(21, enc, -H / 2)
    p22 = pointset.define(22, enc + e, -H / 2)

    # Ground
    p25 = pointset.define(25, h + d + r, -y)
    p26 = pointset.define(26, 0, -y)
    p27 = pointset.define(27, Dx + enc, 0.0)

    # Middle plane
    p28 = pointset.define(28, enc, 0.0)
    p29 = pointset.define(29, e + enc, 0.0)
    # Spring
    p30 = pointset.define(30, 0.0, -H / 2)



    # Curves and wires (same as original)
    curveset = geometry.getCurveSet()
    # blade
    c1 = curveset.add(Line(1, p1, p2))
    c2 = curveset.add(Line(2, p29, p3))
    c3 = curveset.add(Line(3, p3, p4))
    c4 = curveset.add(Line(4, p4, p28))
    # rod
    c5 = curveset.add(Line(5, p5, p17))
    c6 = curveset.add(Line(6, p17, p6))
    c7 = curveset.add(Line(7, p6, p21))
    c8 = curveset.add(Line(8, p21, p22))
    c9 = curveset.add(Line(9, p22, p7))
    c10 = curveset.add(Line(10, p7, p18))
    c11 = curveset.add(Line(11, p18, p8))
    c12 = curveset.add(Line(12, p8, p2))
    c13 = curveset.add(Line(13, p1, p5))
    # mass
    c14 = curveset.add(Line(14, p9, p8))
    c15 = curveset.add(Line(15, p7, p10))
    c16 = curveset.add(Line(16, p10, p11))
    c17 = curveset.add(Line(17, p11, p14))
    c18 = curveset.add(Line(18, p14, p19))
    c19 = curveset.add(Line(19, p19, p13))
    c20 = curveset.add(Line(20, p13, p12))
    c21 = curveset.add(Line(21, p12, p9))
    # End rod
    c22 = curveset.add(Line(22, p14, p15))
    c23 = curveset.add(Line(23, p15, p20))
    c24 = curveset.add(Line(24, p20, p16))
    c25 = curveset.add(Line(25, p16, p13))
    # Ground
    c26 = curveset.add(Line(26, p25, p26))
    # Middle plane
    c27 = curveset.add(Line(27, p17, p28))
    c28 = curveset.add(Line(28, p28, p29))
    c29 = curveset.add(Line(29, p29, p18))

    wireset = geometry.getWireSet()
    w1 = wireset.add(Wire(1, [c28, c2, c3, c4]))
    w2 = wireset.add(Wire(2, [c5, c27, c28, c29, c11, c12, c1, c13]))
    w3 = wireset.add(Wire(3, [c6, c7, c8, c9, c10, c29, c28, c27]))
    w4 = wireset.add(Wire(4, [c14, c11, c10, c15, c16, c17, c18, c19, c20, c21]))
    w5 = wireset.add(Wire(5, [c19, c18, c22, c23, c24, c25]))
    w6 = wireset.add(Wire(6, [c26]))

    sideset = geometry.getSideSet()
    s1 = sideset.add(Side(1, [w1]))
    s2 = sideset.add(Side(2, [w2]))
    s3 = sideset.add(Side(3, [w3]))
    s4 = sideset.add(Side(4, [w4]))
    s5 = sideset.add(Side(5, [w5]))
    s6 = sideset.add(Side(6, [w6]))

    # Mesh with same parameters
    prog = 5
    SimpleMesher1D(c1).execute(ne)
    SimpleMesher1D(c2).execute(nL)
    SimpleMesher1D(c3).execute(ne)
    SimpleMesher1D(c4).execute(nL)
    SimpleMesher1D(c5).execute(n56)
    SimpleMesher1D(c6).execute(n56)
    SimpleMesher1D(c7).execute(n7)
    SimpleMesher1D(c8).execute(ne)
    SimpleMesher1D(c9).execute(n9)
    SimpleMesher1D(c10).execute(n56)
    SimpleMesher1D(c11).execute(n56)
    SimpleMesher1D(c12).execute(n9)
    SimpleMesher1D(c13).execute(n7)
    SimpleMesher1D(c14).execute(n14)
    SimpleMesher1D(c15).execute(n15, 1 / prog)
    SimpleMesher1D(c16).execute(nd)
    SimpleMesher1D(c17).execute(n15, prog)
    SimpleMesher1D(c18).execute(n56)
    SimpleMesher1D(c19).execute(n56)
    SimpleMesher1D(c20).execute(n14)
    SimpleMesher1D(c21).execute(nd)
    SimpleMesher1D(c22).execute(nr)
    SimpleMesher1D(c23).execute(n56)
    SimpleMesher1D(c24).execute(n56)
    SimpleMesher1D(c25).execute(nr)
    SimpleMesher1D(c27).execute(n7)
    SimpleMesher1D(c28).execute(ne)
    SimpleMesher1D(c29).execute(n9)

    TransfiniteMesher2D(s1).execute(True)
    TransfiniteMesher2D(s2).execute2((5, (27, 28, 29), 11, (12, 1, 13)))
    TransfiniteMesher2D(s3).execute2((6, (7, 8, 9), 10, (29, 28, 27)))
    TransfiniteMesher2D(s4).execute2(((14, 11, 10, 15), 16, (17, 18, 19, 20), 21))
    TransfiniteMesher2D(s5).execute2(((19, 18), 22, (23, 24), 25))

    # OPTIMIZED MATERIALS - Temperature dependent Be-Cu
    materials = domain.getMaterialSet()
    laws = domain.getMaterialLawSet()

    # Lois de comportement élastoplastiques
    # Lois de comportement élastoplastiques selon le matériau choisi
    if blade_config.material == 'BE_CU':
        laws.define(1, LinearIsotropicHardening)
        laws(1).put(IH_SIGEL, 1000.0)  # Limite élastique approximative pour Be-Cu
        laws(1).put(IH_H, 1000.0)  # Module d'écrouissage Be-Cu
        yield_num = 1
    elif blade_config.material == 'INVAR':
        laws.define(2, LinearIsotropicHardening)
        laws(2).put(IH_SIGEL, 250.0)  # Limite élastique approximative pour Invar (~250 MPa)
        laws(2).put(IH_H, 800.0)  # Module d'écrouissage Invar (valeur modérée)
        yield_num = 2
    else:
        raise ValueError(f"Unsupported material: {blade_config.material}")

    laws.define(2, LinearIsotropicHardening)
    laws(2).put(IH_SIGEL, 400.0)  # Limite élastique pour acier
    laws(2).put(IH_H, 1000.0)  # Module d'écrouissage

    # Setup temperature-dependent functions
    fctE, fctCTE = setup_temperature_dependent_properties(blade_config.material)

    if sim_config.enable_thermal:
        # CORRECTION: Utiliser TmElastHypoMaterial au lieu de EvpIsoHHypoMaterial
        # pour pouvoir utiliser les propriétés thermiques
        #https://material-properties.org/beryllium-copper-density-strength-hardness-melting-point/#google_vignette
        materials.define(1, TmEvpIsoHHypoMaterial)
        if blade_config.material == 'BE_CU':
            materials(1).put(MASS_DENSITY, 8.36e-9)
            materials(1).put(POISSON_RATIO, 0.285)
            materials(1).put(CONDUCTIVITY, 105.0)
            materials(1).put(HEAT_CAPACITY, 420.e6)
            materials(1).put(ELASTIC_MODULUS, 131e3)  # Valeur à 10C (283.15K)
            materials(1).put(THERM_EXPANSION, 17.0e-6)  # Valeur à 10C (283.15K)
        elif blade_config.material == 'INVAR':
            materials(1).put(MASS_DENSITY, 8.1e-9)
            materials(1).put(POISSON_RATIO, 0.29)
            materials(1).put(CONDUCTIVITY, 10.0)
            materials(1).put(HEAT_CAPACITY, 500.e6)
            materials(1).put(ELASTIC_MODULUS, 141e3)  # Valeur à 10C (283.15K)
            materials(1).put(THERM_EXPANSION, 1.1e-6)  # Valeur à 10C (283.15K)

        materials(1).depend(ELASTIC_MODULUS, fctE, Field1D(TO, RE))
        materials(1).depend(THERM_EXPANSION, fctCTE, Field1D(TO, RE))
        materials(1).put(DISSIP_TE, 0.0)
        materials(1).put(DISSIP_TQ, 0.0)
        materials(1).put(YIELD_NUM, yield_num)
        # Note: Pas de YIELD_NUM et TmElastHypoMaterial si élastique uniquement : Utiliser TmEvpIsoHHypoMaterial si plastique

        # Structure material - Utiliser TmElastHypoMaterial aussi pour cohérence
        # http://metafor.ltas.ulg.ac.be/dokuwiki/doc/user/elements/volumes/iso_hypo_materials
        materials.define(2, TmEvpIsoHHypoMaterial)  # CHANGÉ ici
        materials(2).put(MASS_DENSITY, 7.93e-9)
        materials(2).put(ELASTIC_MODULUS, 210e3)
        materials(2).put(POISSON_RATIO, 0.3)
        materials(2).put(THERM_EXPANSION, 0.0)  # Pas d'expansion thermique pour la structure
        materials(2).put(CONDUCTIVITY, 50)
        materials(2).put(HEAT_CAPACITY, 500.e6)
        materials(2).put(DISSIP_TE, 0.0)
        materials(2).put(DISSIP_TQ, 0.0)
        materials(2).put(YIELD_NUM, 2)
        # Note: Pas de YIELD_NUM et TmElastHypoMaterial si élastique uniquement

    else:
        # Mechanical only - Utiliser ElastHypoMaterial si élastique simple (et enlever Yield_num)
        materials.define(1, EvpIsoHHypoMaterial)
        if blade_config.material == 'BE_CU':
            materials(1).put(MASS_DENSITY, 8.36e-9)
            materials(1).put(POISSON_RATIO, 0.285)
            materials(1).put(ELASTIC_MODULUS, 131e3)  # Valeur à 10C (283.15K)
        elif blade_config.material == 'INVAR':
            materials(1).put(MASS_DENSITY, 8.1e-9)
            materials(1).put(POISSON_RATIO, 0.29)
            materials(1).put(ELASTIC_MODULUS, 141e3)  # Valeur à 21C (283.15K)

        materials(1).put(YIELD_NUM, yield_num)


        materials.define(2, EvpIsoHHypoMaterial)
        materials(2).put(MASS_DENSITY, 7.93e-9)
        materials(2).put(ELASTIC_MODULUS, 210e3)
        materials(2).put(POISSON_RATIO, 0.3)
        materials(2).put(YIELD_NUM, 2)

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
    pa1 = pointset.define(23, enc + e / 2, L / 2)
    pa2 = pointset.define(24, enc + e / 2, L / 2, 1.0)
    axe1 = Axe(pa1, pa1)
    axe1.setSymZ1(1.0)

    pa3 = pointset.define(31, Dx + Dx1 + enc + e / 2, Dy, 0.0)
    pa4 = pointset.define(32, Dx + Dx1 + enc + e / 2, Dy, 1.0)
    axe2 = Axe(pa3, pa4)

    # OPTIMIZED LOADING FUNCTIONS - Smoother and faster convergence
    fctX = PieceWiseLinearFunction()
    fctX.setData(0.0, 0.0)
    fctX.setData(T_load / 5, 0.0)  # Delayed start for better convergence
    fctX.setData(T_load / 2, 0.5)  # Progressive loading
    fctX.setData(3 * T_load / 4, 0.8)
    fctX.setData(T_load, 1.0)
    fctX.setData(12.0, 1.0)
    fctX.setData(T, 1.0)

    fctY = PieceWiseLinearFunction()
    fctY.setData(0.0, 0.0)
    fctY.setData(T_load / 2, 0.0)
    fctY.setData(3 * T_load / 4, 1.0)
    fctY.setData(T_load, 1.0)
    fctY.setData(12.0, 1.0)
    fctY.setData(T, 1.0)

    domain.getLoadingSet().define(pa1, Field1D(TX, RE), (Dx + Dx1), fctX)
    domain.getLoadingSet().define(pa2, Field1D(TX, RE), (Dx + Dx1), fctX)
    domain.getLoadingSet().define(pa1, Field1D(TY, RE), Dy, fctY)
    domain.getLoadingSet().define(pa2, Field1D(TY, RE), Dy, fctY)

    # OPTIMIZED ROTATION FUNCTIONS - Much smoother for better convergence
    fctR = PieceWiseLinearFunction()
    fctR.setData(0.0, 0.0)
    fctR.setData(T_load / 5, 0.0)  # Delayed start
    fctR.setData(T_load / 2, 0.5)  # Progressive rotation
    fctR.setData(3 * T_load / 4, 0.8)
    fctR.setData(T_load, 1.0)
    fctR.setData(12.0, 1.0)
    fctR.setData(T, 1.0)

    fctR2 = PieceWiseLinearFunction()
    fctR2.setData(0.0, 0.0)
    fctR2.setData(T_load, 0.0)
    fctR2.setData(12.0, 1.0)
    fctR2.setData(T, 1.0)

    domain.getLoadingSet().defineRot2(c3, Field3D(TXTYTZ, RE), axe2, angleClamp, fctR2, axe1, 180, fctR, False)

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

    # Ground displacement - Delayed ground displacement after mechanical stabilization
    fctSol = PieceWiseLinearFunction()
    fctSol.setData(0.0, 0.0)
    fctSol.setData(T_load, 0.0)  # No displacement during mechanical phase
    fctSol.setData(T, 1.0)  # Displacement only at the end of simulation
    DSol = -10
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
        tiMech = AlphaGeneralizedTimeIntegration(metafor)  #----------------------------------------------Ligne change (quasistatic -> ALPHA)
        # Note: Contrôle de la convergence via les gestionnaires d'itération plutôt que setMaxNumberOfLoadIncrements

        # Thermal time integration - OPTIMIZED parameters
        tiTher = TrapezoidalThermalTimeIntegration(metafor)
        tiTher.setTheta(1.0)  # Fully implicit for stability with large temperature changes

        # Staggered integration - OPTIMIZED coupling
        ti = StaggeredTmTimeIntegration(metafor)
        ti.setIsAdiabatic(False)
        ti.setWithStressReevaluation(True)  # IMPORTANT: Re-evaluate stress after thermal at the expense of time
        # If put to false, comment the setCouplingTolerance and the setMaxCouplingIterations)
        ti.setMechanicalTimeIntegration(tiMech)
        ti.setThermalTimeIntegration(tiTher)
        metafor.setTimeIntegration(ti)

        ti.setCouplingTolerance(1e-6)  # Tight coupling tolerance
        ti.setMaxCouplingIterations(10)  # Allow sufficient coupling iterations

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

    # TIME STEP MANAGEMENT - CRITICAL FOR PERFORMANCE - REFINED
    tsm = metafor.getTimeStepManager()
    tsm.setInitialTime(0.0, 0.01)  # Smaller initial step for better convergence

    # ADAPTIVE TIME STEPPING based on simulation phases - ENHANCED
    if sim_config.enable_thermal:
        # Phase 1: Mechanical loading (0 to T_load) - progressive refinement
        tsm.setNextTime(sim_config.loading_time * 0.25, 12, 0.01)  # Early loading Jusqu'à 2.5s
        tsm.setNextTime(sim_config.loading_time * 0.5, 12, 0.01)  # Mid loading Jusqu'à 5s
        tsm.setNextTime(sim_config.loading_time * 0.75, 25, 0.005)  # Late loading Jusqu'à 7.5s
        tsm.setNextTime(sim_config.loading_time, 25, 0.005)  # Final mechanical phase Jusqu'à 10s

        # Phase 2: Stabilization (T_load to temp_start_time) - MUCH SMALLER STEPS
        # Cette phase est critique pour la convergence
        tsm.setNextTime(sim_config.loading_time + 1.0, 10, 0.01)  # Premier seconde après loading Jusqu'à 11s
        tsm.setNextTime(sim_config.loading_time + 2.5, 15, 0.01)  # Stabilisation intermédiaire Jusqu'à 12.5s
        tsm.setNextTime(sim_config.temp_start_time - 2.0, 11, 0.05)  # Pré-thermal Jusqu'à 18s
        tsm.setNextTime(sim_config.temp_start_time, 4, 0.05)  # Juste avant thermal Jusqu'à 20s

        # Phase 3: Thermal loading (temp_start_time to temp_end_time) - adaptive steps
        thermal_duration = sim_config.temp_end_time - sim_config.temp_start_time
        tsm.setNextTime(sim_config.temp_start_time + thermal_duration * 0.1, 3, 0.05)  # Start thermal Jusqu'à 21.5s
        tsm.setNextTime(sim_config.temp_start_time + thermal_duration * 0.5, 12, 0.05)  # Mid thermal Jusqu'à 27.5s
        tsm.setNextTime(sim_config.temp_end_time, 37, 0.02)  # End thermal - finest steps Jusqu'à 35s

        # Phase 4: Final phase - can use larger steps again
        if sim_config.temp_end_time < sim_config.final_time:
            tsm.setNextTime(sim_config.final_time, 5, 0.1)

    else:
        # Pure mechanical - simpler time stepping
        tsm.setNextTime(T_load * 0.5, 25, 0.01)
        tsm.setNextTime(T_load, 50, 0.005)
        tsm.setNextTime(T, 40, 0.05)

    # TOLERANCE MANAGEMENT - ADAPTIVE for different phases
    fct_TOL = PieceWiseLinearFunction()
    fct_TOL.setData(0.0, 1.0)  # Standard tolerance for mechanical
    fct_TOL.setData(sim_config.loading_time * 0.75, 0.05)  # Tighter end of loading
    fct_TOL.setData(sim_config.loading_time, 0.03)  # Strict for transition
    fct_TOL.setData(sim_config.loading_time + 1.0, 0.01)  # Très strict stabilisation
    fct_TOL.setData(sim_config.loading_time + 2.5, 0.005)  # Maximum stricte 7.5-12s
    fct_TOL.setData(sim_config.temp_start_time - 1.0, 0.01)  # Relachement progressif
    fct_TOL.setData(sim_config.temp_start_time, 0.01)  # Standard pour thermal

    if sim_config.enable_thermal:
        # Stricter tolerance during thermal phase due to large temperature gradients
        fct_TOL.setData(sim_config.temp_start_time + (sim_config.temp_end_time - sim_config.temp_start_time) * 0.5,
                        0.2)  # Tightest mid-thermal
        fct_TOL.setData(sim_config.temp_end_time, 0.5)  # Relax after thermal
        fct_TOL.setData(sim_config.final_time, 1.0)  # Standard for final phase
    else:
        fct_TOL.setData(T, 1.0)

    # MECHANICAL ITERATION MANAGER - OPTIMIZED avec gestion de compatibilité
    mim = metafor.getMechanicalIterationManager()
    mim.setResidualComputationMethod(Method4ResidualComputation()) #----------------------------------------------Ligne ajoutée
    # Paramètres plus stricts pour la plasticité
    # ENHANCED CONVERGENCE CONTROL for critical phases
    fct_MaxIter = PieceWiseLinearFunction()
    fct_MaxIter.setData(0.0, 25.0)  # Standard pour début
    fct_MaxIter.setData(sim_config.loading_time * 0.75, 35.0)  # Plus d'itérations fin loading
    fct_MaxIter.setData(sim_config.loading_time, 40.0)  # Max pour transition
    fct_MaxIter.setData(sim_config.loading_time + 2.5, 35.0)  # Stabilisation critique
    fct_MaxIter.setData(sim_config.temp_start_time, 30.0)  # Retour normal avant thermal
    fct_MaxIter.setData(sim_config.temp_end_time, 40.0)  # Max pendant thermal
    fct_MaxIter.setData(sim_config.final_time, 25.0)  # Standard pour fin

    # Apply adaptive iteration control
    try:
        mim.setMaxNbOfIterations(40, fct_MaxIter)  # Fonction adaptative
        print("Adaptive iteration control set successfully")
    except (AttributeError, TypeError):
        print("Using fixed iteration control")
        mim.setMaxNbOfIterations(40)  # Fallback fixe plus élevé

    mim.setResidualTolerance(5e-5, fct_TOL)  # Plus strict : de 1e-4 à 5e-5
    mim.setPredictorComputationMethod(EXTRAPOLATION_MRUA)  # Better prediction

    # Line search method - avec gestion de compatibilité pour différentes versions
    try:
        mim.setLineSearchMethod(LINESEARCH_ARMIJO)  # Robust line search
        print("Line search method set successfully")
    except AttributeError:
        print("Line search method not available in this Metafor version - using default")
        # Alternative: essayer d'autres noms possibles
        try:
            mim.setLineSearch(LINESEARCH_ARMIJO)
        except AttributeError:
            try:
                # Autre possibilité selon la version
                mim.enableLineSearch(True)
            except AttributeError:
                print("No line search method available - continuing with default settings")

    # Alternative methods for load increment control
    try:
        # Try to set load increment parameters if available in your Metafor version
        mim.setMaxNbOfLoadIncrements(50)  # Alternative method name
        print("Load increment control set successfully")
    except AttributeError:
        # If not available, rely on time step control and iteration limits
        print("Using time step control for load increment management")
        try:
            # Essayer d'autres méthodes possibles
            mim.setMaxNumberOfLoadIncrements(50)
        except AttributeError:
            print("No load increment control available - using time step control only")

    # Auto-remeshing parameters for large deformation (if needed)
    try:
        mim.setAutoRemeshingParameters(0.8, 0.2, 2)  # Quality thresholds
        print("Auto-remeshing parameters set successfully")
    except AttributeError:
        print("Auto-remeshing not available in this version")

    # Additional stability settings pour améliorer la convergence
    try:
        # Paramètres de convergence plus robustes
        mim.setResidualComputationMethod(Method4ResidualComputation())
    except AttributeError:
        print("Residual computation method not available - using default")

    try:
        # Contrôle de l'incrément de charge
        mim.setConvergenceAccelerationMethod(CONVERGENCE_AITKEN)
    except AttributeError:
        print("Convergence acceleration not available - using default")

    # History curves (Save data in .txt files) - OPTIMIZED for thermal analysis
    interactionset = domain.getInteractionSet()
    if not p['postpro']:
        hcurves = metafor.getValuesManager()
        hcurves.add(1, MiscValueExtractor(metafor, EXT_T), 'time')
        hcurves.add(2, NormalForceValueExtractor(ci), SumOperator(), 'ContactForceY')
        hcurves.add(3, DbNodalValueExtractor(p20, Field1D(TY, RE)), SumOperator(), 'displacement_rod_end_Y')
        hcurves.add(4, DbNodalValueExtractor(c3, Field1D(TY, GF1)), SumOperator(), 'forceYExtClampinPt')
        hcurves.add(5, DbNodalValueExtractor(c3, Field1D(TX, GF1)), SumOperator(), 'forceXExtClampinPt')
        hcurves.add(6, DbNodalValueExtractor(p17, Field1D(TY, GF1)), SumOperator(), 'forceYHinge')
        hcurves.add(7, DbNodalValueExtractor(p17, Field1D(TX, GF1)), SumOperator(), 'forceXHinge')

        # Moment at clamping point
        ext0 = MomentValueExtractor(c3, pa3, TZ, GF1)
        hcurves.add(8, ext0, SumOperator(), 'MomentExtClampingPt')

        # Von Mises stress in blade
        ext_becu = IFNodalValueExtractor(interactionset(1), IF_EVMS)
        hcurves.add(9, ext_becu, MaxOperator(), 'Max_VonMises_BeCu')

        # Mass corner displacements - KEY for your analysis
        hcurves.add(10, DbNodalValueExtractor(p10, Field1D(TY, RE)), SumOperator(), 'dispY_Bottom_left_mass')
        hcurves.add(11, DbNodalValueExtractor(p11, Field1D(TY, RE)), SumOperator(), 'dispY_Bottom_right_mass')
        hcurves.add(12, DbNodalValueExtractor(p12, Field1D(TY, RE)), SumOperator(), 'dispY_Top_right_mass')
        hcurves.add(13, DbNodalValueExtractor(p9, Field1D(TY, RE)), SumOperator(), 'dispY_Top_left_mass')

        # X displacements for thermal effects
        hcurves.add(14, DbNodalValueExtractor(p20, Field1D(TX, RE)), SumOperator(), 'dispX_rod_end')

        # THERMAL EXTRACTORS - Critical for your study
        if sim_config.enable_thermal:
            # Temperature monitoring
            hcurves.add(15, DbNodalValueExtractor(s1, Field1D(TO, RE)), MeanOperator(), 'temp_mean_blade_K')
            hcurves.add(16, DbNodalValueExtractor(s1, Field1D(TO, RE)), MaxOperator(), 'temp_max_blade_K')
            hcurves.add(17, DbNodalValueExtractor(s1, Field1D(TO, RE)), MinOperator(), 'temp_min_blade_K')

            # Thermal strains - Important for understanding thermal effects
            hcurves.add(18, IFNodalValueExtractor(interactionset(1), IF_THERMAL_STRAIN), MeanOperator(),
                        'thermal_strain_mean_blade')
            hcurves.add(19, IFNodalValueExtractor(interactionset(1), IF_THERMAL_STRAIN), MaxOperator(),
                        'thermal_strain_max_blade')

            # Effective thermal displacement (mass movement due to thermal expansion)
            hcurves.add(20, DbNodalValueExtractor(p12, Field1D(TX, RE)), SumOperator(), 'thermal_dispX_mass_top_right')
            hcurves.add(21, DbNodalValueExtractor(p12, Field1D(TY, RE)), SumOperator(), 'thermal_dispY_mass_top_right')

            # Blade tip temperature and displacement for calibration
            hcurves.add(22, DbNodalValueExtractor(p3, Field1D(TO, RE)), SumOperator(), 'temp_blade_tip_K')
            hcurves.add(23, DbNodalValueExtractor(p3, Field1D(TX, RE)), SumOperator(), 'dispX_blade_tip')
            hcurves.add(24, DbNodalValueExtractor(p3, Field1D(TY, RE)), SumOperator(), 'dispY_blade_tip')

        # Validation of extractors
        # Plastic strain extractors pour surveiller la plasticité
        #if sim_config.enable_thermal:
        #    hcurves.add(25, IFNodalValueExtractor(interactionset(1), IF_PLASTIC_STRAIN), MaxOperator(),
        #                'max_plastic_strain_blade')
        #    hcurves.add(26, IFNodalValueExtractor(interactionset(1), IF_YIELD_FUNCTION), MaxOperator(),
        #                'max_yield_function_blade')
        #    hcurves.add(27, IFNodalValueExtractor(interactionset(2), IF_PLASTIC_STRAIN), MaxOperator(),
        #                'max_plastic_strain_structure')
        #    max_extractor = 27
        #else:
        #    max_extractor = 14

        max_extractor = 24 if sim_config.enable_thermal else 14
        for i in range(1, max_extractor + 1):
            metafor.getTestSuiteChecker().checkExtractor(i)

    # REAL-TIME PLOTTING - OPTIMIZED for thermal monitoring
    if not p['postpro']:
        try:
            # Plot 1: Mass displacements (your main interest)
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

            if sim_config.enable_thermal:
                # Plot 2: Temperature evolution in blade
                plot2 = DataCurveSet()
                plot2.add(VectorDataCurve(15, hcurves.getDataVector(1), hcurves.getDataVector(15), 'Mean Temp Blade'))
                plot2.add(VectorDataCurve(16, hcurves.getDataVector(1), hcurves.getDataVector(16), 'Max Temp Blade'))
                plot2.add(VectorDataCurve(17, hcurves.getDataVector(1), hcurves.getDataVector(17), 'Min Temp Blade'))
                plot2.add(VectorDataCurve(22, hcurves.getDataVector(1), hcurves.getDataVector(22), 'Blade Tip Temp'))

                win2 = VizWin()
                win2.add(plot2)
                win2.setPlotTitle("Temperature Evolution - Be-Cu Blade (10C -> 50C)")
                win2.setPlotXLabel("Time [s]")
                win2.setPlotYLabel("Temperature [K]")
                metafor.addObserver(win2)

                # Plot 3: Thermal effects on displacement
                plot3 = DataCurveSet()
                plot3.add(
                    VectorDataCurve(20, hcurves.getDataVector(1), hcurves.getDataVector(20), 'Mass Thermal Disp X'))
                plot3.add(
                    VectorDataCurve(21, hcurves.getDataVector(1), hcurves.getDataVector(21), 'Mass Thermal Disp Y'))
                plot3.add(VectorDataCurve(14, hcurves.getDataVector(1), hcurves.getDataVector(14), 'Rod End Disp X'))

                win3 = VizWin()
                win3.add(plot3)
                win3.setPlotTitle("Thermal-Induced Displacements - Sensor Response")
                win3.setPlotXLabel("Time [s]")
                win3.setPlotYLabel("Displacement [mm]")
                metafor.addObserver(win3)

                # Plot 4: Thermal strain evolution
                plot4 = DataCurveSet()
                plot4.add(
                    VectorDataCurve(18, hcurves.getDataVector(1), hcurves.getDataVector(18), 'Mean Thermal Strain'))
                plot4.add(
                    VectorDataCurve(19, hcurves.getDataVector(1), hcurves.getDataVector(19), 'Max Thermal Strain'))

                win4 = VizWin()
                win4.add(plot4)
                win4.setPlotTitle("Thermal Strain Evolution - Material Response")
                win4.setPlotXLabel("Time [s]")
                win4.setPlotYLabel("Thermal Strain [-]")
                metafor.addObserver(win4)

        except NameError:
            print("Warning: Visualization not available")
            pass

    return metafor


# def optimize_archiving(metafor, sim_config): #Ne fonctionne pas (A retravailler) :almost every step is saved, significantly increasing calculation time (+15go data) .
#     """Optimise la gestion des archives selon les phases de simulation"""
#     try:
#         fac = metafor.getFacManager()
#         archive = metafor.getArchiveManager()
#
#         if sim_config.enable_thermal:
#             print("Configuring OPTIMIZED archiving for thermal simulation...")
#             fac.setArchivingTime(45.0)  # Sauvegarde toutes les 45s CPU
#             fac.setCompressionLevel(2)  # Compression rapide
#             print(f"  - FacManager: archive every 45s CPU")
#             print(f"  - Compression level: 2")
#         else:
#             print("Configuring archiving for mechanical-only simulation...")
#             fac.setArchivingFrequency(150)  # Tous les 150 steps
#             fac.setCompressionLevel(3)
#             print(f"  - FacManager: archive every 150 steps")
#             print(f"  - Compression level: 3")
#
#         # ✅ Partie CRITIQUE : Réduire le nombre de sauvegardes réelles
#         archive.setTimeStepSavingFrequency(100)  # Sauvegarde 1 step sur 100
#         archive.setSaveByDefault(False)  # Ne pas tout enregistrer par défaut
#         print("ArchiveManager settings:")
#         print("  - Time step saving frequency: every 100 steps")
#         print("  - Save by default: False (only explicit saves)")
#
#         return fac
#
#     except AttributeError as e:
#         print(f"⚠️ Warning: Archive optimization not available in this Metafor version: {e}")
#         print("Continuing with default archiving settings...")
#         return None
#     except Exception as e:
#         print(f"❌ Error configuring archiving: {e}")
#         return None


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

    # CORRECTION: Utiliser load() sans paramètre comme dans votre ancien code
    # Cela charge le dernier état disponible SANS forcer l'enregistrement continu
    loader = fac.FacManager(metafor)
    loader.load()  # Sans paramètre = charge le dernier état disponible

    print('\n=== CORRECTED MODAL ANALYSIS AFTER THERMAL LOADING ===')

    # Enhanced frequency analysis on thermally loaded state
    curves = metafor.getValuesManager()

    lanczos = LanczosFrequencyAnalysisMethod(domain)
    lanczos.setNumberOfEigenValues(8)
    lanczos.setSpectralShifting(0.0)
    lanczos.setComputeEigenVectors(True)
    lanczos.setWriteMatrix2Matlab(True)

    # Paramètres plus précis pour l'analyse modale
    lanczos.setTolerance(1e-12)
    lanczos.setMaxIterations(1000)

    print('Computing eigenvalues on thermally deformed structure...')
    print('(Analysis performed on final thermal state)')

    try:
        lanczos.execute()
        lanczos.writeTSC()

        # Extraction des fréquences
        curves.add(11, FrequencyAnalysisValueExtractor(lanczos), 'freqs_thermal_corrected')

        # CORRECTION: Utiliser une approche plus robuste pour l'extraction
        # Au lieu de fillNow qui peut causer des problèmes, on fait l'extraction directement

        print('\n--- EIGENVALUE RESULTS (Direct extraction) ---')
        eigenvalues = []
        frequencies_hz = []

        for i in range(min(8, lanczos.getNumberOfEigenValues())):
            eigenval = lanczos.getEigenValue(i)

            if eigenval > 0:
                eigenvalues.append(eigenval)
                frequency_hz = math.sqrt(eigenval) / (2 * math.pi)
                frequencies_hz.append(frequency_hz)

                print(f'Mode {i + 1}: Eigenvalue = {eigenval:.8e}, Frequency = {frequency_hz:.4f} Hz')

                # Affichage du mode (optionnel)
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

        # Sauvegarde des résultats (optionnelle)
        try:
            curves.toAscii()
            curves.flush()

            # Lecture du fichier généré pour vérification
            with open('freqs_thermal_corrected.ascii') as f:
                txt = f.readlines()
            freq_values = [float(v) for v in txt[0].strip().split()]

            print(f'\n--- FILE VERIFICATION ---')
            print(f'Saved eigenvalues = {[f"{v:.6e}" for v in freq_values[:5]]}')
            file_frequencies = [math.sqrt(abs(val)) / (2 * math.pi) for val in freq_values[:5] if val > 0]
            print(f'File frequencies = {[f"{f:.4f} Hz" for f in file_frequencies]}')

        except Exception as e:
            print(f"File operations failed (non-critical): {e}")

        # Résumé final
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


def additional_diagnostics():
    """Diagnostics supplémentaires pour vérifier l'état thermique"""
    print('\n=== THERMAL STATE DIAGNOSTICS ===')

    # Vérification des fichiers de sortie thermique
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
    print('- Thermal strain: ~40C × 17e-6/C = 6.8e-4')
    print('- Frequency shift due to thermal expansion and stiffness change')


# Version alternative simplifiée si des problèmes persistent
def simple_modal_check():
    """Version ultra-simple pour diagnostic rapide"""
    print('\n=== SIMPLE MODAL CHECK ===')

    p = {}
    p['postpro'] = True
    metafor = instance(p)
    domain = metafor.getDomain()

    # Chargement de l'état final
    loader = fac.FacManager(metafor)
    loader.load()

    # Analyse modale basique
    lanczos = LanczosFrequencyAnalysisMethod(domain)
    lanczos.setNumberOfEigenValues(3)

    try:
        lanczos.execute()

        for i in range(3):
            eigenval = lanczos.getEigenValue(i)
            if eigenval > 0:
                freq = math.sqrt(eigenval) / (2 * math.pi)
                print(f'Mode {i + 1}: {freq:.4f} Hz')
    except Exception as e:
        print(f'Simple modal analysis failed: {e}')


if __name__ == "__main__":
    # Analyse principale
    postpro()
    additional_diagnostics()

    # Vérification simple en cas de problème
    # simple_modal_check()