# TFE: Towards the development of an ultra-high performance inertial sensor for
# gravitational waves detection

# Study of the muVINS with configurable blade parameters
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
    """General simulation parameters"""

    def __init__(self):
        # Time parameters
        self.final_time = 30.0  # T
        self.loading_time = 10.0  # T_load

        # Clamping parameters
        self.Dx = -67.5227  # reference horizontal location
        self.Dx1 = 0.0  # horizontal shift
        self.Dy = 0.0  # vertical shift
        self.angleClamp = 0.0  # clamping angle

        # Temperature parameters
        self.enable_thermal = True
        self.temp_initial = 10.0
        self.temp_final = 50.0
        self.temp_start_time = 12.0
        self.temp_end_time = 20.0


# Material database
MATERIALS_DB = {
    'BE_CU': {
        'name': 'Beryllium-Copper',
        'density': 8.36e-9,  # kg/mm³
        'elastic_modulus': 131e3,  # MPa
        'poisson_ratio': 0.285,
        'yield_strength': 1000.0,  # MPa
        'hardening': 1000.0,
        'thermal_expansion': 17e-6,  # 1/°C
        'material_id': 1,
        'law_id': 1
    },
    'INVAR': {
        'name': 'Invar (Fe-Ni 36%)',
        'density': 8.1e-9,  # kg/mm³
        'elastic_modulus': 143e3,  # MPa
        'poisson_ratio': 0.26,
        'yield_strength': 276.0,  # MPa
        'hardening': 800.0,
        'thermal_expansion': 1.2e-6,  # 1/°C (very low)
        'material_id': 5,
        'law_id': 3
    },
    'STEEL': {
        'name': 'Steel',
        'density': 8.0415e-9,  # kg/mm³
        'elastic_modulus': 210e3,  # MPa
        'poisson_ratio': 0.3,
        'yield_strength': 400.0,  # MPa
        'hardening': 1000.0,
        'thermal_expansion': 12e-6,  # 1/°C
        'material_id': 6,
        'law_id': 4
    }
}


def getMetafor(d={}):
    """
    Main function with configurable parameters
    Usage examples:
    - getMetafor()  # default BE_CU blade
    - getMetafor({'blade_material': 'INVAR', 'blade_thickness': 0.15})
    - getMetafor({'blade_length': 120, 'blade_width': 50})
    """

    # Initialize configurations
    blade_config = BladeConfig()
    sim_config = SimulationConfig()

    # Apply user overrides
    p = {'postpro': False}
    p.update(d)

    # Override blade parameters if provided
    if 'blade_material' in d:
        blade_config.material = d['blade_material']
    if 'blade_thickness' in d:
        blade_config.thickness = d['blade_thickness']
    if 'blade_length' in d:
        blade_config.length = d['blade_length']
    if 'blade_width' in d:
        blade_config.width = d['blade_width']
    if 'elements_thickness' in d:
        blade_config.elements_thickness = d['elements_thickness']

    # Override simulation parameters if provided
    if 'final_time' in d:
        sim_config.final_time = d['final_time']
    if 'loading_time' in d:
        sim_config.loading_time = d['loading_time']
    if 'enable_thermal' in d:
        sim_config.enable_thermal = d['enable_thermal']

    # Validate material selection
    if blade_config.material not in MATERIALS_DB:
        raise ValueError(f"Unknown material: {blade_config.material}. Available: {list(MATERIALS_DB.keys())}")

    blade_material = MATERIALS_DB[blade_config.material]

    print(f"=== SIMULATION CONFIGURATION ===")
    print(f"Blade Material: {blade_material['name']}")
    print(
        f"Blade Dimensions: L={blade_config.length:.2f}mm, e={blade_config.thickness:.2f}mm, w={blade_config.width:.1f}mm")
    print(f"Thermal effects: {'Enabled' if sim_config.enable_thermal else 'Disabled'}")
    print(f"=================================")

    metafor = Metafor()
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

    # Fixed rod and mass parameters (can be made configurable too if needed)
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

    # Geometry (unchanged)
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

    # Curves and wires (unchanged geometry definition)
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

    # Mesh with configurable parameters
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

    # Materials - automatically configured based on selection
    materials = domain.getMaterialSet()
    laws = domain.getMaterialLawSet()

    # Define selected blade material
    mat_id = blade_material['material_id']
    law_id = blade_material['law_id']

    materials.define(mat_id, EvpIsoHHypoMaterial)
    materials(mat_id).put(MASS_DENSITY, blade_material['density'])
    materials(mat_id).put(ELASTIC_MODULUS, blade_material['elastic_modulus'])
    materials(mat_id).put(POISSON_RATIO, blade_material['poisson_ratio'])
    materials(mat_id).put(YIELD_NUM, law_id)

    laws.define(law_id, LinearIsotropicHardening)
    laws(law_id).put(IH_SIGEL, blade_material['yield_strength'])
    laws(law_id).put(IH_H, blade_material['hardening'])

    # Steel for structure (unchanged)
    materials.define(2, EvpIsoHHypoMaterial)
    materials(2).put(MASS_DENSITY, 8.0415e-9)
    materials(2).put(ELASTIC_MODULUS, 210e3)
    materials(2).put(POISSON_RATIO, 0.3)
    materials(2).put(YIELD_NUM, 2)

    laws.define(2, LinearIsotropicHardening)
    laws(2).put(IH_SIGEL, 400.0)
    laws(2).put(IH_H, 1000.0)

    # Spring material
    materials.define(4, ConstantSpringMaterial)
    materials(4).put(SPRING_FK, 11.7211)

    # Element properties - blade uses selected material
    prp1 = ElementProperties(Volume2DElement)
    prp1.put(MATERIAL, mat_id)  # Use selected material
    prp1.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)
    prp1.put(THICKNESS, blade_config.width)  # Use configurable width

    fctG = PieceWiseLinearFunction()
    fctG.setData(0.0, 1.0)
    fctG.setData(T_load / 10, 1.0)
    fctG.setData(T, 1.0)

    prp2 = ElementProperties(Volume2DElement)
    prp2.put(MATERIAL, 2)
    prp2.put(GRAVITY_Y, -9.81e3)
    prp2.depend(GRAVITY_Y, fctG, Field1D(TM))
    prp2.put(THICKNESS, 63.0)
    prp2.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)

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

    # Spring setup
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

    # Loading functions
    fctX = PieceWiseLinearFunction()
    fctX.setData(0.0, 0.0)
    fctX.setData(T_load / 8, 0.0)
    fctX.setData(T_load / 2, Dx / (Dx + Dx1))
    fctX.setData(3 * T_load / 4, 1.0)
    fctX.setData(T_load, 1.0)
    fctX.setData(T, 1.0)

    fctY = PieceWiseLinearFunction()
    fctY.setData(0.0, 0.0)
    fctY.setData(T_load / 2, 0.0)
    fctY.setData(3 * T_load / 4, 1.0)
    fctY.setData(T_load, 1.0)
    fctY.setData(T, 1.0)

    domain.getLoadingSet().define(pa1, Field1D(TX, RE), (Dx + Dx1), fctX)
    domain.getLoadingSet().define(pa2, Field1D(TX, RE), (Dx + Dx1), fctX)
    domain.getLoadingSet().define(pa1, Field1D(TY, RE), Dy, fctY)
    domain.getLoadingSet().define(pa2, Field1D(TY, RE), Dy, fctY)

    # Rotation functions
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

    domain.getLoadingSet().defineRot2(c3, Field3D(TXTYTZ, RE), axe2, angleClamp, fctR2, axe1, 180, fctR, False)

    # Thermal loading - conditionally applied
    if sim_config.enable_thermal:
        print(f"Applying thermal loading: {sim_config.temp_initial}°C → {sim_config.temp_final}°C")

        materials(mat_id).put(THERM_EXPANSION, blade_material['thermal_expansion'])
        materials(mat_id).put(REFERENCE_TEMPERATURE, sim_config.temp_initial)


        fct_Temp = PieceWiseLinearFunction()
        fct_Temp.setData(0.0, sim_config.temp_initial)
        fct_Temp.setData(sim_config.temp_start_time, sim_config.temp_initial)
        fct_Temp.setData(sim_config.temp_end_time, sim_config.temp_final)
        fct_Temp.setData(T, sim_config.temp_final)

        domain.getLoadingSet().define(s1, Field1D(TEMPERATURE, RE), fct_Temp)

    # Ground displacement
    fctSol = PieceWiseLinearFunction()
    fctSol.setData(0.0, 0.0)
    fctSol.setData(T_load, 0.0)
    fctSol.setData(T, 1.0)
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

    # Time integration
    ti = AlphaGeneralizedTimeIntegration(metafor)
    metafor.setTimeIntegration(ti)

    tsm = metafor.getTimeStepManager()
    tsm.setInitialTime(0.0, 0.01)
    tsm.setNextTime(T, 1, 0.1)

    # Tolerance
    fct_TOL = PieceWiseLinearFunction()
    fct_TOL.setData(0.0, 1.0)
    fct_TOL.setData(T_load, 1.0)
    fct_TOL.setData(11 * T_load / 10, 1 / 10)
    fct_TOL.setData(T, 1 / 10)

    mim = metafor.getMechanicalIterationManager()
    mim.setMaxNbOfIterations(4)
    mim.setResidualTolerance(1e-4, fct_TOL)
    mim.setPredictorComputationMethod(EXTRAPOLATION_MRUA)
    # History curves (Save data is .txt files)
    interactionset = domain.getInteractionSet()
    if not p['postpro']:
        hcurves = metafor.getValuesManager()
        hcurves.add(1, MiscValueExtractor(metafor, EXT_T), 'time')
        hcurves.add(2, NormalForceValueExtractor(ci), SumOperator(), 'ContactForceY')
        hcurves.add(3, DbNodalValueExtractor(p20, Field1D(TY, RE)), SumOperator(),
                    'displacement')  # End rod pt displacement
        hcurves.add(4, DbNodalValueExtractor(c3, Field1D(TY, GF1)), SumOperator(), 'forceYExtClampinPt')
        hcurves.add(5, DbNodalValueExtractor(c3, Field1D(TX, GF1)), SumOperator(), 'forceXExtClampinPt')
        hcurves.add(6, DbNodalValueExtractor(p17, Field1D(TY, GF1)), SumOperator(), 'forceYHinge')
        hcurves.add(7, DbNodalValueExtractor(p17, Field1D(TX, GF1)), SumOperator(), 'forceXHinge')
        ext0 = MomentValueExtractor(c3, pa3, TZ, GF1)
        hcurves.add(8, ext0, SumOperator(), 'MomentExtClampingPt')

        ext_becu = IFNodalValueExtractor(interactionset(1), IF_EVMS)
        ext_invar = IFNodalValueExtractor(interactionset(5), IF_EVMS)

        hcurves.add(9, ext_becu, MaxOperator(), 'Max_VonMises_BeCu')
        hcurves.add(14, ext_invar, MaxOperator(), 'Max_VonMises_Invar')

        # Addition of vertical displacements of p10 and p11
        hcurves.add(10, DbNodalValueExtractor(p10, Field1D(TY, RE)), SumOperator(), 'dispY_Bottom left mass corner')
        hcurves.add(11, DbNodalValueExtractor(p11, Field1D(TY, RE)), SumOperator(), 'dispY_Bottom right mass corner')
        hcurves.add(12, DbNodalValueExtractor(p12, Field1D(TY, RE)), SumOperator(), 'dispY_Top right mass corner')
        hcurves.add(13, DbNodalValueExtractor(p9, Field1D(TY, RE)), SumOperator(), 'dispY_Top left mass corner')

        for i in range(1, 15):  # Adjusted up to 13 included
            metafor.getTestSuiteChecker().checkExtractor(i)

    # plot curves during simulation
    if not p['postpro']:
        try:
            plot1 = DataCurveSet()
            plot1.add(VectorDataCurve(2, hcurves.getDataVector(1), hcurves.getDataVector(3), 'Rod end (middle) Y'))
            plot1.add(VectorDataCurve(10, hcurves.getDataVector(1), hcurves.getDataVector(10), 'Bottom left mass corner Y'))
            plot1.add(VectorDataCurve(11, hcurves.getDataVector(1), hcurves.getDataVector(11), 'Bottom right mass corner Y'))
            plot1.add(VectorDataCurve(12, hcurves.getDataVector(1), hcurves.getDataVector(12), 'Top right mass corner Y'))
            plot1.add(VectorDataCurve(13, hcurves.getDataVector(1), hcurves.getDataVector(13), 'Top left mass corner Y'))

            win1 = VizWin()
            win1.add(plot1)
            win1.setPlotTitle("Vertical displacements vs Time")
            win1.setPlotXLabel("Time [s]")
            win1.setPlotYLabel("Y displacement [mm]")
            metafor.addObserver(win1)

        except NameError:
            pass

    return metafor

# ----------------------------------------------------------------------------------
# Modal analysis to extract the sensor resonance frequencies and mode shapes
from toolbox.utilities import *

#To use the postpro you need to: Load the file and start the metafor program. After it was completed, close the window and start metafor again.
#Then execute the python files. To get the next eigen values, click on "browse"

def postpro():
    import os.path
    # The working directory is defined here
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

    with open('freqs.ascii') as f: # 'freqs.ascii' is correct here as setDir has been called
        txt = f.readlines()

    print(f'eigenvalues = {[float(v) for v in txt[0].strip().split()]}')

    # Reading the maximum Von Mises values (when executing click on "vizu")

    # Read time.ascii file to get corresponding times
    time_values = []
    if os.path.exists('time.ascii'):
        with open('time.ascii', 'r') as f_time:
            time_values = [float(line.strip()) for line in f_time if line.strip()]
    else:
        print("File not found: time.ascii. Cannot associate time with Von Mises values.")
        # The program will continue, but associated times will not be displayed.

    von_mises_files = {
        "Be-Cu": 'Max_VonMises_BeCu.ascii',
        "Invar": 'Max_VonMises_Invar.ascii'
    }

    for material, filename in von_mises_files.items():
        print(f"Reading file: {filename}")
        if os.path.exists(filename):
            print(f"File found: {filename}")
            with open(filename, 'r') as f:
                # Read values, one per line (correct for your format)
                values = [float(line.strip()) for line in f if line.strip()]

                if values:
                    # Check if we have time values and if lengths match
                    if time_values and len(time_values) == len(values):
                        max_val = max(values)
                        max_time_index = values.index(max_val)  # Find the index of the max value
                        max_time = time_values[max_time_index]  # Use this index to find the corresponding time
                        print(f"{material}: Maximum Von Mises stress = {max_val:.2f} MPa at t = {max_time:.2f} s")
                    else:
                        # If times cannot be associated (missing file or different lengths)
                        print(
                            f"Warning: Could not associate time for {material}. Time steps count: {len(time_values)}, Von Mises values count: {len(values)}.")
                        if values:  # Still print the max value even without associated time
                            max_val = max(values)
                            print(f"{material}: Maximum Von Mises stress (time unconfirmed) = {max_val:.2f} MPa")
                else:
                    print(f"Empty file: {filename}")
        else:
            print(f"File not found: {filename}")


# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    postpro()