# TFE: Towards the development of an ultra-high performance inertial sensor for
# gravitational waves detection

# Study of the muVINS and the influence of the leaf-spring suspension
# length and clamping point location and orientation on the equilibrium position
# and the resonance frequency.

# This code has to be run with the software METAFOR
# Morgane Zeoli

# -*- coding: utf-8 -*-

# Thread Configuration:
#
# In V9, the number of threads for Intel Threading Building Blocks (TBB) is explicitly set to 16 using the environment variable TBB_NUM_THREADS. This configuration aims to optimize the parallel processing capabilities of the simulation, likely to improve performance on a multi-core processor.
# Thickness Adjustments:
#
# The thickness of the Beryllium-Copper (Be-Cu) blade has been increased:
# e = 0.27: This is an increase from the previous value of 0.24, which will affect the blade's mechanical properties and stiffness.
# The thickness of the Invar blade has also been increased:
# ei = 0.25: This is an increase from the previous value of 0.07, significantly changing the blade's flexibility and interaction characteristics.
# Thickness Properties for Blades:
#
# The thickness properties for both the Be-Cu and Invar blades have been updated:
# Be-Cu blade thickness: prp1.put(THICKNESS, 20.5)
# Invar blade thickness: prp5.put(THICKNESS, 27.0)
# These changes are likely intended to adjust the stiffness and mechanical behavior of the blades in the simulation.
# Comments and Documentation:
#
# Additional comments and clarifications have been added to explain the purpose of certain parameters and adjustments, such as the thread configuration and thickness settings.

from wrap import *
from wrap.mtFrequencyAnalysisw import *
import math
import os
from toolbox.utilities import *

# Set the number of threads for TBB (Intel Threading Building Blocks)
# A value of 12 is chosen as a starting point for a 24-thread CPU (e.g., i7 13700hx),
# representing half of the available logical cores (threads).
os.environ['TBB_NUM_THREADS'] = '16'

# enable full parallelism
StrVectorBase.useTBB()
StrMatrixBase.useTBB()
ContactInteraction.useTBB()

def getMetafor(d={}):

    # default parameters
    p = {}
    p['postpro'] = False
    p.update(d)

    metafor = Metafor()
    domain = metafor.getDomain()

    # Parameters

    # blade
    e = 0.27    # thickness
    L = 105.25  # length

    # rod
    l = 79.2    # total length
    H = 3.875   # thickness

    r = 7       # distance between mass and rod end
    R = H
    enc =  57.32    # Leaf-spring clamping point location on the rod (hinge = origin pt)

    # Invar blade
    ei = 0.25  # invar thickness
    # Relative position of the blades (horizontal offset of 0.1mm)
    decalage = 0.1  # offset between the blades
    rayon_interne_BeCu = enc / 2
    R_beCu = rayon_interne_BeCu + e / 2  # Median radius of the Be-Cu blade
    R_invar = rayon_interne_BeCu - decalage - ei / 2  # Median radius of the Invar blade
    ratio = R_invar / R_beCu
    Li = L * ratio # The length must be proportional to the radius for the same bending angle

    # mass
    D = 39.99   # height
    d = 13.96   # length
    #y = 14.53  # shift mass
    y = D/2     # the mass is centered on the rod
    h = l-r-d

    # mesh
    ne = 8      # blade - elements through the thickness
    nL = int(L*20)  # blade - elements along the length
    nd = 10
    nr = 1
    n56 = 2   # rod vertical (/2)
    n7 = 5 #1  # rod horizontal
    n9 = 1    # rod horiz 2
    n14 = 3  # mass vertical 1
    n15 = 17  # mass vertical 2

    import math
    T = 12.0        # final simulation time
    T_load = 10.0   # loading time
    Dx = -67.5227   # Param to set the reference horizontal location of the clamping pt (dist btw the 2 clamping pt)
    Dx1 = 0.0       # set horizontal shift (>0: shift closer to hinge)
    Dy = 0.0        # set vertical shift (>0: shift upward)
    angleClamp = 0.0    # set angle (>0: anti-clockwise)

    #Dx_invar = Dx * ratio  # <--- smaller than Dx
    Dx_invar = Dx * ratio  # Adjusted to keep the blades separate when folded

    # Geometry

    geometry = domain.getGeometry()
    geometry.setDimPlaneStrain(1.0)

    pointset = geometry.getPointSet()

     # blade
    p1 = pointset.define(1, enc, H/2)
    p2 = pointset.define(2, enc+e, H/2)
    p3 = pointset.define(3, enc+e, L)
    p4 = pointset.define(4, enc, L)
    # rod
    p5 = pointset.define(5, 0.0, H/2)
    p6 = pointset.define(6, 0.0, -H/2)
    p7 = pointset.define(7, h, -H/2)
    p8 = pointset.define(8, h, H/2)
    # mass
    p9 = pointset.define(9, h, D-y)
    p10 = pointset.define(10, h, -y)
    p11 = pointset.define(11, h+d, -y)
    p12 = pointset.define(12, h+d, D-y)
    # end rod
    p13 = pointset.define(13, h+d, R/2)
    p14 = pointset.define(14, h+d, -R/2)
    p15 = pointset.define(15, h+d+r, -R/2)
    p16 = pointset.define(16, h+d+r, R/2)
    # Geometry compatibility
    p17 = pointset.define(17, 0.0, 0.0)
    p18 = pointset.define(18, h, 0.0)
    p19 = pointset.define(19, h+d, 0.0)
    p20 = pointset.define(20, h+d+r, 0.0)
    p21 = pointset.define(21, enc, -H/2)
    p22 = pointset.define(22, enc+e, -H/2)

    # Ground
    p25 = pointset.define(25, h+d+r, -y)
    p26 = pointset.define(26, 0, -y)

    p27 = pointset.define(27, Dx+enc, 0.0)

    # Middle plane
    p28 = pointset.define(28, enc, 0.0)
    p29 = pointset.define(29, e+enc, 0.0)
    # Spring
    p30 = pointset.define(30, 0.0, -H/2)

    # New Invar leaf spring (side by side with the Be-Cu blade)
    p31 = pointset.define(31, enc - ei - decalage, H / 2)      # Bottom left point Invar blade
    p32 = pointset.define(32, enc - decalage, H / 2)  # Bottom right point Invar blade
    p33 = pointset.define(33, enc - decalage, Li)  # Top right point Invar blade
    p34 = pointset.define(34, enc - ei - decalage, Li)  # Top left point Invar blade

    # Compatibility planes for Invar blade
    p35 = pointset.define(35, enc - ei - 0.1, -H / 2)  # Left median plane point Invar blade
    p36 = pointset.define(36, enc - 0.1, -H / 2)  # Right median plane point Invar blade

    # Median plane
    p37 = pointset.define(37, enc - ei - 0.1, 0.0)
    p38 = pointset.define(38, enc - 0.1, 0.0)

    curveset = geometry.getCurveSet()
    # blade
    c1 = curveset.add(Line(1, p1, p2))
    c2 = curveset.add(Line(2, p29, p3))
    c3 = curveset.add(Line(3, p3, p4))
    c4 = curveset.add(Line(4, p4, p28))
    # rod
    c5 = curveset.add(Line(5, p5, p17))
    c6 = curveset.add(Line(6, p17, p6))
    c71 = curveset.add(Line(71, p6, p35))
    c72 = curveset.add(Line(72, p35, p36))
    c73 = curveset.add(Line(73, p36, p21))
    c8 = curveset.add(Line(8, p21, p22))
    c9 = curveset.add(Line(9, p22, p7))
    c10 = curveset.add(Line(10, p7, p18))
    c11 = curveset.add(Line(11, p18, p8))
    c12 = curveset.add(Line(12, p8, p2))
    c131 = curveset.add(Line(131, p1, p32))
    c132 = curveset.add(Line(132, p31, p5))

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
    c271 = curveset.add(Line(271, p17, p37))
    c272 = curveset.add(Line(272, p38, p28))
    c28 = curveset.add(Line(28, p28, p29))
    c29 = curveset.add(Line(29, p29, p18))
    # Median plane Invar blade
    c34 = curveset.add(Line(34, p37, p38))

    # Invar leaf spring
    c30 = curveset.add(Line(30, p31, p32))
    c31 = curveset.add(Line(31, p38, p33))
    c32 = curveset.add(Line(32, p33, p34))
    c33 = curveset.add(Line(33, p34, p37))

    wireset = geometry.getWireSet()
    w1 = wireset.add( Wire(1, [c28, c2, c3, c4]) )
    w2 = wireset.add(Wire(2, [c5, c271,c34, c272,  c28, c29, c11, c12, c1, c131,c30 ,c132]))
    w3 = wireset.add(Wire(3, [c6, c71, c72, c73, c8, c9, c10, c29, c28, c272,c34,c271]))
    w4 = wireset.add( Wire(4, [c14, c11, c10, c15, c16, c17, c18, c19, c20, c21]) )
    w5 = wireset.add( Wire(5, [c19, c18, c22, c23, c24, c25]) )
    w6 = wireset.add( Wire(6, [c26]) )

    # Wire for the Invar blade
    w7 = wireset.add( Wire(7, [c34, c31, c32, c33]) )

    sideset = geometry.getSideSet()
    s1 = sideset.add(Side(1, [w1]))
    s2 = sideset.add(Side(2, [w2]))
    s3 = sideset.add(Side(3, [w3]))
    s4 = sideset.add(Side(4, [w4]))
    s5 = sideset.add(Side(5, [w5]))
    s6 = sideset.add(Side(6, [w6]))

    # Side for the Invar blade
    s7 = sideset.add(Side(7, [w7]))

    if 0:
        win = VizWin()
        win.add(pointset)
        win.add(curveset)
        win.open()
        input()

    # Mesh
    prog = 5
    # Curves of the blade
    SimpleMesher1D(c1).execute(ne)
    SimpleMesher1D(c2).execute(nL)
    SimpleMesher1D(c3).execute(ne)
    SimpleMesher1D(c4).execute(nL)
    # Curves of the rod
    SimpleMesher1D(c5).execute(n56)
    #SimpleMesher1D(c5).execute(14)  # Adjusted to 15 nodes exactly for Edge #1
    SimpleMesher1D(c6).execute(n56)
    SimpleMesher1D(c71).execute(n7)
    SimpleMesher1D(c72).execute(ne)
    SimpleMesher1D(c73).execute(n7)
    SimpleMesher1D(c8).execute(ne)
    SimpleMesher1D(c9).execute(n9)
    SimpleMesher1D(c10).execute(n56)
    SimpleMesher1D(c11).execute(n56)
    #SimpleMesher1D(c11).execute(19)  # Adjusted to 20 nodes exactly for Edge #3
    SimpleMesher1D(c12).execute(n9)
    SimpleMesher1D(c131).execute(n7)
    SimpleMesher1D(c132).execute(n7)
    # Curves of the mass
    SimpleMesher1D(c14).execute(n14)
    SimpleMesher1D(c15).execute(n15, 1 / prog)
    SimpleMesher1D(c16).execute(nd)
    SimpleMesher1D(c17).execute(n15, prog)
    SimpleMesher1D(c18).execute(n56)
    SimpleMesher1D(c19).execute(n56)
    SimpleMesher1D(c20).execute(n14)
    SimpleMesher1D(c21).execute(nd)
    # Curves of the end rod
    SimpleMesher1D(c22).execute(nr)
    SimpleMesher1D(c23).execute(n56)
    SimpleMesher1D(c24).execute(n56)
    SimpleMesher1D(c25).execute(nr)
    # Curves of the median plane
    SimpleMesher1D(c271).execute(n7)
    SimpleMesher1D(c272).execute(n7)
    SimpleMesher1D(c28).execute(ne)
    SimpleMesher1D(c29).execute(n9)
    SimpleMesher1D(c34).execute(ne)
    # New curves for the Invar blade
    SimpleMesher1D(c30).execute(ne)
    SimpleMesher1D(c31).execute(nL)
    SimpleMesher1D(c32).execute(ne)
    SimpleMesher1D(c33).execute(nL)

    # 2. Modified transfinite mesh
    TransfiniteMesher2D(s1).execute(True)
    TransfiniteMesher2D(s2).execute2((5, (271,34,272, 28, 29), 11, (12, 1, 131, 30, 132)))
    TransfiniteMesher2D(s3).execute2((6, (71, 72, 73, 8, 9), 10, (29, 28, 272,34,271)))
    TransfiniteMesher2D(s4).execute2(((14, 11, 10, 15), 16, (17, 18, 19, 20), 21))
    TransfiniteMesher2D(s5).execute2(((19, 18), 22, (23, 24), 25))
    TransfiniteMesher2D(s7).execute(True)

    if 0:
        win = VizWin()
        win.add(geometry.getMesh().getPointSet())
        win.add(geometry.getMesh().getCurveSet())
        win.add(curveset)
        win.open()
        input()

    # Volumic material/prp/elements
    # beryllium - copper
    # https://material-properties.org/beryllium-copper-density-strength-hardness-melting-point/
    materials = domain.getMaterialSet()
    materials.define (1, EvpIsoHHypoMaterial)
    materials(1).put(MASS_DENSITY,     8.36e-9)
    materials(1).put(ELASTIC_MODULUS, 131e3)
    materials(1).put(POISSON_RATIO,    0.285)
    materials(1).put(YIELD_NUM,           1)
    # Steel
    # https://eurocodeapplied.com/design/en1993/steel-design-properties
    materials.define (2, EvpIsoHHypoMaterial)
    materials(2).put(MASS_DENSITY,     8.0415e-9)#1.19824e-9) #7.831e-9
    materials(2).put(ELASTIC_MODULUS, 210e3)
    materials(2).put(POISSON_RATIO,    0.3)
    materials(2).put(YIELD_NUM,           2)

    # Spring
    materials.define(4,ConstantSpringMaterial)
    materials(4).put(SPRING_FK,  11.7211)   # Hinge rotational stiffness in N/mm (4*k_rot/H^2)

    # Invar (Fe-Ni 36%) - Very low thermal expansion coefficient
    #https://www.azom.com/properties.aspx?ArticleID=515
    materials.define (5, EvpIsoHHypoMaterial)
    materials(5).put(MASS_DENSITY,     8.1e-9)      # kg/mm³ (min: 8.05 – max: 8.15)
    materials(5).put(ELASTIC_MODULUS, 143e3)        # MPa (min: 140 – max: 148)
    materials(5).put(POISSON_RATIO,    0.26)  #(min: 0.23 – max: 0.294)
    materials(5).put(YIELD_NUM,           3)

    laws = domain.getMaterialLawSet()
    laws.define (1, LinearIsotropicHardening)
    laws(1).put(IH_SIGEL, 1000.0)
    laws(1).put(IH_H,    1000.0)

    laws.define (2, LinearIsotropicHardening)
    laws(2).put(IH_SIGEL, 400.0)
    laws(2).put(IH_H,    1000.0)

    # Behavior law for Invar
    laws.define (3, LinearIsotropicHardening)
    laws(3).put(IH_SIGEL, 276.0)    # Yield limit Invar (min: 240 – max: 380)
    laws(3).put(IH_H,    800.0) # (min: 500 – max: 1500)

    prp1 = ElementProperties(Volume2DElement)
    prp1.put(MATERIAL, 1)
    prp1.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)
    prp1.put(THICKNESS, 20.5)   # Set blade width

    # Properties for the Invar blade
    prp5 = ElementProperties(Volume2DElement)
    prp5.put(MATERIAL, 5)
    prp5.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)
    prp5.put(THICKNESS, 27.0)   # Decrease thickness to reduce stiffness

    fctG = PieceWiseLinearFunction()
    fctG.setData(0.0, 1.0)
    fctG.setData(T_load/10, 1.0)
    fctG.setData(T, 1.0)

    prp2 = ElementProperties(Volume2DElement)
    prp2.put(MATERIAL, 2)
    prp2.put(GRAVITY_Y, -9.81e3) # [mm/s^2]
    prp2.depend(GRAVITY_Y, fctG, Field1D(TM))
    prp2.put(THICKNESS, 63.0)   # Set inertial mass width
    prp2.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)

    app = FieldApplicator(1)
    app.push(s1)
    app.addProperty(prp1)
    domain.getInteractionSet().add(app)

    # Application of properties to the Invar blade
    app5 = FieldApplicator(5)
    app5.push(s7)
    app5.addProperty(prp5)
    domain.getInteractionSet().add(app5)

    app2 = FieldApplicator(2)
    app2.push(s2)
    app2.push(s3)
    app2.push(s4)
    app2.push(s5)
    app2.addProperty(prp2)
    domain.getInteractionSet().add(app2)

    # ATTRIBUTE GROUPS
    groupset = geometry.getGroupSet()
    groupset.add(Group(1)); groupset(1).addMeshPoint(p6)
    groupset.add(Group(2)); groupset(2).addMeshPoint(p30)
    # Generation of the "spring mesh"
    springMesher = CellLineMesher(groupset(1),groupset(2))
    springMesher.execute()
    # Spring inducing the stiffness of the hinge
    prp4 = ElementProperties(Spring2DElement)
    prp4.put(MATERIAL, 4)
    prp4.put(SPRING_CLI,   0)
    prp4.put(STIFFMETHOD, STIFF_ANALYTIC)
    app4 = FieldApplicator(4)
    app4.push(groupset(1))
    app4.addProperty(prp4)
    domain.getInteractionSet().add(app4)

    # Prescribed displacements
    # bottom: clamped
    domain.getLoadingSet().define(p17, Field1D(TX,RE), 0.0) # hinge clamped
    domain.getLoadingSet().define(p17, Field1D(TY,RE), 0.0) # hinge clamped
    domain.getLoadingSet().define(c26, Field1D(TX,RE), 0.0) # ground first fixed
    domain.getLoadingSet().define(c26, Field1D(TY,RE), 0.0) # ground first fixed
    domain.getLoadingSet().define(p30, Field1D(TY,RE), 0.0) # hinge spring end fixed
    domain.getLoadingSet().define(p30, Field1D(TX,RE), 0.0) # hinge spring end fixed

    # Rotation axis for the Be-Cu blade
    pa1 = pointset.define(23, enc + e / 2, L / 2)
    pa2 = pointset.define(24, enc + e / 2, L / 2, 1.0)
    axe1_BeCu = Axe(pa1, pa2)
    axe1_BeCu.setSymZ1(1.0)

    # Rotation axis for the Invar blade
    pa5 = pointset.define(41, enc - ei / 2 - decalage, Li / 2)
    pa6 = pointset.define(42, enc - ei / 2 - decalage, Li / 2, 1.0)
    axe1_Invar = Axe(pa5, pa6)
    axe1_Invar.setSymZ1(1.0)

    # For Be-Cu blade
    pa3 = pointset.define(39, Dx + Dx1 + enc + e / 2, Dy, 0.0)
    pa4 = pointset.define(40, Dx + Dx1 + enc + e / 2, Dy, 1.0)
    axe2 = Axe(pa3, pa4)

    # For Invar blade - ensure consistent spacing
    pa3i = pointset.define(43, Dx + Dx1 + enc - ei / 2 - decalage, Dy, 0.0)  # Adjusted x-coordinate
    pa4i = pointset.define(44, Dx + Dx1 + enc - ei / 2 - decalage, Dy, 1.0)
    axe2_Invar = Axe(pa3i, pa4i)

    # axis displacement
    fctX = PieceWiseLinearFunction()
    fctX.setData(0.0, 0.0)
    fctX.setData(T_load / 8, 0.0)
    fctX.setData(T_load / 2, Dx / (Dx + Dx1))
    fctX.setData(3 * T_load / 4, 1.0)
    fctX.setData(T_load, 1.0)
    fctX.setData(T, 1.0)

    fctX_invar = PieceWiseLinearFunction()
    fctX_invar.setData(0.0, 0.0)
    fctX_invar.setData(T_load / 8, 0.0)
    fctX_invar.setData(T_load / 2, Dx_invar / (Dx + Dx1))  # <-- new value
    fctX_invar.setData(3 * T_load / 4, 1.0)
    fctX_invar.setData(T_load, 1.0)
    fctX_invar.setData(T, 1.0)

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

    #domain.getLoadingSet().define(pa5, Field1D(TX, RE), (Dx + Dx1), fctX)
    #domain.getLoadingSet().define(pa6, Field1D(TX, RE), (Dx + Dx1), fctX)
    domain.getLoadingSet().define(pa5, Field1D(TX, RE), Dx_invar, fctX_invar)
    domain.getLoadingSet().define(pa6, Field1D(TX, RE), Dx_invar, fctX_invar)
    domain.getLoadingSet().define(pa3i, Field1D(TX, RE), Dx_invar, fctX_invar)
    domain.getLoadingSet().define(pa4i, Field1D(TX, RE), Dx_invar, fctX_invar)
    domain.getLoadingSet().define(pa5, Field1D(TY, RE), Dy, fctY)
    domain.getLoadingSet().define(pa6, Field1D(TY, RE), Dy, fctY)
    domain.getLoadingSet().define(pa3i, Field1D(TY, RE), Dy, fctY)
    domain.getLoadingSet().define(pa4i, Field1D(TY, RE), Dy, fctY)

    # rotation functions
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

    # Rotation functions for Invar (same timing, but frozen axis movement)

    # Rotation (same as fctR)
    fctR_invar = PieceWiseLinearFunction()
    fctR_invar.setData(0.0, 0.0)
    fctR_invar.setData(T_load / 10, 0.0)
    fctR_invar.setData(T_load / 2, 1.0)
    fctR_invar.setData(3 * T_load / 4, 1.0)
    fctR_invar.setData(T_load, 1.0)
    fctR_invar.setData(T, 1.0)

    # Displacement of the Invar rotation axis (fixed at 0.0)
    fctR2_invar = PieceWiseLinearFunction()
    fctR2_invar.setData(0.0, 0.0)
    fctR2_invar.setData(T_load / 10, 0.0)
    fctR2_invar.setData(T_load / 2, 0.0)
    fctR2_invar.setData(3 * T_load / 4, 0.0)
    fctR2_invar.setData(T_load, 1.0) #0.0 before
    fctR2_invar.setData(T, 1.0) #0.0 before

    # Apply rotation to Be-Cu blade
    domain.getLoadingSet().defineRot2(c3, Field3D(TXTYTZ, RE), axe2, angleClamp, fctR2, axe1_BeCu, 180, fctR, False)

    # Rotation applied to the Invar blade
    domain.getLoadingSet().defineRot2(
        c32, Field3D(TXTYTZ, RE),
        axe2_Invar, angleClamp, fctR2_invar,
        axe1_Invar, 180, fctR_invar, False)

    # Ground displacement
    # When the mass is at equilibrium, the ground is removed to avoid perturbing
    # the free oscillations of the sensor.
    fctSol = PieceWiseLinearFunction()
    fctSol.setData(0.0, 0.0)
    fctSol.setData(T_load, 0.0)
    fctSol.setData(T, 1.0)
    DSol = -10
    domain.getLoadingSet().define(c26, Field1D(TY,RE), DSol, fctSol)

    # Contact material/prp/elements
    # Contact between mass and ground (the mass first lies on the ground until
    # the LF is able to compensate for its gravitational load)
    materials.define (3, FrictionlessContactMaterial)
    materials(3).put(PEN_NORMALE,   1e2)
    materials(3).put(PROF_CONT,     1.0)

    prp3 = ElementProperties(Contact2DElement)
    prp3.put(MATERIAL, 3)
    prp3.put(AREAINCONTACT, AIC_ONCE)

    ci = RdContactInteraction(3)
    ci.setTool(curveset(26))
    ci.push(curveset(16))
    ci.addProperty(prp3)
    domain.getInteractionSet().add(ci)

    # Time integration scheme
    ti = AlphaGeneralizedTimeIntegration(metafor)
    metafor.setTimeIntegration(ti)

    tsm = metafor.getTimeStepManager()
    tsm.setInitialTime(0.0, 0.01)
    #tsm.setNextTime(  T, 1, 0.005)  # Set min time set (useful for free oscillations)
    tsm.setNextTime(  T, 1, 0.1)
    # To reduce the simulation time, a longer dt can be used but it affects the
    # accuracy of the free oscillations and thus the final equilibrium configuration
    # of the sensor, especially when close to instability. This dt must be used
    # only in the trial and error phase to find a rough estimation of the clamping
    # point location that leads to the sensor equilibrium

    # Tolerance
    fct_TOL = PieceWiseLinearFunction()
    fct_TOL.setData(0.0, 1.0)
    fct_TOL.setData(T_load, 1.0)
    fct_TOL.setData(11*T_load/10, 1/10)
    fct_TOL.setData(T, 1/10)

    mim = metafor.getMechanicalIterationManager()
    mim.setMaxNbOfIterations(4)
    mim.setResidualTolerance(1e-4, fct_TOL)
    #mim.setVerbose(True)
    mim.setPredictorComputationMethod(EXTRAPOLATION_MRUA) # 2nd deg extrapolation

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