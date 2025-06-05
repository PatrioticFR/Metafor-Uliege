# Study the impact of cryogenic condition on CSIS-V aligenment
# Hinge chnaged from point+spring to blade

# Geometry based on InertiaMassWITHscrew with srew at mid range
# First trial to implement the cryogenic part:
# - temperature dependnat material properties of BeCu
# - thermal intergration scheme
# - Temperature varying from 293K to 20K

# This code has to be run with the software METAFOR
# Morgane Zeoli
# April 2023

# -*- coding: utf-8 -*-

from wrap import *
from wrap.mtFrequencyAnalysisw import *
import math

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
    e = 0.24    # thickness
    L = 107.0#.307  # length

    # rod
    l = 89.6753    # total length
    H = 4.5   # thickness

    # Hole
    r = 28.8871
    R = 23.332
    a = 1.574

    # mass
    D = 31.1092   # height
    d = 39.3506   # length

    h = l-d
    y_enc = -14.3#-19.025#-26
    x_enc = 39.88#39.75

    # Hinge blade
    l_hinge = 0.5
    t_hinge = 0.05
    w_hinge = 8*2  #2 blades

    # mesh
    ne = 8      # blade - elements through the thickness
    nL = int(L*20)  # blade - elements along the length
    nh1 = 10
    nh2 = 3
    nH = 1
    nR = 2
    ncl = 5 #1  # clamping
    nd1 = 1
    nd2 = 5
    nd3 = 5
    np = 7
    nD = 3
    nt = 1
    nt_hinge = 4
    nl_hinge = 30#int(l_hinge*20)

    T_load = 5.0    # loading time
    T_cryo_start = 7.0
    T_cryo_end = 8.5
    T = 12.0 # final simulation time
    Dx = -72.04   # Param to set the reference horizontal location of the clamping pt (dist btw the 2 clamping pt)
    Dx1 = 0.0       # set horizontal shift (>0: shift closer to hinge)
    Dy = -3.0        # set vertical shift (>0: shift upward)
    angleClamp = 0.0    # set angle (>0: anti-clockwise)

    Tabs = 0.0
    Tamb = 273.15+20.0
    Tcryo = 20


    # Geometry

    geometry = domain.getGeometry()
    geometry.setDimPlaneStrain(1.0)

    pointset = geometry.getPointSet()

     # blade
    p1 = pointset.define(1, x_enc-e, y_enc)
    p2 = pointset.define(2, x_enc, y_enc)
    p3 = pointset.define(3, x_enc, y_enc+L)
    p4 = pointset.define(4, x_enc-e, y_enc+L)
    # rod
    p5 = pointset.define(5, 0.0, H/2)
    p6 = pointset.define(6, 0.0, -H/2)
    p7 = pointset.define(7, h, -H/2)
    p8 = pointset.define(8, h, H/2)
    # mass left piece
    p9 = pointset.define(9, h, D/2)
    p10 = pointset.define(10, h, -D/2)
    p11 = pointset.define(11, h+a, -D/2)
    p12 = pointset.define(12, h+a, D/2)
    # mass bottom piece
    p13 = pointset.define(13, h+d, -D/2)
    p14 = pointset.define(14, h+d, -R/2)
    p15 = pointset.define(15, h+a, -R/2)
    # mass top piece
    p16 = pointset.define(16, h+a, R/2)
    p17 = pointset.define(17, h+d, R/2)
    p18 = pointset.define(18, h+d, D/2)
    # mass right piece
    p19 = pointset.define(19, h+a+r, R/2)
    p20 = pointset.define(20, h+a+r, -R/2)
    # Geometry compatibility
    pO = pointset.define(21, 0.0, 0.0)
    p22 = pointset.define(22, h, 0.0)
    p23 = pointset.define(23, h+a, 0.0)
    p28 = pointset.define(28, h, R/2)
    p29 = pointset.define(29, h, -R/2)
    p30 = pointset.define(30, h+a, -H/2)
    p31 = pointset.define(31, h+a, H/2)
    p32 = pointset.define(32, h+a+r, -D/2)
    p33 = pointset.define(33, h+a+r, D/2)
    # clamping
    p24 = pointset.define(24, x_enc-e, -H/2)
    p25 = pointset.define(25, x_enc, -H/2)
    p26 = pointset.define(26, x_enc, H/2)
    p27 = pointset.define(27, x_enc-e, H/2)
    # Ground
    pGR = pointset.define(34, h+d+1, -D/2)
    pGL = pointset.define(35, 0, -D/2)
    # reference
    pref = pointset.define(36, Dx+x_enc, y_enc+Dy)
    # Hinge blade
    p39 = pointset.define(39, -l_hinge, t_hinge/2)
    p40 = pointset.define(40, -l_hinge, -t_hinge/2)
    p41 = pointset.define(41, 0, -t_hinge/2)
    p42 = pointset.define(42, 0, t_hinge/2)
    p43 = pointset.define(43, h, -t_hinge/2)
    p44 = pointset.define(44, h, t_hinge/2)
    p45 = pointset.define(45, h+a, -t_hinge/2)
    p46 = pointset.define(46, h+a, t_hinge/2)
    p47 = pointset.define(47, -l_hinge, 0)


    curveset = geometry.getCurveSet()
    # blade
    c1 = curveset.add(Line(1, p1, p2))
    c2 = curveset.add(Line(2, p2, p3))
    c3 = curveset.add(Line(3, p3, p4))
    c4 = curveset.add(Line(4, p4, p1))
    # rod
    c5 = curveset.add(Line(5, p5, p42))
    c6 = curveset.add(Line(6, p41, p6))
    c7 = curveset.add(Line(7, p6, p24))
    c8 = curveset.add(Line(8, p24, p25))
    c9 = curveset.add(Line(9, p25, p7))
    c10 = curveset.add(Line(10, p7, p43))
    c11 = curveset.add(Line(11, p44, p8))
    c12 = curveset.add(Line(12, p8, p26))
    c13 = curveset.add(Line(13, p26, p27))
    c14 = curveset.add(Line(14, p27, p5))
    # mass left piece
    c15 = curveset.add(Line(15, p9, p28))
    c16 = curveset.add(Line(16, p28, p8))
    c17 = curveset.add(Line(17, p7, p29))
    c18 = curveset.add(Line(18, p29, p10))
    c19 = curveset.add(Line(19, p10, p11))
    c20 = curveset.add(Line(20, p11, p15))
    c21 = curveset.add(Line(21, p15, p30))
    c22 = curveset.add(Line(22, p30, p45))
    c23 = curveset.add(Line(23, p46, p31))
    c24 = curveset.add(Line(24, p31, p16))
    c25 = curveset.add(Line(25, p16, p12))
    c26 = curveset.add(Line(26, p12, p9))
    # mass bottom piece
    c27 = curveset.add(Line(27, p11, p32))
    c28 = curveset.add(Line(28, p32, p13))
    c29 = curveset.add(Line(29, p13, p14))
    c30 = curveset.add(Line(30, p14, p20))
    c31 = curveset.add(Line(31, p20, p15))
    # mass top piece
    c32 = curveset.add(Line(32, p16, p19))
    c33 = curveset.add(Line(33, p19, p17))
    c34 = curveset.add(Line(34, p17, p18))
    c35 = curveset.add(Line(35, p18, p33))
    c36 = curveset.add(Line(36, p33, p12))
    # mass right piece
    c37 = curveset.add(Line(37, p19, p20))
    c38 = curveset.add(Line(38, p14, p17))
    # clamping
    c39 = curveset.add(Line(39, p24, p1))
    c40 = curveset.add(Line(40, p2, p25))
    # Ground
    c41 = curveset.add(Line(41, pGR, pGL))
    # Hinge and hinge compatibility
    c42 = curveset.add(Line(42, p42, pO))
    c43 = curveset.add(Line(43, pO, p41))
    c44 = curveset.add(Line(44, p43, p22))
    c45 = curveset.add(Line(45, p22, p44))
    c46 = curveset.add(Line(46, p45, p23))
    c47 = curveset.add(Line(47, p23, p46))
    c48 = curveset.add(Line(48, p42, p39))
    c49 = curveset.add(Line(49, p39, p47))
    c50 = curveset.add(Line(50, p47, p40))
    c51 = curveset.add(Line(51, p40, p41))

    wireset = geometry.getWireSet()
    w1 = wireset.add( Wire(1, [c1, c2, c3, c4]) ) # blade
    w2 = wireset.add( Wire(2, [c5, c42, c43, c6, c7, c8, c9, c10, c44, c45, c11, c12, c13, c14]) ) # rod
    w3 = wireset.add( Wire(3, [c15, c16, c11, c45, c44, c10, c17, c18, c19, c20, c21, c22, c46, c47, c23, c24, c25, c26]) ) # mass left piece
    w4 = wireset.add( Wire(4, [c20, c27, c28, c29, c30, c31]) ) # mass bottom piece
    w5 = wireset.add( Wire(5, [c25, c32, c33, c34, c35, c36]) ) # mass top piece
    w6 = wireset.add( Wire(6, [c37, c30, c38, c33]) ) # mass right piece
    w7 = wireset.add( Wire(7, [c1, c40, c8, c39]) ) # clamping
    w8 = wireset.add( Wire(8, [c48, c49, c50, c51, c43, c42]) ) # Hinge
    w9 = wireset.add( Wire(9, [c41]) ) # ground


    sideset = geometry.getSideSet()
    s1 = sideset.add(Side(1, [w1]))
    s2 = sideset.add(Side(2, [w2]))
    s3 = sideset.add(Side(3, [w3]))
    s4 = sideset.add(Side(4, [w4]))
    s5 = sideset.add(Side(5, [w5]))
    s6 = sideset.add(Side(6, [w6]))
    s7 = sideset.add(Side(7, [w7]))
    s8 = sideset.add(Side(8, [w8]))
    s9 = sideset.add(Side(9, [w9]))

    if 0:
        win = VizWin()
        win.add(pointset)
        win.add(curveset)
        win.open()
        input()

    # Mesh
    prog = 5
    SimpleMesher1D(c1).execute(ne)
    SimpleMesher1D(c2).execute(nL)
    SimpleMesher1D(c3).execute(ne)
    SimpleMesher1D(c4).execute(nL)
    SimpleMesher1D(c5).execute(nH)
    SimpleMesher1D(c6).execute(nH)
    SimpleMesher1D(c7).execute(nh1)
    SimpleMesher1D(c8).execute(ne)
    SimpleMesher1D(c9).execute(nh2)
    SimpleMesher1D(c10).execute(nH)
    SimpleMesher1D(c11).execute(nH)
    SimpleMesher1D(c12).execute(nh2)
    SimpleMesher1D(c13).execute(ne)
    SimpleMesher1D(c14).execute(nh1)
    SimpleMesher1D(c15).execute(nt)
    SimpleMesher1D(c16).execute(nR)
    SimpleMesher1D(c17).execute(nR)
    SimpleMesher1D(c18).execute(np, 1/prog)
    SimpleMesher1D(c19).execute(nd1)
    SimpleMesher1D(c20).execute(np, prog)
    SimpleMesher1D(c21).execute(nR)
    SimpleMesher1D(c22).execute(nH)
    SimpleMesher1D(c23).execute(nH)
    SimpleMesher1D(c24).execute(nR)
    SimpleMesher1D(c25).execute(nt)
    SimpleMesher1D(c26).execute(nd1)
    SimpleMesher1D(c27).execute(nd2)
    SimpleMesher1D(c28).execute(nd3)
    SimpleMesher1D(c29).execute(np, prog)
    SimpleMesher1D(c30).execute(nd3)
    SimpleMesher1D(c31).execute(nd2)
    SimpleMesher1D(c32).execute(nd2)
    SimpleMesher1D(c33).execute(nd3)
    SimpleMesher1D(c34).execute(nt)
    SimpleMesher1D(c35).execute(nd3)
    SimpleMesher1D(c36).execute(nd2)
    SimpleMesher1D(c37).execute(nD)
    SimpleMesher1D(c38).execute(nD)
    SimpleMesher1D(c39).execute(ncl)
    SimpleMesher1D(c40).execute(ncl)
    SimpleMesher1D(c42).execute(nt_hinge)
    SimpleMesher1D(c43).execute(nt_hinge)
    SimpleMesher1D(c44).execute(nt_hinge)
    SimpleMesher1D(c45).execute(nt_hinge)
    SimpleMesher1D(c46).execute(nt_hinge)
    SimpleMesher1D(c47).execute(nt_hinge)
    SimpleMesher1D(c48).execute(nl_hinge)
    SimpleMesher1D(c49).execute(nt_hinge)
    SimpleMesher1D(c50).execute(nt_hinge)
    SimpleMesher1D(c51).execute(nl_hinge)

    TransfiniteMesher2D(s1).execute(True)
    TransfiniteMesher2D(s2).execute2( ((5, 42, 43, 6),(7, 8, 9),(10, 45, 44, 11),(12,13,14)) )
    TransfiniteMesher2D(s3).execute2( ((15, 16, 11, 44, 45, 10, 17, 18),19,(20, 21, 22, 46, 47, 23, 24, 25),26) )
    TransfiniteMesher2D(s4).execute2( (20,(27, 28),29,(30, 31)) )
    TransfiniteMesher2D(s5).execute2( (25,(32, 33),34,(35, 36)) )
    TransfiniteMesher2D(s6).execute(True)
    TransfiniteMesher2D(s7).execute(True)
    TransfiniteMesher2D(s8).execute2( (48,(49, 50),51,(43, 42)) )

    #Création d'un groupset permettant d'extraire la température sur les noeuds de la longueur
    groupset = geometry.getGroupSet()
    grp = groupset.add(Group(1))
    grp.addMeshPointsFromObject(curveset(2))

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
    # Thermal Expansion of Technical Solids at Low Temperatures, National Bureau of Standards
    fctE = PieceWiseLinearFunction()
    fctE.setData(293.15, 131e3)
    fctE.setData(287.086092715232, 131e3)
    fctE.setData(274.503311258278,	131.961538461538e3)
    fctE.setData(253.311258278146,	131.961538461538e3)
    fctE.setData(241.390728476821,	131.961538461538e3)
    fctE.setData(229.139072847682,	132.442307692308e3)
    fctE.setData(219.205298013245,	132.923076923077e3)
    fctE.setData(210.596026490066,	132.923076923077e3)
    fctE.setData(200.662251655629,	133.403846153846e3)
    fctE.setData(186.423841059603,	134.365384615385e3)
    fctE.setData(179.139072847682,	134.846153846154e3)
    fctE.setData(165.562913907285,	134.846153846154e3)
    fctE.setData(153.642384105960,	135.807692307692e3)
    fctE.setData(139.072847682119,	136.288461538462e3)
    fctE.setData(123.178807947020,	137.250000000000e3)
    fctE.setData(106.622516556291,	138.211538461538e3)
    fctE.setData(99.6688741721854,	138.692307692308e3)
    fctE.setData(82.7814569536424,	139.653846153846e3)
    fctE.setData(71.1920529801325,	140.615384615385e3)
    fctE.setData(60.9271523178808,	141.096153846154e3)
    fctE.setData(51.6556291390728,	141.576923076923e3)
    fctE.setData(42.7152317880795,	142.057692307692e3)
    fctE.setData(32.7814569536424,	143.019230769231e3)
    fctE.setData(20,	144.038e3)
    # E = 0.0001085x^2-0.08148x+145.6


    fctCTE = PieceWiseLinearFunction()
    #fctCTE.setData(300, 18.1e-6)
    fctCTE.setData(293.15, 17.9e-6)
    fctCTE.setData(280, 17.4e-6)
    fctCTE.setData(273, 17.2e-6)
    fctCTE.setData(260, 16.7e-6)
    fctCTE.setData(240, 16.0e-6)
    fctCTE.setData(220, 15.2e-6)
    fctCTE.setData(200, 14.5e-6)
    fctCTE.setData(180, 13.8e-6)
    fctCTE.setData(160, 13.2e-6)
    fctCTE.setData(140, 12.4e-6)
    fctCTE.setData(120, 11.6e-6)
    fctCTE.setData(100, 10.4e-6)
    fctCTE.setData(90, 9.6e-6)
    fctCTE.setData(80, 8.4e-6)
    fctCTE.setData(70, 6.5e-6)
    fctCTE.setData(60, 4.3e-6)
    fctCTE.setData(50, 2.7e-6)
    fctCTE.setData(40, 1.4e-6)
    fctCTE.setData(30, 0.5e-6)
    fctCTE.setData(20, 0.09e-6)
    fctCTE.setData(10, 0.04e-6)
    fctCTE.setData(0.0, 0.0)

    materials = domain.getMaterialSet()
    materials.define (1, TmElastHypoMaterial)
    materials(1).put(MASS_DENSITY,     8.36e-9)
    materials(1).put(ELASTIC_MODULUS, 1.0)
    materials(1).depend(ELASTIC_MODULUS, fctE, Field1D(TO,RE))
    materials(1).put(POISSON_RATIO,    0.285)
    materials(1).put(THERM_EXPANSION, 1.0)
    materials(1).depend(THERM_EXPANSION, fctCTE, Field1D(TO,RE))
    materials(1).put(CONDUCTIVITY, 1.0)
    materials(1).put(HEAT_CAPACITY, 1.e6)
    materials(1).put(DISSIP_TE, 0.0)
    materials(1).put(DISSIP_TQ, 0.0)

    # Steel
    # https://eurocodeapplied.com/design/en1993/steel-design-properties
    materials.define (2,TmElastHypoMaterial)
    materials(2).put(MASS_DENSITY,     7.93e-9)
    materials(2).put(ELASTIC_MODULUS, 210e3)
    materials(2).put(POISSON_RATIO,    0.3)
    materials(2).put(THERM_EXPANSION, 0.0)
    materials(2).put(CONDUCTIVITY, 1.0)
    materials(2).put(HEAT_CAPACITY, 1.e6)
    materials(2).put(DISSIP_TE, 0.0)
    materials(2).put(DISSIP_TQ, 0.0)
    # /!\ TO CHANGE

    laws = domain.getMaterialLawSet()
    laws.define (1, LinearIsotropicHardening)
    laws(1).put(IH_SIGEL, 1000.0)
    laws(1).put(IH_H,    1000.0)

    laws.define (2, LinearIsotropicHardening)
    laws(2).put(IH_SIGEL, 400.0)
    laws(2).put(IH_H,    1000.0)

    prp1 = ElementProperties(TmVolume2DElement)
    prp1.put(MATERIAL, 1)
    prp1.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)
    prp1.put(THICKNESS, 45.0)   # Set blade width

    prp3 = ElementProperties(TmVolume2DElement)
    prp3.put(MATERIAL, 1)
    prp3.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)
    prp3.put(THICKNESS, w_hinge)   # Set blade width

    fctG = PieceWiseLinearFunction()
    fctG.setData(0.0, 1.0)
    fctG.setData(T_load/10, 1.0)
    fctG.setData(T, 1.0)

    prp2 = ElementProperties(TmVolume2DElement)
    prp2.put(MATERIAL, 2)
    prp2.put(GRAVITY_Y, -9.81e3) # [mm/s^2]
    prp2.depend(GRAVITY_Y, fctG, Field1D(TM))
    prp2.put(THICKNESS, 63.0)   # Set inertial mass width
    prp2.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)

    app = FieldApplicator(1)
    app.push(s1)
    app.addProperty(prp1)
    domain.getInteractionSet().add(app)

    app4 = FieldApplicator(4)
    app4.push(s8)
    app4.addProperty(prp3)
    domain.getInteractionSet().add(app4)

    app2 = FieldApplicator(2)
    app2.push(s2)
    app2.push(s3)
    app2.push(s4)
    app2.push(s5)
    app2.push(s6)
    app2.push(s7)
    app2.addProperty(prp2)
    domain.getInteractionSet().add(app2)


    # Boundary conditions

    # Mechanical BC
    # bottom: clamped
    domain.getLoadingSet().define(c49, Field1D(TX,RE), 0.0) # hinge clamped
    domain.getLoadingSet().define(c49, Field1D(TY,RE), 0.0) # hinge clamped
    domain.getLoadingSet().define(c50, Field1D(TX,RE), 0.0) # hinge clamped
    domain.getLoadingSet().define(c50, Field1D(TY,RE), 0.0) # hinge clamped
    domain.getLoadingSet().define(c41, Field1D(TX,RE), 0.0) # ground first fixed
    domain.getLoadingSet().define(c41, Field1D(TY,RE), 0.0) # ground first fixed

    # Thermal BC
    domain.getInitialConditionSet().define(s1,Field1D(TO,AB),Tabs) #Température initiale
    domain.getInitialConditionSet().define(s2,Field1D(TO,AB),Tabs)
    domain.getInitialConditionSet().define(s3,Field1D(TO,AB),Tabs)
    domain.getInitialConditionSet().define(s4,Field1D(TO,AB),Tabs)
    domain.getInitialConditionSet().define(s5,Field1D(TO,AB),Tabs)
    domain.getInitialConditionSet().define(s6,Field1D(TO,AB),Tabs)
    domain.getInitialConditionSet().define(s7,Field1D(TO,AB),Tabs)
    domain.getInitialConditionSet().define(s8,Field1D(TO,AB),Tabs)
    domain.getInitialConditionSet().define(s1,Field1D(TO,RE),Tamb) #Température initiale
    domain.getInitialConditionSet().define(s2,Field1D(TO,RE),Tamb)
    domain.getInitialConditionSet().define(s3,Field1D(TO,RE),Tamb)
    domain.getInitialConditionSet().define(s4,Field1D(TO,RE),Tamb)
    domain.getInitialConditionSet().define(s5,Field1D(TO,RE),Tamb)
    domain.getInitialConditionSet().define(s6,Field1D(TO,RE),Tamb)
    domain.getInitialConditionSet().define(s7,Field1D(TO,RE),Tamb)
    domain.getInitialConditionSet().define(s8,Field1D(TO,RE),Tamb)


    # Rotation axis for the rotation of the blade
    pa1 = pointset.define(37, x_enc-e/2, y_enc+L/2, 0.0)
    pa2 = pointset.define(38, x_enc-e/2, y_enc+L/2, 1.0)
    axe1 = Axe(pa1,pa2)

    # axis displacement
    # The rotation axis moves during the loading so that the clamping point is
    # located at the desired location in the end. First the LF the deplacement
    # and rotation are set to find the reference equilibrium configuration,
    # then the clamping offset is added.
    fctX = PieceWiseLinearFunction()
    fctX.setData(0.0, 0.0)
    fctX.setData(T_load/8, 0.0)
    fctX.setData(T_load, 1.0)
    fctX.setData(T, 1.0)
    fctY = PieceWiseLinearFunction()
    fctY.setData(0.0, 0.0)
    fctY.setData(T_load/2, 0.0)
    fctY.setData(T_load, 1.0)
    fctY.setData(T, 1.0)
    domain.getLoadingSet().define(pa1, Field1D(TX,RE) ,Dx, fctX)
    domain.getLoadingSet().define(pa2, Field1D(TX,RE), Dx, fctX)
    domain.getLoadingSet().define(pa1, Field1D(TY,RE), Dy, fctY)
    domain.getLoadingSet().define(pa2, Field1D(TY,RE), Dy, fctY)

    # rotation
    # The blade is rotation around a pre-defined axis to obtain its curved shape.
    # The first rotation enables to be in the reference equilibrium configuration.
    # The second rotation is the clmaping point rotation.
    fctR = PieceWiseLinearFunction()
    fctR.setData(0.0, 0.0)
    fctR.setData(T_load/10, 0.0)
    fctR.setData(3*T_load/4, 1.0)
    fctR.setData(T_load, 1.0)
    fctR.setData(T, 1.0)

    domain.getLoadingSet().defineRot(c3, Field3D(TXTYTZ,RE), pa1, pa2, False, 180.0, fctR) # If only the first rotation is needed

    # Ground displacement
    # When the mass is at equilibrium, the ground is removed to avoid perturbing
    # the free oscillations of the sensor.
    fctSol = PieceWiseLinearFunction()
    fctSol.setData(0.0, 0.0)
    fctSol.setData(T_load, 0.0)
    fctSol.setData(T_cryo_start, 1.0)
    fctSol.setData(T, 1.0)
    DSol = -60
    domain.getLoadingSet().define(c41, Field1D(TY,RE), DSol, fctSol)


    # Themral management
    fctT = PieceWiseLinearFunction()
    fctT.setData(0.0, Tamb)
    fctT.setData(T_load, Tamb)
    fctT.setData(T_cryo_start, Tamb)
    fctT.setData(T_cryo_end, Tcryo)
    fctT.setData(T, Tcryo)
    domain.getLoadingSet().define(s1, Field1D(TO,RE), 1.0, fctT)
    domain.getLoadingSet().define(s2, Field1D(TO,RE), 1.0, fctT)
    domain.getLoadingSet().define(s3, Field1D(TO,RE), 1.0, fctT)
    domain.getLoadingSet().define(s4, Field1D(TO,RE), 1.0, fctT)
    domain.getLoadingSet().define(s5, Field1D(TO,RE), 1.0, fctT)
    domain.getLoadingSet().define(s6, Field1D(TO,RE), 1.0, fctT)
    domain.getLoadingSet().define(s7, Field1D(TO,RE), 1.0, fctT)
    domain.getLoadingSet().define(s8, Field1D(TO,RE), 1.0, fctT)

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
    ci.setTool(curveset(41))
    ci.push(curveset(28))
    ci.addProperty(prp3)
    domain.getInteractionSet().add(ci)


    # Time integration scheme
    tsm = metafor.getTimeStepManager()
    tsm.setInitialTime(0.0, 0.01)
    #tsm.setNextTime(  T, 1, 0.01)
    tsm.setNextTime(  T, 1, 0.005)

    fct_TOL = PieceWiseLinearFunction()
    fct_TOL.setData(0.0, 1.0)
    fct_TOL.setData(T, 1.0)

    mim = metafor.getMechanicalIterationManager()
    mim.setResidualComputationMethod(Method4ResidualComputation())
    mim.setMaxNbOfIterations(4)
    mim.setResidualTolerance(1e-4, fct_TOL)
    mim.setPredictorComputationMethod(EXTRAPOLATION_MRUA)

    tim = metafor.getThermalIterationManager()
    tim.setResidualComputationMethod(Method4ResidualComputation())
    tim.setResidualTolerance(1e-4)

    tiMech = AlphaGeneralizedTimeIntegration(metafor)
    tiTher = TrapezoidalThermalTimeIntegration(metafor)
    tiTher.setTheta(1.0)
    ti = StaggeredTmTimeIntegration(metafor)
    ti.setIsAdiabatic(False)
    ti.setWithStressReevaluation(False)
    ti.setMechanicalTimeIntegration(tiMech)
    ti.setThermalTimeIntegration(tiTher)
    metafor.setTimeIntegration(ti)

    # History curves (Save data is .txt files)
    interactionset = domain.getInteractionSet()
    if not p['postpro']:
        hcurves = metafor.getValuesManager()
        hcurves.add(1, MiscValueExtractor(metafor,EXT_T), 'time')
        hcurves.add(2, DbNodalValueExtractor(p13, Field1D(TY,RE)), SumOperator(), 'displacement') # End rod pt displacement
        ext = IFNodalValueExtractor(interactionset(1), IF_EVMS)
        hcurves.add(3, ext, MaxOperator(),'Max VonMises')
        hcurves.add(4, InteractionGravityCenterAndMassValueExtractor(interactionset(2)))
        #hcurves.add(5, DbNodalValueExtractor(groupset(1), Field1D(TO,RE), SortByX0()), 'temp_re')
        hcurves.add(5, DbNodalValueExtractor(p22, Field1D(TO,RE)), 'temp_re_mass')


        for i in range(1,6):
            metafor.getTestSuiteChecker().checkExtractor(i)

    # plot curves during simulation
    if not p['postpro']:
        try:
            plot1 = DataCurveSet()
            plot1.add(VectorDataCurve(1, hcurves.getDataVector(1),
                                         hcurves.getDataVector(2), 'displacement'))
            win1 = VizWin()
            win1.add(plot1)
            win1.setPlotTitle(" Rod end point vertical displacement as a function of time")
            win1.setPlotXLabel("Time [s]")
            win1.setPlotYLabel("Y [mm]")
            metafor.addObserver(win1)

            plot2 = DataCurveSet()
            plot2.add(VectorDataCurve(2, hcurves.getDataVector(1),
                                         hcurves.getDataVector(5), 'Temperature'))
            win2 = VizWin()
            win2.add(plot2)
            win2.setPlotTitle(" Temperature evolution")
            win2.setPlotXLabel("Time [s]")
            win2.setPlotYLabel("Temperature [K]")
            metafor.addObserver(win2)

            plot3 = DataCurveSet()
            plot3.add(VectorDataCurve(3, hcurves.getDataVector(1),
                                         hcurves.getDataVector(3), 'Von Mises'))
            win3 = VizWin()
            win3.add(plot3)
            win3.setPlotTitle(" Von Mises evolution")
            win3.setPlotXLabel("Time [s]")
            win3.setPlotYLabel("Stress [MPa]")
            metafor.addObserver(win3)

        except NameError:
            pass

    return metafor

# ----------------------------------------------------------------------------------
# Modal analysis to extract the sensor resonance frequencies and mode shapes
from toolbox.utilities import *

def postpro():
    import os.path
    setDir('workspace/%s' % os.path.splitext(os.path.basename(__file__))[0])
    load(__name__)

    p={}
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
    fExtr =  FrequencyAnalysisValueExtractor(lanczos)
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
        print('Eigen Vector ', i , 'EigenValue = ', lanczos.getEigenValue(i))
        input("press enter to continue")


    with open('freqs.ascii') as f:
        txt = f.readlines()

    print(f'eigenvalues = {[float(v) for v in txt[0].strip().split()]}')

# ----------------------------------------------------------------------------------

if __name__=="__main__":
    postpro()
7

