# TFE: Towrads the development of an ultra-high performance inertial sensor for
# gravitational waves detection

# Study of the muVINS and the influence of the leaf-spring spring suspension
# length and clamping point location and orientation on the equilibrium position
# and the resonance frequency.

# This code has to be run with the software METAFOR
# Morgane Zeoli

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
    L = 105.25  # length

    # rod
    l = 79.2    # total length
    H = 3.875   # thickness

    r = 7       # distance betwee mass and rod end
    R = H
    enc =  57.32    # Leaf-spring clamping point location on the rod (hinge = origin pt)

    # mass
    D = 39.99   # height
    d = 13.96   # length
    #y = 14.53  # shift mass
    y = D/2     # the mass is centered on the rod
    h = l-r-d

    import math
    T = 12.0        # final simulation time
    T_load = 10.0   # loading time
    Dx = -67.5227   # Param to set the reference horizontal location of the clamping pt (dist btw the 2 clamping pt)
    Dx1 = 0.0       # set horizontal shift (>0: shift closer to hinge)
    Dy = -67.0        # set vertical shift (>0: shift upward)
    angleClamp = 0.0    # set angle (>0: anti-clockwise)


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
    p20  = pointset.define(20, h + d + r, 0)
    p201 = pointset.define(201, h + d + r, e / 2)
    p202 = pointset.define(202, h + d + r, -e / 2)

    # Plan médian
    p21 = pointset.define(21, enc, H / 2)
    p22 = pointset.define(22, enc, -H / 2)

    # Sol
    p25 = pointset.define(25, h + d + r, -y)
    p26 = pointset.define(26, 0, -y)

    # Point de serrage opposé
    p271 = pointset.define(271, Dx + enc, 0.0) #Left clamping point
    p272 = pointset.define(272, enc, Dy) #Above clamping point
    p273 = pointset.define(273, enc, -Dy) #Under clamping point

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
    SimpleMesher1D(c26).execute(1)  # sol

    # Plans de coupe
    #SimpleMesher1D(c28).execute(nL)  # plan médian sup (même que c3)
    #SimpleMesher1D(c30).execute(nL)  # plan médian inf (même que c2)
    SimpleMesher1D(c31).execute(n56)  # plan coupe vertical sup
    SimpleMesher1D(c32).execute(n56)  # plan coupe vertical inf

    # Maillage 2D corrigé
    TransfiniteMesher2D(s1).execute(True)
    TransfiniteMesher2D(s2).execute2(((51,52,6), 71, (32,4,31), 122))
    TransfiniteMesher2D(s3).execute2(((31,4,32), 72, (101,102,11), 121))
    TransfiniteMesher2D(s4).execute2(((14, 11, 102, 101, 15), 16, (17, 181, 182, 19, 20), 21))
    TransfiniteMesher2D(s5).execute2(((19, 182, 181), 22, (231, 232, 24), 25))


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

    laws = domain.getMaterialLawSet()
    laws.define (1, LinearIsotropicHardening)
    laws(1).put(IH_SIGEL, 1000.0)
    laws(1).put(IH_H,    1000.0)

    laws.define (2, LinearIsotropicHardening)
    laws(2).put(IH_SIGEL, 400.0)
    laws(2).put(IH_H,    1000.0)

    prp1 = ElementProperties(Volume2DElement)
    prp1.put(MATERIAL, 1)
    prp1.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)
    prp1.put(THICKNESS, 45.0)   # Set blade width


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

    app2 = FieldApplicator(2)
    app2.push(s2)
    app2.push(s3)
    app2.push(s4)
    app2.push(s5)
    app2.addProperty(prp2)
    domain.getInteractionSet().add(app2)

    # GROUPES D'ATTRIBUTS
    groupset = geometry.getGroupSet()
    groupset.add(Group(1)); groupset(1).addMeshPoint(p6)
    groupset.add(Group(2)); groupset(2).addMeshPoint(p30)
    # Generation du "maillage" ressort
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

    # Après la création du maillage, avant les matériaux
    print("Vérification des éléments...")
    mesh = geometry.getMesh()
    print(f"Nombre de points: {mesh.getPointSet().size()}")
    print(f"Nombre de courbes: {mesh.getCurveSet().size()}")

    #-----------------Modification from this point---------------------------:

    # Rotation axis for the rotation of the blade
    pa1 = pointset.define(23, enc + L/2  ,0.0)
    pa2 = pointset.define(24, enc + L/2  ,0.0, 1.0)
    axe1 = Axe(pa1, pa1)
    axe1.setSymZ1(1.0)


    # Rotation axis for the second rotation of the blade (clamping)
    # Point d'accroche vers le bas - correction des coordonnées
    pa3 = pointset.define(31, Dx1+enc, Dy, 0.0)
    pa4 = pointset.define(32, Dx1+enc, Dy, 1.0)
    axe2 = Axe(pa3, pa4)

    # axis displacement
    # The rotation axis moves during the loading so that the clamping point is
    # located at the desired location in the end. First the LF the deplacement
    # and rotation are set to find the reference equilibrium configuration,
    # then the clamping offset is added.
    fctX = PieceWiseLinearFunction()
    fctX.setData(0.0, 0.0)
    fctX.setData(T_load / 8, 0.0)
    fctX.setData(T_load / 2, 1.0)  # Pas de déplacement horizontal pour cette configuration
    fctX.setData(3 * T_load / 4, 1.0)
    fctX.setData(T_load, 1.0)
    fctX.setData(T, 1.0)

    fctY = PieceWiseLinearFunction()
    fctY.setData(0.0, 0.0)
    fctY.setData(T_load / 2, 0.0)
    fctY.setData(3 * T_load / 4, 1.0)
    fctY.setData(T_load, 1.0)
    fctY.setData(T, 1.0)

    # Appliquer les déplacements sur les deux axes
    domain.getLoadingSet().define(pa1, Field1D(TX,RE),(Dx+Dx1), fctX)
    domain.getLoadingSet().define(pa2, Field1D(TX,RE),(Dx+Dx1), fctX)
    domain.getLoadingSet().define(pa1, Field1D(TY,RE), Dy, fctY)
    domain.getLoadingSet().define(pa2, Field1D(TY,RE), Dy, fctY)

    # rotation
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


    domain.getLoadingSet().defineRot2(c1, Field3D(TXTYTZ, RE), axe2, angleClamp, fctR2, axe1,- 180, fctR, False)


    #------------------------------------------UP TO THIS POINT-------------------------------------------

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
    tsm.setNextTime(  T, 1, 0.005)  # Set min time set (useful for free oscillations)
    #tsm.setNextTime(  T, 1, 0.05)
    # To reduce the simulation time, a longer dt can be used but it affects the
    # accuracy of the free oscillations and thus the final equilibrium configuration
    # of the sensor, expecially when close to unstability. This dt must be used
    # only in the trial and error phase to find a rough estimation of the clamping
    # point location that leads to the sensor equilibrium

    # Tolerance
    fct_TOL = PieceWiseLinearFunction()
    fct_TOL.setData(0.0, 1.0)
    fct_TOL.setData(T_load, 1.0)
    fct_TOL.setData(11*T_load/10, 1/10)
    fct_TOL.setData(T, 1/10)

    mim = metafor.getMechanicalIterationManager()
    mim.setMaxNbOfIterations(10)
    mim.setResidualTolerance(1e-4, fct_TOL)
    #mim.setVerbose(True)
    mim.setPredictorComputationMethod(EXTRAPOLATION_MRUA) # 2nd deg extrapolation

    # History curves (Save data is .txt files)
    interactionset = domain.getInteractionSet()
    if not p['postpro']:
        hcurves = metafor.getValuesManager()
        hcurves.add(1, MiscValueExtractor(metafor,EXT_T), 'time')
        hcurves.add(2, NormalForceValueExtractor(ci), SumOperator(), 'ContactForceY')
        hcurves.add(3, DbNodalValueExtractor(p15, Field1D(TY,RE)), SumOperator(), 'displacement') # End rod pt displacement
        hcurves.add(4, DbNodalValueExtractor(c3, Field1D(TY,GF1)), SumOperator(), 'forceYExtClampinPt')
        hcurves.add(5, DbNodalValueExtractor(c3, Field1D(TX,GF1)), SumOperator(), 'forceXExtClampinPt')
        hcurves.add(7, DbNodalValueExtractor(p17, Field1D(TX,GF1)), SumOperator(), 'forceXHinge')
        hcurves.add(6, DbNodalValueExtractor(p17, Field1D(TY,GF1)), SumOperator(), 'forceYHinge')
        ext0 = MomentValueExtractor(c3, pa3, TZ, GF1)
        hcurves.add(8, ext0, SumOperator(), 'MomentExtClampingPt')
        ext = IFNodalValueExtractor(interactionset(1), IF_EVMS)
        hcurves.add(9, ext, MaxOperator(),'Max VonMises')

        for i in range(1,10):
            metafor.getTestSuiteChecker().checkExtractor(i)

    # plot curves during simulation
    if not p['postpro']:
        try:
            plot1 = DataCurveSet()
            plot1.add(VectorDataCurve(2, hcurves.getDataVector(1),
                                         hcurves.getDataVector(3), 'displacement'))
            win1 = VizWin()
            win1.add(plot1)
            win1.setPlotTitle(" Rod end point vertical displacement as a function of time")
            win1.setPlotXLabel("Time [s]")
            win1.setPlotYLabel("Y [mm]")
            metafor.addObserver(win1)

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