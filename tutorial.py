# -*- coding: utf-8 -*-

# Metafor tutorial

from wrap import *
import math

metafor = Metafor()
domain = metafor.getDomain()

def getMetafor(p={}): 
    return metafor

# Parameters

Lx = 1.0  # length X
Ly = 1.0  # length Y
R  = Lx/2
nx = 15   # nb of elems on X
ny = 15   # nb of elems on Y

D = 0.4      # max displacement
v = 1000.0   # punch velocity

# Geometry

geometry = domain.getGeometry()
geometry.setDimPlaneStrain(1.0)

pointset = geometry.getPointSet()

pointset.define(1, 0, 0)
pointset.define(2, Lx, 0)
pointset.define(3, Lx, Ly)
pointset.define(4, 0, Ly)

pointset.define(5, R, Ly+R)
pointset.define(6, R*math.cos(math.pi/4),
                   Ly+R-R*math.sin(math.pi/4))
pointset.define(7, -R, Ly+R)

curveset = geometry.getCurveSet()
curveset.add(Line(1, pointset(1), pointset(2)))
curveset.add(Line(2, pointset(2), pointset(3)))
curveset.add(Line(3, pointset(3), pointset(4)))
curveset.add(Line(4, pointset(4), pointset(1)))
curveset.add(Arc(5, pointset(7), pointset(6), pointset(5)))

wireset = geometry.getWireSet()
wireset.add( Wire(1, [curveset(1), curveset(2), curveset(3), curveset(4)]) )

sideset = geometry.getSideSet()
sideset.add(Side(1,[wireset(1)]))

if 0:
    win = VizWin()
    win.add(pointset)
    win.add(curveset)
    win.open()
    input()

# Mesh

SimpleMesher1D(curveset(1)).execute(nx)
SimpleMesher1D(curveset(2)).execute(ny)
SimpleMesher1D(curveset(3)).execute(nx)
SimpleMesher1D(curveset(4)).execute(ny)

TransfiniteMesher2D(sideset(1)).execute(True)

if 0:
    win = VizWin()
    win.add(geometry.getMesh().getPointSet())
    win.add(geometry.getMesh().getCurveSet())
    win.add(curveset)
    win.open()
    input()

# Volumic material/prp/elements

materials = domain.getMaterialSet()
materials.define (1, EvpIsoHHypoMaterial)
materials(1).put(MASS_DENSITY,     8.93e-9)
materials(1).put(ELASTIC_MODULUS, 200000.0)
materials(1).put(POISSON_RATIO,        0.3)
materials(1).put(YIELD_NUM,              1)

laws = domain.getMaterialLawSet()
laws.define (1, LinearIsotropicHardening)
laws(1).put(IH_SIGEL, 400.0)
laws(1).put(IH_H,    1000.0)

prp1 = ElementProperties(Volume2DElement)
prp1.put(MATERIAL, 1)
prp1.put(CAUCHYMECHVOLINTMETH, VES_CMVIM_SRIPR)

app = FieldApplicator(1)
app.push(sideset(1))
app.addProperty(prp1)
domain.getInteractionSet().add(app)

# Prescribed displacements

domain.getLoadingSet().define(curveset(4), Field1D(TX,RE), 0.0)
domain.getLoadingSet().define(curveset(1), Field1D(TY,RE), 0.0)

# Tool displacement

T = D/v

fct = PieceWiseLinearFunction()
fct.setData(0.0, 0.0)
fct.setData(T, 1.0)
fct.setData(2*T, 0.5)

domain.getLoadingSet().define(curveset(5), Field1D(TY,RE), -D, fct)

# Contact material/prp/elements

# contact
materials.define (2, CoulombContactMaterial)
materials(2).put(PEN_NORMALE,   1e6)
materials(2).put(PEN_TANGENT,   1e5)
materials(2).put(PROF_CONT,     0.1)
materials(2).put(COEF_FROT_DYN, 0.05)
materials(2).put(COEF_FROT_STA, 0.05)

prp2 = ElementProperties(Contact2DElement)
prp2.put(MATERIAL, 2)
prp2.put(AREAINCONTACT, AIC_ONCE)

ci = RdContactInteraction(2)
ci.setTool(curveset(5))
ci.push(curveset(3))
ci.addProperty(prp2)
domain.getInteractionSet().add(ci)

# Time integration scheme

ti = AlphaGeneralizedTimeIntegration(metafor)
metafor.setTimeIntegration(ti)

tsm = metafor.getTimeStepManager()
tsm.setInitialTime(0.0, 0.01)
tsm.setNextTime(  T, 5, 0.1)
tsm.setNextTime(2*T, 5, 0.1)

mim = metafor.getMechanicalIterationManager()
mim.setResidualTolerance(1.0e-4)

# History curves

hcurves = metafor.getValuesManager()
hcurves.add(1, MiscValueExtractor(metafor,EXT_T),'time')
hcurves.add(2, DbNodalValueExtractor(curveset(1), Field1D(TY,GF2)), SumOperator(), 'force')

# plot curves during simulation

try:
    plot1 = DataCurveSet()
    plot1.add(VectorDataCurve(1, hcurves.getDataVector(1), 
                                 hcurves.getDataVector(2), 'Fy'))
    win1 = VizWin()
    win1.add(plot1)
    win1.setPlotTitle("Vertical reaction force as a function of time")
    win1.setPlotXLabel("Time [s]")
    win1.setPlotYLabel("F [N/mm]")    
    metafor.addObserver(win1)
except:
    pass
