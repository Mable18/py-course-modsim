#!/usr/bin/env python
# coding: utf-8

# Zellen haben eine komplexe Binnenstruktur. In dieser Aufgabe betrachten wir schematisch einen Transportprozess innerhalb einer idealisierten Zelle. Dabei soll eine Substanz per Diffusion vom Zellkern (blau) zur Zellmembran (rot) transportiert werden. Es gibt hierfür einen durchgehenden Weg (hellgrau), allerdings gibt es Bereiche (dunkelgrau), in denen die Diffusion, z.B. durch bestehende Strukturen, stark verändert wird.
# 
# ![test](simple-cell.png)
# 
# Wir betrachten die Konzentration $u$ einer Substanz und nehmen an, dass sich die Verteilung durch die Diffusionsgleichung
# $$ \frac{\partial u}{\partial t} = \nabla \cdot \left( -\alpha \nabla u \right) = 0 $$
# beschreiben lässt. Die Diffusionskonstante $\alpha$ ist dabei ortsabhängig. Sie beträgt $\alpha = 1 mm^2/s$ im (hellgrauen) freien Bereich sowie $\alpha = \epsilon = 10^{-3} mm^2/s$ im (dunkelgrauen) Bereich mit Barriere.  Die Randbedingungen sind so, dass am Zellkern wird die Substanz mit der Konzentration $1\;mmol$ angeliefert wird. An der Zellmembran  wird die Substanz vollständig abgeführt wird, d.h., die Konzentration dort betrage $0\;mmol$.
# Ermitteln Sie numerisch den Konzentrationsverlauf im Gleichgewicht. Dazu können Sie Ihre Lösung z.B. als vtu Datei exportieren und mittels Paraview visualisieren.
# 

# Als erstes importieren wir die Python Distribution von UG4:

# In[3]:


import ug4py.pyugcore as ug4
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd
import sys


# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')
import modsimtools as util


# Nun erstellen wir das Simulationsgebiet (Domain)

# In[5]:


requiredSubsets = ["FREE", "NUCLEUS", "MEMBRANE", "BARRIER"]
gridName = "simple-cell.ugx"
numRefs = 2
dom = util.CreateDomain(gridName, numRefs, requiredSubsets)


# Auf dem Gebiet definieren wir einen Ansatzraum. Hier werden unsere Unbekannten, welche auf den Gitterknoten definiert sind, durch Interpolationsfunktionen auf das gesamte Simulationsgebiet erweitert (diskrete Werte an Knoten -> Feld auf dem Gebiet).

# In[9]:


approxSpace = ug4.ApproximationSpace2d(dom)
approxSpace.add_fct("u", "Lagrange", 1)
# enable multigrid
approxSpace.init_levels()
approxSpace.init_surfaces()
approxSpace.init_top_surface()
approxSpace.print_statistic()


# Nun kümmern wir uns um die Beschreibung unserer Differentialgleichung. Dafür Diskretisieren wir elementweise unsere PDE. Im Modul **ConvectionDiffusion** steht dafür eine Art Baukasten zur Verfügung. Im allgemeinen können hier PDE's der Form
# 
# $$\partial_t (m_1 c + m_2) - \nabla \cdot \left ( D \nabla c - \vec{v} c - \vec{F} \right ) + r_1 \cdot c + r_2 = f + \nabla \cdot \vec{f}_2$$
# 
# implementiert werden. Jedem Term in obiger Gleichung kann ein Wert oder eine Funktion zugewiesen werden. Betrachten wir unsere PDE
# $$ \frac{\partial u}{\partial t} = \nabla \cdot \left( -\alpha \nabla u \right) = 0 $$
# müssen wir also $m_1 = 1$ und $D = \alpha$ setzen (Rest ist 0!).
# 

# In[27]:


alpha_barrier   = 0.00001
alpha_free      = 1.0
m1              = 1

# create element discretizations for each subdomain
elemDisc = {}
elemDisc["BARRIER"] = cd.ConvectionDiffusionFV12d("u", "BARRIER")
elemDisc["FREE"]    = cd.ConvectionDiffusionFV12d("u", "FREE")

# set mass scale
elemDisc["BARRIER"].set_mass_scale(m1)
elemDisc["FREE"].set_mass_scale(m1)

# set diffusion coefficients
elemDisc["BARRIER"].set_diffusion(alpha_barrier)
elemDisc["FREE"].set_diffusion(alpha_free)


# Betrachten wir nun unsere Randbedingungen. In der Aufgabenstellung sind feste Werte für die Konzentration unserer Unbekannten gegeben. Deshalb müssen wir für unser Problem Dirichlet Randbedingungen am Nukleus und an der Membran definieren.

# In[28]:


dirichletBND = ug4.DirichletBoundary2dCPU1()
dirichletBND.add(1.0, "u", "NUCLEUS")
dirichletBND.add(0.0, "u", "MEMBRANE")


# Nun haben wir unser Problem vollständig definiert. Wir haben Diskretisierungen der partiellen Differentialgleichung auf den Elementen und Randbedingungen am Rand. in UG4 müssen wir diese zu einer Gebietsdiskretisierung zusammenfassen, welche das vollständige Problem beschreibt..

# In[29]:


domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
domainDisc.add(elemDisc["FREE"])
domainDisc.add(elemDisc["BARRIER"])
domainDisc.add(dirichletBND)


# Wir werden das zeitabhängige Problem mit einem impliziten Eulerverfahren lösen. Unsere Unbekannten werden in einer sogenannten **Gitterfunktion** gespeichert. Über diese interpolieren wir einen Anfangswert und erhalten so ein vollständig definiertes Anfangswertproblem

# In[30]:


# Create Solver and Time Discretization
lsolver=ug4.LUCPU1()
startTime = 0.0
endTime = 500.0
dt = 1
timeDisc=ug4.ThetaTimeStepCPU1(domainDisc, 1.0)
timeInt = limex.ConstStepLinearTimeIntegrator2dCPU1(timeDisc)
timeInt.set_linear_solver(lsolver)
timeInt.set_time_step(dt)
# Create GridFunction
usol = ug4.GridFunction2dCPU1(approxSpace)
# interpolate initial value
ug4.Interpolate(0.0, usol, "u")


# Starten des Lösungsprozesses:

# In[31]:


def MyVTKCallback(usol, step, time, dt) :
    ug4.WriteGridFunctionToVTK(usol, "vtk/SimpleCell_"+str(int(step)).zfill(5)+".vtu")

vtkobserver = ug4.PythonCallbackObserver2dCPU1(MyVTKCallback)
timeInt.attach_observer(vtkobserver)
try:
    timeInt.apply(usol, endTime, usol, startTime)
except Exception as inst:
    print(inst)



# In[20]:


ug4.WriteGridFunctionToVTK(usol, "vtk/cell_final.vtu")


# In[ ]:
f_end = ug4.IntegrateNormalGradientOnManifold(usol, "u", "MEMBRANE", "FREE")
print("flux"+str(f_end))





# %%
