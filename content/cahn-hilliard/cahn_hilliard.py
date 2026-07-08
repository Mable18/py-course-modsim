#!/usr/bin/env python
# coding: utf-8

import sys
import os
import math
import random

random.seed(42)  # for reproducibility

sys.path.append("..")

import modsimtools as util
import ug4py.pyugcore as ug4
import ug4py.pyconvectiondiffusion as cd
import ug4py.pylimex as limex


# Grid and domain
# gridName = "grids/laplace_sample_grid_2d_l50.ugx"
gridName = "grids/fish_grid_with_eye.ugx"
numRefs = 3
mandatorySubsets = ["Inner", "Boundary"]

# Create domain (modsimtools handles loading and refinement)
dom = util.CreateDomain(gridName, numRefs, mandatorySubsets)


# Approximation space: two unknowns `mu` and `c` (piecewise linear)
approxSpace = ug4.ApproximationSpace2d(dom)
approxSpace.add_fct("mu", "Lagrange", 1)
approxSpace.add_fct("c", "Lagrange", 1)
approxSpace.init_levels()
approxSpace.init_top_surface()
approxSpace.print_statistic()


# Parameters
class CahnHilliardModel:
    M = 10000.0
    gamma = 2.0    # ~ layer thickness
    cinf = 1.0
    f0 = lambda c: 0.25*(c**2 - CahnHilliardModel.cinf**2)**2
    f1 = lambda c: c**3 - CahnHilliardModel.cinf * c
    f2 = lambda c: 3*c**2 - CahnHilliardModel.cinf
    ka = 0.03
    ki = 0.1 # \approx 3*ka, to have a non-trivial steady state at c = 0.25 (since ka*(1-c) - ki*c = 0 => c = ka/(ka+ki) = 0.01/0.04 = 0.25)




# Create element-wise discretizations (FE)
elemDisc = {}
elemDisc["mu"] = cd.ConvectionDiffusionFE2d("mu", "Inner")
elemDisc["c"] = cd.ConvectionDiffusionFE2d("c", "Inner")




# Eqn 1: \partial_t c - M \triangle \mu = q(c)
elemDisc["mu"].set_mass_scale(0.0)
elemDisc["mu"].set_mass(elemDisc["c"].value())
elemDisc["mu"].set_diffusion(CahnHilliardModel.M)
elemDisc["mu"].set_reaction(0.0)
elemDisc["mu"].set_source(0.0)


# Optional: Add a source term q(c) to the first equation, e.g., to model production/degradation of the species.
# Turing source term: q(c) = ka * (1 - c) - ki * c
if False:   
    qc = ug4.PythonUserFunction2d(lambda c: CahnHilliardModel.ka * (1 - c) - CahnHilliardModel.ki * c, 1)   
    qc.set_input_and_deriv(0, elemDisc["c"].value(), lambda c: -CahnHilliardModel.ki - CahnHilliardModel.ka)  # Derivative with respect to c is -ki - ka
    elemDisc["mu"].set_source(qc)


# Eqn 2: \mu = f'(c) - \lambda \triangle c

# create Python user function and bind input/derivative to elemDisc["c"].value()
fPrime = ug4.PythonUserFunction2d(CahnHilliardModel.f1, 1)  # f'(c) = c^3 - c
fPrime.set_input_and_deriv(0, elemDisc["c"].value(), CahnHilliardModel.f2)  # f''(c) = 3c^2 - 1

elemDisc["c"].set_stationary()
elemDisc["c"].set_diffusion(CahnHilliardModel.gamma)
elemDisc["c"].set_reaction(fPrime)
elemDisc["c"].set_source(elemDisc["mu"].value())



# Domain discretization
domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
domainDisc.add(elemDisc["mu"])
domainDisc.add(elemDisc["c"])


# Debug writer (optional)
try:
    dbgWriter = ug4.GridFunctionDebugWriter2dCPU1(approxSpace)
except Exception:
    dbgWriter = None


# Solution gridfunction and initialization
u = ug4.GridFunction2dCPU1(approxSpace)


# u.set(0.0)
# Initialize with random perturbation around 0.33 for c, and 0 for mu, as in original Lua

# TODO: Determine the (three) steady states? 
# Which are unstable and which are stable?

# Initial values
def MyInitialValueMu(x, y, t, si):
    return 0.0

def MyInitialValueC(x, y, t, si):
    return 0.01 * (random.random()-0.5) 

ug4.Interpolate(ug4.PythonUserNumber2d(MyInitialValueMu), u, "mu")
ug4.Interpolate(ug4.PythonUserNumber2d(MyInitialValueC), u, "c")



# TODO: Determine characteristic time scale and adjust endTime accordingly.

startTime = 0.0
charLength = 100.0  # Assuming a characteristic length scale of 100.0
charTimeDiff = charLength**4/(CahnHilliardModel.M * CahnHilliardModel.gamma)  # Time scale for Cahn-Hilliard
charTimeFeat = charLength**3 # Time scale based on feature evolution,
# assuming a characteristic velocity of 1.0 (this is a rough estimate and can be adjusted based on the specific problem setup)

print (f"Characteristic time scale based on diffusion: {charTimeDiff}")
print (f"Characteristic time scale based on reaction: {charTimeFeat}")

dt = min(charTimeDiff, charTimeFeat)*1e-4
endTime = max(charTimeDiff, charTimeFeat)*100





limexDesc = util.GetLimexDefaultDesc()
print("LIMEX config:")
print(limexDesc)
limexDesc["TOL"]=1e-2
limex = util.CreateLimexIntegrator(domainDisc, limexDesc, dt)

limex.set_dt_min(dt*1e-3)
limex.set_dt_max((endTime-startTime)/50)

# Create callback observer for measuring mass.
ti = []
mi = []
ji = []
def MyMassCallback(usol, step, time, dt) :
    m=ug4.Integral(usol, "c", "Inner") 
    j=ug4.H1SemiNorm(usol, "mu", 4, "Inner")
    j=-CahnHilliardModel.M*j*j
    print (f"Step {step}, time {time:.4f}, mass {m:.6f}, energy change {j:.6f}") 
 
    ti.append(time)
    mi.append(m)
    ji.append(j)
massobserver = ug4.PythonCallbackObserver2dCPU1(MyMassCallback) 
limex.attach_observer(massobserver)



# Create callback observer for file I/O.
def MyVTKCallback(usol, step, time, dt) :
    ug4.WriteGridFunctionToVTK(usol, "vtk/CahnHilliardFish"+str(int(step)).zfill(5)+".vtu")
vtkobserver = ug4.PythonCallbackObserver2dCPU1(MyVTKCallback) 
limex.attach_observer(vtkobserver)


# Run problem!
limex.apply(u, endTime, u, startTime)


if True:
    import numpy as np
    import matplotlib.pyplot as plt

    plt.plot(ti, ji, 'o-')  # circles connected by lines
    plt.xlabel('time')
    plt.ylabel('mass')
    plt.grid()
    plt.show()


print("done")
