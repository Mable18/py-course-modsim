#!/usr/bin/env python
# coding: utf-8

# [<img src="../../header.svg">](../index.ipynb)
# 
# ---
# # Finite elements: Solution of the Poisson equation in UG4 
# 
# This example shows the basic steps to solve 
# $$-\triangle u = f $$
# on the unit square using Dirichlet boundary conditions. 
# 
# An extension is given in the [following example](./tutorial-fem-02.ipynb).
# 
# ## Initialization
# Loading modules:

# In[1]:


import sys

import ug4py.pyugcore as ug4
import ug4py.pyconvectiondiffusion as cd
import ug4py.pylimex as limex

sys.path.append("..")
import modsimtools as util


# ## Problem definition
# 
# The following variables are used to define the problem:

# In[2]:


myGridName= "grids/unit_square_tri.ugx" # grids/unit_square_tri.ugx",
myNumRefs= 3
mySubsets = {"Inner", "Boundary"}


# ## Computational domain
# 
# We can read the domain as follows:

# In[18]:


dom = util.CreateDomain(gridName=myGridName, numRefs=myNumRefs, requiredSubsets=mySubsets)


# ## Create ansatz space
# Create a FEM ansatz space. Here, we use piecewise linear functions(Lagrange polynomial of 1st order):

# In[5]:


approxSpaceDesc = dict(fct = "u", type = "Lagrange", order = 1)
approxSpace = util.CreateApproximationSpace(dom, approxSpaceDesc)


# In[6]:


#approxSpace.print_statistic()


# ## Discretization
# 
# The following object holds an **element-wise discretization** for the convection-diffusion equation:

# In[7]:


elemDisc = cd.ConvectionDiffusionFE2d("u", "Inner")
elemDisc.set_diffusion(1.0)

# Optionally: Set right hand side f
elemDisc.set_source(1.0)


# Create object for **boundary condiditions**:

# In[8]:


dirichletBND = ug4.DirichletBoundary2dCPU1()
dirichletBND.add(0.0, "u", "Boundary")
# dirichletBND.add(pyDirichletBndCallback, "u", "Boundary")


# Both objects contribute to the **domain discretization**, which serves as a container for the full problem:

# In[9]:


domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
domainDisc.add(elemDisc)
domainDisc.add(dirichletBND)


# ## Assemble linear system:

# In[10]:


A = ug4.AssembledLinearOperatorCPU1(domainDisc)
x = ug4.GridFunction2dCPU1(approxSpace)
b = ug4.GridFunction2dCPU1(approxSpace)

# x.clear(0.0)

domainDisc.assemble_linear(A, b)
domainDisc.adjust_solution(x)


# ## Solve linear system

# In[11]:


try:
    import ug4py.pysuperlu as slu
    solver=slu.SuperLUCPU1()
except ImportError:
    solver=ug4.LUCPU1()

solver.init(A, x)
solver.apply(x, b)


# ## Output of results
# 
# Results can be visualized using Paraview/pyvista (\*.vtu) as well as using UG4's ConnectionViewer (\*.vec):

# In[12]:


try:
    import pyvista
    #pyvista.start_xvfb()
    #pyvista.set_jupyter_backend('trame')
    pyvista.set_jupyter_backend('static')

except Exception:
    pyvista=None


# a) Solution u

# In[13]:


solFileName = "tmp/fem01_solution_u"
ug4.WriteGridFunctionToVTK(x, solFileName)
ug4.SaveVectorForConnectionViewer(x, solFileName + ".vec")


# In[14]:


if pyvista is not None:
    result = pyvista.read(solFileName + ".vtu")
    result.plot(scalars="u", show_edges=True, cmap='jet')


# b) Right hand side $b$ and matrix $A$

# In[15]:


solFileName = "tmp/fem01_rhs_b"
ug4.WriteGridFunctionToVTK(b, solFileName)
ug4.SaveVectorForConnectionViewer(b, solFileName + ".vec")
ug4.SaveMatrixForConnectionViewer(x, A, "tmp/fem01_matrix_A.mat")


# In[16]:


if pyvista is not None:
    result = pyvista.read(solFileName + ".vtu")
    result.plot(scalars="u", show_edges=True, cmap='jet')


# In[ ]:







# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




