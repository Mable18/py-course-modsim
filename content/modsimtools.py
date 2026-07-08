import ug4py.pyugcore as ug4
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd
# import ug4py.pysuperlu as slu

def CreateDomain(gridName, numRefs, requiredSubsets):
    # Choosing a domain object
    # (either 1d, 2d or 3d suffix)
    dom = ug4.Domain2d()

    # Loading the given grid into the domain
    print("Loading Domain '" + gridName +"'...")
    ug4.LoadDomain(dom, gridName)
    print("Domain loaded.")
    
    
    # Optional: Refining the grid
    if numRefs > 0:
        print("Refining ...")
        refiner = ug4.GlobalDomainRefiner(dom)
        for i in range(numRefs):
            ug4.TerminateAbortedRun()
            refiner.refine()
            print("Refining step {" + str(i) +"} ...")

        print("Refining done")

    # checking if geometry has the needed subsets of the probelm
    sh = dom.subset_handler()
    for e in requiredSubsets:
        if sh.get_subset_index(e) == -1:
            print(f"Domain does not contain subset {e}.")
            sys.exit(1)
    
    return dom

def CreateApproximationSpace(dom, approxSpaceDesc):
    approxSpace = ug4.ApproximationSpace2d(dom)
    approxSpace.add_fct(approxSpaceDesc["fct"], approxSpaceDesc["type"], approxSpaceDesc["order"])
    approxSpace.init_levels()
    approxSpace.init_top_surface()
    return approxSpace

def CreateDiffusionElemDisc(fname, subdom, mass_scale, diffusion, reaction):
    elemDisc = cd.ConvectionDiffusionFV12d(fname, subdom)
    elemDisc.set_mass_scale(mass_scale)
    elemDisc.set_diffusion(diffusion)
    elemDisc.set_reaction(reaction)
    return elemDisc


# Create a LIMEX time disc.
limexDefaultDesc = {
        "nstages": 2,
        "lsolver" : ug4.LUCPU1(), 
        "TOL": 1e-3,
        "metricSpace": None
}

# Get default LIMEX time disc.
def GetLimexDefaultDesc():
    return limexDefaultDesc

# Create a LIMEX time disc.
def CreateLimexIntegrator(domainDisc, limexDesc=limexDefaultDesc, dt=1.0, dtmin=None, dtmax=None):

    nstages = limexDesc["nstages"] 
    lsolver = limexDesc["lsolver"]
    TOL     = limexDesc["TOL"]  
    metricSpace = limexDesc["metricSpace"]

    if (dtmin is None):
        dtmin=dt*1e-3

    if (dtmax is None):
        dtmax=dt*1e+2   

    if (nstages<2):
        print("Using implicit Euler (nstages=1).")
        return None
    
    # LIMEX config.
    timeInt = limex.LimexTimeIntegrator2dCPU1(nstages)
    nlsolver = limex.LimexNewtonSolverCPU1()
    nlsolver.set_linear_solver(lsolver)
    for i in range(nstages):
        timeInt.add_stage(i+1, nlsolver, domainDisc)

    # Time stepping config.
    timeInt.set_time_step(dt)
    timeInt.set_dt_min(dtmin)
    timeInt.set_dt_max(dtmax)
    timeInt.set_increase_factor(1.5) # max. Faktor, um den dt erhöht werden darf
    timeInt.disable_matrix_cache()

    # LIMEX w/ error estimation
    timeInt.set_tolerance(TOL)

    # Definition of error estimator.
    errorEst = None
    if metricSpace is not None:
        print("Using custom metric space for error estimation.")
        errorEst = limex.CompositeGridFunctionEstimator2dCPU1()
        errorEst.add(metricSpace)
    else:
        print("Using  euclidean norm for error estimation.")
        errorEst = limex.Norm2EstimatorCPU1() # Euclidean norm.
    
    timeInt.add_error_estimator(errorEst)
    return timeInt
    
