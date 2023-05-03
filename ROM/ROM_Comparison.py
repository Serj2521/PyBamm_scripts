import pybamm
import numpy as np
import matplotlib.pyplot as plt

full_model = pybamm.BaseModel(name="full model")
reduced_model = pybamm.BaseModel(name="reduced model")
models = [full_model, reduced_model]
R = pybamm.Parameter("Particle radius [m]")
D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
j = pybamm.Parameter("Interfacial current density [A.m-2]")
F = pybamm.Parameter("Faraday constant [C.mol-1]")
c0 = pybamm.Parameter("Initial concentration [mol.m-3]")
c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")
c_av = pybamm.Variable("Average concentration [mol.m-3]")
# governing equations for full model
N = -D * pybamm.grad(c)  # flux
dcdt = -pybamm.div(N)
full_model.rhs = {c: dcdt} 

# governing equations for reduced model
dc_avdt = -3 * j / R / F
reduced_model.rhs = {c_av: dc_avdt} 

# initial conditions (these are the same for both models)
full_model.initial_conditions = {c: c0}
reduced_model.initial_conditions = {c_av: c0}

# boundary conditions (only required for full model)
lbc = pybamm.Scalar(0)
rbc = -j / F / D
full_model.boundary_conditions = {c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}

# full model
full_model.variables = {
    "Concentration [mol.m-3]": c,
    "Surface concentration [mol.m-3]": pybamm.surf(c),
    "Average concentration [mol.m-3]": pybamm.r_average(c),
}

# reduced model
reduced_model.variables = {
    "Concentration [mol.m-3]": pybamm.PrimaryBroadcast(c_av, "negative particle"),
    "Surface concentration [mol.m-3]": c_av,  # in this model the surface concentration is just equal to the scalar average concentration 
    "Average concentration [mol.m-3]": c_av,
}

param = pybamm.ParameterValues(
    {
        "Particle radius [m]": 10e-6,
        "Diffusion coefficient [m2.s-1]": 3.9e-14,
        "Interfacial current density [A.m-2]": 1.4,
        "Faraday constant [C.mol-1]": 96485,
        "Initial concentration [mol.m-3]": 2.5e4,
    }
)

# geometry
r = pybamm.SpatialVariable("r", domain=["negative particle"], coord_sys="spherical polar")
geometry = {"negative particle": {r: {"min": pybamm.Scalar(0), "max": R}}}
param.process_geometry(geometry)

# models
for model in models:
    param.process_model(model)

    # mesh
submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
var_pts = {r: 20}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

# discretisation
spatial_methods = {"negative particle": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)

# process models
for model in models:
    disc.process_model(model);

# loop over models to solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 600)
solutions = [None] * len(models)  # create list to hold solutions


solver_1 = pybamm.ScipySolver()
solver_2 = pybamm.ScipySolver()
t_eval = np.linspace(0, 3600, 600)
# for model in models:
solution_1 = solver_1.solve(models[0], t_eval)
solution_2 = solver_2.solve(models[1], t_eval)

# post-process the solution of the full model
c_full = solution_1["Concentration [mol.m-3]"]
c_av_full = solution_1["Average concentration [mol.m-3]"]


# post-process the solution of the reduced model
c_reduced = solution_2["Concentration [mol.m-3]"]
c_av_reduced = solution_2["Average concentration [mol.m-3]"]

# plot
r = mesh["negative particle"].nodes # radial position

def plot(t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    
    # Plot concetration as a function of r
    ax1.plot(r * 1e6, c_full(t=t,r=r), label="Full Model")
    ax1.plot(r * 1e6, c_reduced(t=t,r=r), label="Reduced Model")    
    ax1.set_xlabel("Particle radius [microns]")
    ax1.set_ylabel("Concentration [mol.m-3]")
    ax1.legend()
    
    # Plot average concentration over time
    t_hour = np.linspace(0, 3600, 600)  # plot over full hour
    c_min = c_av_reduced(t=3600) * 0.98  # minimum axes limit 
    c_max = param["Initial concentration [mol.m-3]"] * 1.02   # maximum axes limit 
    
    ax2.plot(t_hour, c_av_full(t=t_hour), label="Full Model")
    ax2.plot(t_hour, c_av_reduced(t=t_hour), label="Reduced Model") 
    ax2.plot([t, t], [c_min, c_max], "k--")  # plot line to track time
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Average concentration [mol.m-3]") 
    ax2.legend()

    plt.tight_layout()
    plt.show()
                   
import ipywidgets as widgets
widgets.interact(plot, t=widgets.FloatSlider(min=0,max=3600,step=1,value=0));
   