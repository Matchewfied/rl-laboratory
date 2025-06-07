import sys
from ..base import BaseCustomEnv
from .dynamics import *

"""
Class to define first ceviche environment.
"""
class CevicheEnv(BaseCustomEnv):
    def __init__(self,
                 omega1=2*np.pi*200e12,  # The two angular frequencies
                 omega2=2*np.pi*230e12,
                 dl=40e-9,               # Spatial resolution in meters
                 Nx=120,                 # Number of pixels in x-direction
                 Ny=120,                 # Number of pixels in y-direction
                 Npml=20,                # Number of pixels in the PMLs in each direction
                 epsr_min=1.0,           # Minimum value of the relative permittivity
                 epsr_max=12.0,          # Maximum value of the relative permittivity
                 blur_radius=2,          # Radius of the smoothening features
                 N_blur=1,               # Number of times to apply the blur
                 beta=10.0,              # Strength of the binarizing projection
                 eta=0.5,                # Middle point of the binarizing projection
                 N_proj=1,               # Number of times to apply the blur
                 space=10,               # Space between the PMLs and the design region (in pixels)
                 wg_width=12,            # Width of the waveguide (in pixels)
                 space_slice=8,          # Length in pixels of the source/probe slices on each side of the center point
                 # action_space = Box(low=0.0, high=1.0, shape=(3600,), dtype=np.float32) # Updates the entire design region, returns updated density values
                 ):
        super().__init__()
        self.omega1, self.omega2 = omega1, omega2
        self.dl = dl
        self.Nx, self.Ny, self.Npml = Nx, Ny, Npml

        self.epsr_min, self.epsr_max = epsr_min, epsr_max
        self.blur_radius, self.N_blur = blur_radius, N_blur

        self.beta, self.eta = beta, eta

        self.N_proj = N_proj

        self.space, self.wg_width, self.space_slice = space, wg_width, space_slice

        self.rho, self.bg_rho, self.design_region, self.input_slice, self.output_slice1, self.output_slice2 = init_domain(Nx, Ny, Npml, 
                                                                                                                          space, wg_width, space_slice)
        
        epsr_init = epsr_parameterization(self.rho, self.bg_rho, self.design_region, 
                                          radius=blur_radius, N_blur=N_blur, beta=beta, 
                                          eta=eta, N_proj=N_proj)

        self.source1 = insert_mode(omega1, dl, 
                                   self.input_slice.x, 
                                   self.input_slice.y,
                                   epsr_init, m=1)

        self.source2 = insert_mode(omega2, dl, 
                                   self.input_slice.x, 
                                   self.input_slice.y,
                                   epsr_init, m=1)
        
        self.probe1 = insert_mode(omega1, dl, 
                                  self.output_slice1.x, self.output_slice1.y, 
                                  epsr_init, m=1)

        self.probe2 = insert_mode(omega2, dl, 
                                  self.output_slice2.x, self.output_slice2.y, 
                                  epsr_init, m=1)
        
        self.simulation1, self.simulation2 = sim(epsr_init, omega1, omega2, dl, 
                                                Npml, self.source1, self.source2, 
                                                slices=[self.input_slice, 
                                                        self.output_slice1, 
                                                        self.output_slice2])
        
        _, _, Ez1 = self.simulation1.solve(self.source1)
        _, _, Ez2 = self.simulation2.solve(self.source2)

        self.E01 = mode_overlap(Ez1, self.probe1)
        self.E02 = mode_overlap(Ez2, self.probe2)
    
    # Assume action is a list of two vectors
    # Action: List of two arrays: 
    #   Array 01: All of the changes we want to add to the current grid. 
    #               With 10% change, yields to 360 values. 
    #   Array 02: List of indices. With current setup, consists of 3600 values. 
    #               Don't worry about it, it will handle the math. 

    def step(self, action):
        """
        Takes in an action. 
        Returns: small_rho -> next state
                 reward -> reward associated with the next state
        """
        # properly break up the action
        new_densities, indices = action

        small_rho = extract_small_rho(self.rho, self.Nx, self.Ny, self.Npml, self.space)
        small_rho[indices] = new_densities
        self.rho = load_rho(self.rho, small_rho, self.Nx, self.Ny, self.Npml, self.space)

        reward = objective(self.rho, self.bg_rho, self.design_region,
                           self.blur_radius, self.N_blur, self.beta, self.eta,
                           self.N_proj, self.source1, self.source2,
                           self.probe1, self.probe2, 
                           self.simulation1, self.simulation2,
                           self.E01, self.E02)

        return small_rho, reward

    def reset(self):
        self.rho, 
        self.bg_rho, 
        self.design_region, 
        self.input_slice,
        self.output_slice1,
        self.output_slice2 = init_domain(self.Nx, self.Ny, self.Npml, 
                                        self.space, self.wg_width, 
                                        self.space_slice)

        return extract_small_rho(self.rho, self.Nx, self.Ny, self.Npml, self.space)
    
    def render(self):
        pass

    def print_state(self):
        print("Print state called")
        #print(extract_small_rho(self.rho, self.Nx, self.Ny, self.Npml, self.space))
        #sys.stdout.flush()

    def set_state(self, state):
        self.rho = load_rho(self.rho, state, self.Nx, self.Ny, self.Npml, self.space)
        return extract_small_rho(self.rho, self.Nx, self.Ny, self.Npml, self.space)


def load_rho(rho, to_load, Nx, Ny, Npml, space):
    design_x = Nx - Npml - space - (Npml + space)
    design_y = Ny - Npml - space - (Npml + space)
    rho[Npml + space:Nx - Npml - space, Npml + space:Ny - Npml - space] = to_load.reshape((design_x, design_y))
    return rho


def extract_small_rho(rho, Nx, Ny, Npml, space):
    design_x = Nx - Npml - space - (Npml + space)
    design_y = Ny - Npml - space - (Npml + space)
    return rho[Npml + space:Nx - Npml - space,
                Npml + space:Ny - Npml - space].reshape((design_x * design_y,))
