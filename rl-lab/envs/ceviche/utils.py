import numpy as np
import matplotlib as mpl
import autograd.numpy as npa
import copy
import matplotlib.pylab as plt
from autograd.scipy.signal import convolve as conv
from skimage.draw import disk
import ceviche
from ceviche import fdfd_ez, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode
import collections

Slice = collections.namedtuple('Slice', 'x y')

# The two angular frequencies
omega1=2*np.pi*200e12
omega2=2*np.pi*230e12
# Spatial resolution in meters
dl=40e-9
# Number of pixels in x-direction
Nx=120
# Number of pixels in y-direction
Ny=120
# Number of pixels in the PMLs in each direction
Npml=20
# Minimum value of the relative permittivity
epsr_min=1.0
# Maximum value of the relative permittivity
epsr_max=12.0
# Radius of the smoothening features
blur_radius=2
# Number of times to apply the blur
N_blur=1
# Strength of the binarizing projection
beta=10.0
# Middle point of the binarizing projection
eta=0.5
# Number of times to apply the blur
N_proj=1
# Space between the PMLs and the design region (in pixels)
space=10
# Width of the waveguide (in pixels)
wg_width=12
# Length in pixels of the source/probe slices on each side of the center point
space_slice=8

# Number of epochs in the optimization
Nsteps=100
# Step size for the Adam optimizer
step_size=1e-3

def set_parameters (parameters: dict) -> None:
    if 'omega1' in parameters:
        omega1 = parameters['omega1']
    if 'omega2' in parameters['omega2']:
        omega2 = parameters['omega2']
    if 'dl' in parameters['dl']:
        dl = parameters['dl']
    if 'Nx' in parameter
    Nx=120
# Number of pixels in y-direction
Ny=120
# Number of pixels in the PMLs in each direction
Npml=20
# Minimum value of the relative permittivity
epsr_min=1.0
# Maximum value of the relative permittivity
epsr_max=12.0
# Radius of the smoothening features
blur_radius=2
# Number of times to apply the blur
N_blur=1
# Strength of the binarizing projection
beta=10.0
# Middle point of the binarizing projection
eta=0.5
# Number of times to apply the blur
N_proj=1
# Space between the PMLs and the design region (in pixels)
space=10
# Width of the waveguide (in pixels)
wg_width=12
# Length in pixels of the source/probe slices on each side of the center point
space_slice=8

# Number of epochs in the optimization
Nsteps=100
# Step size for the Adam optimizer
step_size=1e-3
    