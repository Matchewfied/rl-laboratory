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

def init_domain(Nx, Ny, Npml, space=10, wg_width=10, space_slice=5, wg_shift=9):
    """Initializes the domain and design region

    space       : The space between the PML and the structure
    wg_width    : The feed and probe waveguide width
    space_slice : The added space for the probe and source slices
    """
    rho = np.zeros((Nx, Ny))
    bg_rho = np.zeros((Nx, Ny))

    design_region = np.zeros((Nx, Ny))

    # Input waveguide
    bg_rho[0:Npml + space, 
            (Ny // 2) - (wg_width // 2):(Ny // 2) + (wg_width // 2)] = 1

    # Input probe slice
    input_slice = Slice(x=np.array(Npml + 1),
                        y=np.arange((Ny // 2) - (wg_width // 2) -space_slice, 
                                    (Ny // 2) + (wg_width // 2) + space_slice))

    # Output waveguide 1
    # Matthew's Note: Don't think casting as int is necessary
    bg_rho[int(Nx - Npml - space)::,
            Npml + space + wg_shift:Npml + space + wg_width + wg_shift] = 1

    # Output waveguide 2
    bg_rho[int(Nx - Npml - space)::,
            Ny - Npml - space - wg_width - wg_shift:Ny - Npml - space - wg_shift] = 1

    # Output probe slice 1
    output_slice1 = Slice(x=np.array(Nx - Npml - 1),
                            y=np.arange(Npml + space - space_slice + wg_shift, 
                                        Npml + space + wg_width + space_slice + wg_shift))

    # Output probe slice 2
    output_slice2 = Slice(x=np.array(Nx - Npml - 1),
                            y=np.arange(Ny - Npml - space - wg_width - wg_shift - space_slice, 
                                        Ny - Npml - space - wg_shift + space_slice))

    design_region[Npml + space:Nx - Npml - space, 
                    Npml + space:Ny - Npml - space] = 1

    # Const init
    rho = design_region * 0.5

    # Ranom init
    # np.random.seed(4)
    # rho = design_region * ( 0.5 + 0.001 * np.random.rand(Nx, Ny) )
    # np.random.seed(None)
    # rho = operator_blur(rho, radius=4, N=1)
    # rho = operator_proj(rho, beta=600, N=1)
    # rho = operator_blur(rho, radius=10, N=1)

    return rho, bg_rho, design_region, input_slice, output_slice1, output_slice2


def mask_combine_rho(rho, bg_rho, design_region):
        """Utility function for combining the design region rho and the background rho
        """
        return rho * design_region + bg_rho * (design_region==0).astype(float)

# Blurring filter that results in smooth features of the structure
# First we define a function to create the kernel
def create_blur_kernel(radius):
    """Helper function used below for creating the conv kernel"""
    rr, cc = disk((radius, radius), radius+1)
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=float)
    kernel[rr, cc] = 1
    return  kernel/kernel.sum()

# Define optimization objective
def mode_overlap(E1, E2):
    """Defines an overlap integral between the sim field and desired field
    """
    return npa.abs(npa.sum(npa.conj(E1)*E2))*1e6

#_, _, Ez1 = simulation1.solve(source1)
#_, _, Ez2 = simulation2.solve(source2)

#E01 = mode_overlap(Ez1, probe1)
#E02 = mode_overlap(Ez2, probe2)

def operator_proj(rho, eta=0.5, beta=100, N=1):
    """Density projection
    eta     : Center of the projection between 0 and 1
    beta    : Strength of the projection
    N       : Number of times to apply the projection
    """
    for i in range(N):
        rho =  npa.divide(npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta)),
                          npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))

    return rho


# Then we define the function to apply the operation
def operator_blur(rho, radius=2, N=1):
    """Blur operator implemented via two-dimensional convolution
    radius    : Radius of the circle for the conv kernel filter
    N         : Number of times to apply the filter

    Note that depending on the radius, the kernel is not always a
    perfect circle due to "pixelation" / stair casing
    """

    kernel = create_blur_kernel(radius)

    for i in range(N):
        # For whatever reason HIPS autograd doesn't support 'same' mode, so we need to manually crop the output
        rho = conv(rho, kernel, mode='full')[radius:-radius,radius:-radius]

    return rho

def objective(rho, bg_rho, design_region, radius, N_blur, beta, eta, N_proj,
              source1, source2, probe1, probe2, simulation1, simulation2, E01, E02):
        """Objective function called by optimizer

        1) Takes the density distribution as input
        2) Constructs epsr
        2) Runs the simulation
        3) Returns the overlap integral between the output wg field
        and the desired mode field"""

        # ensure that rho is the right shape
        epsr = epsr_parameterization(rho, bg_rho, design_region, 
                                     radius=radius, N_blur=N_blur, beta=beta, eta=eta, N_proj=N_proj)
        simulation1.eps_r = epsr
        simulation2.eps_r = epsr

        _, _, Ez1 = simulation1.solve(source1)
        _, _, Ez2 = simulation2.solve(source2)

        # Objective function used in the first set of experiments
        #return mode_overlap(Ez1, probe1) / E01 * mode_overlap(Ez2,probe2) / E02


        # Second objective function (incorporates penalty for cross combineation of probes and signals)
        obj = (mode_overlap(Ez1, probe1) / E01) * (mode_overlap(Ez2, probe2) / E02) - (mode_overlap(Ez1, probe2) / E01) - (mode_overlap(Ez2, probe1) / E02)
 
        return obj

def epsr_parameterization(rho, bg_rho, design_region, epsr_min=1.0, epsr_max=12.0,
                          radius=2, N_blur=1, beta=100, eta=0.5, N_proj=1):
        """Defines the parameterization steps for constructing rho"""
        # Combine rho and bg_rho; Note: this is so the subsequent blur sees the waveguides
        rho = mask_combine_rho(rho, bg_rho, design_region)

        rho = operator_blur(rho, radius=radius, N=N_blur)
        rho = operator_proj(rho, beta=beta, eta=eta, N=N_proj)

        # Final masking undoes the blurring of the waveguides
        rho = mask_combine_rho(rho, bg_rho, design_region)

        return epsr_min + (epsr_max-epsr_min) * rho

def sim(epsr, omega1, omega2, dl, Npml, source1, source2, slices=[]):
    """Solve a simulation with permittivity 'epsr'"""
    simulation1 = fdfd_ez(omega1, dl, epsr, [Npml, Npml])
    #_, _, Ez1 = simulation1.solve(source1)
    simulation2 = fdfd_ez(omega2, dl, epsr, [Npml, Npml])
    #_, _, Ez2 = simulation2.solve(source2)
    return simulation1, simulation2

def viz_sim(epsr, source1, source2, 
            omega1, dl, omega2, Npml,
            slices=[], directory=None, experiment_name="", 
            run=0, saverho=None):
    """Solve and visualize a simulation with permittivity 'epsr'"""

    simulation1 = fdfd_ez(omega1, dl, epsr, [Npml, Npml])
    _, _, Ez1 = simulation1.solve(source1)
    simulation2 = fdfd_ez(omega2, dl, epsr, [Npml, Npml])
    _, _, Ez2 = simulation2.solve(source2)

    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(9,3))
    ceviche.viz.abs(Ez1, outline=epsr, ax=ax[0], cbar=False)
    ceviche.viz.abs(Ez2, outline=epsr, ax=ax[1], cbar=False)
    ceviche.viz.abs(epsr, ax=ax[2], cmap='Greys')
    for sl in slices:
        ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'w-', alpha=0.5)
        ax[1].plot(sl.x*np.ones(len(sl.y)), sl.y, 'w-', alpha=0.5)

    ax[0].set_title('$\lambda_1$ = %.2f $\mu$m' % (299792458/(omega1/2/np.pi)/1e-6))
    ax[1].set_title('$\lambda_2$ = %.2f $\mu$m' % (299792458/(omega2/2/np.pi)/1e-6))

    if directory is not None:
        plt.savefig(directory + experiment_name + str(run) + '.png')
        if saverho is not None:
            np.save(directory + experiment_name + str(run) + '.npy', saverho)

    return (simulation1, simulation2, ax, fig)


# Apparently the below code can be used to make gifs, figure this out later.

# def callback_output_structure(iteration, of_list, rho):
#     """Callback function to output fields and the structures (for making sweet gifs)"""
#     rho = rho.reshape((Nx, Ny))
#     epsr = epsr_parameterization(rho, bg_rho, design_region, \
#                                 radius=blur_radius, N_blur=N_blur, beta=beta, eta=eta, N_proj=N_proj)
#     _, _, axs, fig = viz_sim(epsr, source1, source2, slices = [input_slice, output_slice1, output_slice2])
#     for ax in axs:
#         ax.set_xlabel('')
#         ax.set_ylabel('')
#         ax.set_yticks([])
#         ax.set_xticks([])

#     # note: make sure `workshop-invdesign/tmp` directory exists for this to work
#     plt.savefig('tmp/epsr_%03d.png' % iteration, dpi=70)
#     plt.close()


# def callback_beta_schedule(iteration, of_list, rho):
#     """Callback function for the optimizer to schedule changes to beta with the iteration number"""

#     # I am commiting a terrible sin by using globals here, but I am feeling lazy...
#     global beta

#     if iteration < 50:
#         beta = 10
#     elif 50 <= iteration & iteration < 75:
#         beta = 100
#     elif 75 <= iteration & iteration < 100:
#         beta = 200
#     else:
#         beta = 300

#     # Chain with the output structure callback
#     callback_output_structure(iteration, of_list, rho)