#!/usr/bin/env python
import numpy as np
from pyscf import gto, scf
from pyscf.solvent import hsm
from pyscf.solvent.hessian import hsm as hsm_hess


# ============================================================
# User settings
# ============================================================
ATOM = '''
O   0.00000000000000   0.00000000000000   0.06865240486852
H   0.00000000000000   0.74760659813079  -0.54482630232708
H   0.00000000000000  -0.74760659813079  -0.54482630232708
'''
BASIS = 'sto-3g'
SPIN = 0
UNIT = 'angstrom'

PCM_METHOD = 'C-PCM'
EPS = 80.1510
VDW_SCALE = 1.2
LEBEDEV_ORDER = 17

SCF_CONV_TOL = 1e-12
SCF_CONV_TOL_GRAD = 1e-10

HSTEP = 0.005   # Bohr
VERBOSE = 4


# ============================================================
# Build reference molecule
# ============================================================
mol0 = gto.M(
    atom=ATOM,
    basis=BASIS,
    spin=SPIN,
    unit=UNIT,
    verbose=VERBOSE,
)

coords0 = mol0.atom_coords(unit='B').copy()
symbols = [mol0.atom_symbol(i) for i in range(mol0.natm)]
natm = mol0.natm

# fixed cavity coordinates for GC-HSM
cavity_coords_ref = coords0.copy()


# ============================================================
# Helpers
# ============================================================
def make_hsm_pcm(mol):
    pcm_obj = hsm.PCM(mol)
    pcm_obj.cavity_coords = cavity_coords_ref.copy()   # fixed cavity
    pcm_obj.method = PCM_METHOD
    pcm_obj.eps = EPS
    pcm_obj.vdw_scale = VDW_SCALE
    pcm_obj.lebedev_order = LEBEDEV_ORDER
    return pcm_obj


def run_hsm_all(mol, verbose=0):
    """Run RHF+GC-HSM and return mf, gradient, Hessian."""
    mf = scf.RHF(mol).PCM(make_hsm_pcm(mol))
    mf.verbose = verbose
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("HSM SCF did not converge")

    grad = mf.nuc_grad_method().kernel()

    # IMPORTANT:
    # use HSM-specific Hessian object directly
    hobj = hsm_hess.make_hess_object(mf)
    hess = hobj.kernel()

    return mf, grad, hess, hobj


def run_vac_all(mol, verbose=0):
    """Run vacuum RHF and return mf, gradient, Hessian."""
    mf = scf.RHF(mol)
    mf.verbose = verbose
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Vacuum SCF did not converge")

    grad = mf.nuc_grad_method().kernel()
    hess = mf.Hessian().kernel()

    return mf, grad, hess


def numerical_hessian_from_grad(run_grad_func, coords_ref, hstep):
    """Build numerical Hessian by central difference of analytic gradients."""
    h_num = np.zeros((natm, natm, 3, 3))

    for A in range(natm):
        for x in range(3):
            coords_p = coords_ref.copy()
            coords_m = coords_ref.copy()
            coords_p[A, x] += hstep
            coords_m[A, x] -= hstep

            mol_p = gto.M(
                atom=[(symbols[i], coords_p[i]) for i in range(natm)],
                basis=BASIS,
                spin=SPIN,
                unit='Bohr',
                verbose=0,
            )
            mol_m = gto.M(
                atom=[(symbols[i], coords_m[i]) for i in range(natm)],
                basis=BASIS,
                spin=SPIN,
                unit='Bohr',
                verbose=0,
            )

            g_p = run_grad_func(mol_p)
            g_m = run_grad_func(mol_m)

            h_num[A, :, x, :] = (g_p - g_m) / (2.0 * hstep)

    return h_num


def run_hsm_grad_only(mol):
    mf = scf.RHF(mol).PCM(make_hsm_pcm(mol))
    mf.verbose = 0
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("HSM SCF did not converge in gradient-only run")
    return mf.nuc_grad_method().kernel()


def run_vac_grad_only(mol):
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Vacuum SCF did not converge in gradient-only run")
    return mf.nuc_grad_method().kernel()


def print_hessian_comparison(title, h_ana, h_num):
    diff = h_ana - h_num
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print("||H_ana||      =", np.linalg.norm(h_ana))
    print("||H_num||      =", np.linalg.norm(h_num))
    print("Max abs diff   =", np.max(np.abs(diff)))
    print("RMS diff       =", np.sqrt(np.mean(diff**2)))

    for A in range(natm):
        for B in range(natm):
            block = diff[A, B]
            print(f"\nBlock ({A},{B}) max abs diff = {np.max(np.abs(block)):.12e}")
            print(block)


# ============================================================
# 1. Analytical Hessians
# ============================================================
print("\nRunning reference analytical calculations...")
mf_hsm0, g_hsm_ana, h_hsm_ana, hobj_hsm0 = run_hsm_all(mol0, verbose=VERBOSE)
mf_vac0, g_vac_ana, h_vac_ana = run_vac_all(mol0, verbose=0)

print("\nHSM Hessian object class:", type(hobj_hsm0))
print("||de_solvent|| =", np.linalg.norm(hobj_hsm0.de_solvent))

# solvent-only analytical Hessian
h_solv_ana = h_hsm_ana - h_vac_ana


# ============================================================
# 2. Numerical Hessians from central difference of gradients
# ============================================================
print("\nBuilding numerical HSM Hessian...")
h_hsm_num = numerical_hessian_from_grad(run_hsm_grad_only, coords0, HSTEP)

print("\nBuilding numerical vacuum Hessian...")
h_vac_num = numerical_hessian_from_grad(run_vac_grad_only, coords0, HSTEP)

# solvent-only numerical Hessian
h_solv_num = h_hsm_num - h_vac_num


# ============================================================
# 3. Comparisons
# ============================================================
print_hessian_comparison("Total HSM Hessian: analytical vs numerical", h_hsm_ana, h_hsm_num)
print_hessian_comparison("Vacuum Hessian: analytical vs numerical", h_vac_ana, h_vac_num)
print_hessian_comparison("Solvent-only Hessian: analytical vs numerical", h_solv_ana, h_solv_num)
