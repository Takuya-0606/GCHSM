#!/usr/bin/env python
import numpy as np
from pyscf import gto, scf
from pyscf.solvent import hsm
from pyscf.solvent.hessian import hsm as hsm_hess

ATOM = '''
O   0.00000000000000   0.00000000000000   0.06865240486852
H   0.00000000000000   0.74760659813079  -0.54482630232708
H   0.00000000000000  -0.74760659813079  -0.54482630232708
'''
BASIS = 'sto-3g'
SPIN = 0
HSTEP = 0.005

PCM_METHOD = 'C-PCM'
EPS = 80.1510
VDW_SCALE = 1.2
LEBEDEV_ORDER = 17

SCF_CONV_TOL = 1e-12
SCF_CONV_TOL_GRAD = 1e-10

mol0 = gto.M(atom=ATOM, basis=BASIS, spin=SPIN, unit='angstrom', verbose=4)
coords0 = mol0.atom_coords(unit='B').copy()
symbols = [mol0.atom_symbol(i) for i in range(mol0.natm)]
natm = mol0.natm
cavity_coords_ref = coords0.copy()

def make_pcm(mol):
    pcm_obj = hsm.PCM(mol)
    pcm_obj.cavity_coords = cavity_coords_ref.copy()
    pcm_obj.method = PCM_METHOD
    pcm_obj.eps = EPS
    pcm_obj.vdw_scale = VDW_SCALE
    pcm_obj.lebedev_order = LEBEDEV_ORDER
    return pcm_obj

def run_hsm_all(mol, verbose=0):
    mf = scf.RHF(mol).PCM(make_pcm(mol))
    mf.verbose = verbose
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("HSM SCF did not converge")

    grad = mf.nuc_grad_method().kernel()

    # ここが重要
    hobj = hsm_hess.make_hess_object(mf)
    hess = hobj.kernel()
    return mf, grad, hess, hobj

def run_vac_all(mol, verbose=0):
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

def run_hsm_grad_only(mol):
    mf = scf.RHF(mol).PCM(make_pcm(mol))
    mf.verbose = 0
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("HSM SCF did not converge")
    return mf.nuc_grad_method().kernel()

def run_vac_grad_only(mol):
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Vacuum SCF did not converge")
    return mf.nuc_grad_method().kernel()

def numerical_hessian_from_grad(run_grad_func):
    h_num = np.zeros((natm, natm, 3, 3))
    for A in range(natm):
        for x in range(3):
            coords_p = coords0.copy()
            coords_m = coords0.copy()
            coords_p[A, x] += HSTEP
            coords_m[A, x] -= HSTEP

            mol_p = gto.M(
                atom=[(symbols[i], coords_p[i]) for i in range(natm)],
                basis=BASIS, spin=SPIN, unit='Bohr', verbose=0
            )
            mol_m = gto.M(
                atom=[(symbols[i], coords_m[i]) for i in range(natm)],
                basis=BASIS, spin=SPIN, unit='Bohr', verbose=0
            )

            g_p = run_grad_func(mol_p)
            g_m = run_grad_func(mol_m)
            h_num[A, :, x, :] = (g_p - g_m) / (2.0 * HSTEP)
    return h_num

def compare(title, h_ana, h_num):
    diff = h_ana - h_num
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print("||H_ana||      =", np.linalg.norm(h_ana))
    print("||H_num||      =", np.linalg.norm(h_num))
    print("Max abs diff   =", np.max(np.abs(diff)))
    print("RMS diff       =", np.sqrt(np.mean(diff**2)))

mf_hsm0, g_hsm_ana, h_hsm_ana, hobj_hsm0 = run_hsm_all(mol0, verbose=4)
mf_vac0, g_vac_ana, h_vac_ana = run_vac_all(mol0, verbose=0)

print("\nHSM Hessian object class:", type(hobj_hsm0))
print("||de_solvent|| =", np.linalg.norm(hobj_hsm0.de_solvent))

h_hsm_num = numerical_hessian_from_grad(run_hsm_grad_only)
h_vac_num = numerical_hessian_from_grad(run_vac_grad_only)

h_solv_ana = h_hsm_ana - h_vac_ana
h_solv_num = h_hsm_num - h_vac_num

compare("Total HSM Hessian: analytical vs numerical", h_hsm_ana, h_hsm_num)
compare("Vacuum Hessian: analytical vs numerical", h_vac_ana, h_vac_num)
compare("Solvent-only Hessian: analytical vs numerical", h_solv_ana, h_solv_num)
