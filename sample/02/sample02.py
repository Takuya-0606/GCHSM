#!/usr/bin/env python
import numpy as np
from pyscf import gto, scf
from pyscf.solvent import hsm

# -------------------------
# Settings
# -------------------------
atom = '''
O   0.00000000000000   0.00000000000000   0.06865240486852
H   0.00000000000000   0.74760659813079  -0.54482630232708
H   0.00000000000000  -0.74760659813079  -0.54482630232708
'''
basis = 'sto-3g'
spin = 0
unit = 'angstrom'

eps = 80.1510
vdw_scale = 1.2
lebedev_order = 17
pcm_method = 'C-PCM'

# finite-difference step for Hessian
# HSM Hessian check is usually more stable with 0.005 or 0.01 Bohr
hstep = 0.005

# -------------------------
# Reference molecule
# -------------------------
mol0 = gto.M(
    atom=atom,
    basis=basis,
    spin=spin,
    unit=unit,
    verbose=4,
)

cavity_coords_ref = mol0.atom_coords(unit='B').copy()
symbols = [mol0.atom_symbol(i) for i in range(mol0.natm)]
coords0 = mol0.atom_coords(unit='B').copy()
natm = mol0.natm


def make_pcm(mol):
    pcm_obj = hsm.PCM(mol)
    pcm_obj.cavity_coords = cavity_coords_ref.copy()
    pcm_obj.method = pcm_method
    pcm_obj.eps = eps
    pcm_obj.vdw_scale = vdw_scale
    pcm_obj.lebedev_order = lebedev_order
    return pcm_obj


def run_grad(mol, verbose=0):
    mf = scf.RHF(mol).PCM(make_pcm(mol))
    mf.verbose = verbose
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    g = mf.nuc_grad_method().kernel()
    return g, mf


# -------------------------
# Analytical Hessian
# -------------------------
mf0 = scf.RHF(mol0).PCM(make_pcm(mol0))
mf0.conv_tol = 1e-12
mf0.conv_tol_grad = 1e-10
mf0.kernel()
if not mf0.converged:
    raise RuntimeError("Reference SCF did not converge")

h_anal = mf0.Hessian().kernel()   # shape = (natm,natm,3,3)

# -------------------------
# Numerical Hessian from central difference of gradients
# -------------------------
h_num = np.zeros((natm, natm, 3, 3))

for A in range(natm):
    for x in range(3):
        coords_p = coords0.copy()
        coords_m = coords0.copy()
        coords_p[A, x] += hstep
        coords_m[A, x] -= hstep

        mol_p = gto.M(
            atom=[(symbols[i], coords_p[i]) for i in range(natm)],
            basis=basis,
            spin=spin,
            unit='Bohr',
            verbose=0,
        )
        mol_m = gto.M(
            atom=[(symbols[i], coords_m[i]) for i in range(natm)],
            basis=basis,
            spin=spin,
            unit='Bohr',
            verbose=0,
        )

        g_p, _ = run_grad(mol_p, verbose=0)
        g_m, _ = run_grad(mol_m, verbose=0)

        # d g(B,y) / d R(A,x)
        h_num[A, :, x, :] = (g_p - g_m) / (2.0 * hstep)

# -------------------------
# Compare
# -------------------------
diff = h_anal - h_num

print("\n=== Hessian comparison ===")
print("Max abs diff :", np.max(np.abs(diff)))
print("RMS diff     :", np.sqrt(np.mean(diff**2)))
print("||H_anal||    :", np.linalg.norm(h_anal))
print("||H_num||     :", np.linalg.norm(h_num))

for A in range(natm):
    for B in range(natm):
        block_diff = diff[A, B]
        print(f"\nBlock ({A},{B}) max abs diff = {np.max(np.abs(block_diff)):.12e}")
        print(block_diff)
