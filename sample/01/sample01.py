#!/usr/bin/env python
import numpy as np
from pyscf import gto, scf
from pyscf.solvent import hsm

# =========================
# User settings
# =========================
atom = '''
Na   0.00000000000000   0.00000000000000   0.00000000000000
Cl   2.50000000000000   0.00000000000000   0.00000000000000
'''
basis = 'sto-3g'
charge = 0
spin = 0
unit = 'angstrom'

pcm_method = 'C-PCM'
eps = 80.1510
vdw_scale = 1.2
lebedev_order = 17

# finite-difference step in Bohr
h = 5.0e-3

# =========================
# Build reference molecule
# =========================
mol0 = gto.M(
    atom=atom,
    basis=basis,
    charge=charge,
    spin=spin,
    unit=unit,
    verbose=4,
)

# Reference cavity coordinates for HSM (fixed cavity)
cavity_coords_ref = mol0.atom_coords(unit='B').copy()


def make_pcm(mol, cavity_coords_fixed):
    """Create HSM PCM object with fixed cavity coordinates."""
    pcm_obj = hsm.PCM(mol)
    pcm_obj.cavity_coords = cavity_coords_fixed.copy()
    pcm_obj.method = pcm_method
    pcm_obj.eps = eps
    pcm_obj.vdw_scale = vdw_scale
    pcm_obj.lebedev_order = lebedev_order
    return pcm_obj


def run_scf_energy(mol, cavity_coords_fixed, dm0=None, verbose=0):
    """Run RHF+HSM single-point energy."""
    pcm_obj = make_pcm(mol, cavity_coords_fixed)
    mf = scf.RHF(mol).PCM(pcm_obj)
    mf.verbose = verbose
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10

    e = mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("SCF did not converge.")
    return e, mf


# =========================
# Analytical gradient
# =========================
pcm0 = make_pcm(mol0, cavity_coords_ref)
mf0 = scf.RHF(mol0).PCM(pcm0)
mf0.conv_tol = 1e-12
mf0.conv_tol_grad = 1e-10
e0 = mf0.kernel()

if not mf0.converged:
    raise RuntimeError("Reference SCF did not converge.")

g_anal = mf0.nuc_grad_method().kernel()   # shape = (natm, 3), Hartree/Bohr

print("\n=== Analytical gradient (Hartree/Bohr) ===")
print(g_anal)

# =========================
# Numerical gradient by central difference
# =========================
coords0 = mol0.atom_coords(unit='B').copy()
symbols = [mol0.atom_symbol(i) for i in range(mol0.natm)]
g_num = np.zeros_like(coords0)

# use reference density as initial guess for nearby points
dm_ref = mf0.make_rdm1()

for ia in range(mol0.natm):
    for xyz in range(3):
        coords_p = coords0.copy()
        coords_m = coords0.copy()
        coords_p[ia, xyz] += h
        coords_m[ia, xyz] -= h

        mol_p = gto.M(
            atom=[(symbols[i], coords_p[i]) for i in range(mol0.natm)],
            basis=basis,
            charge=charge,
            spin=spin,
            unit='Bohr',
            verbose=0,
        )
        mol_m = gto.M(
            atom=[(symbols[i], coords_m[i]) for i in range(mol0.natm)],
            basis=basis,
            charge=charge,
            spin=spin,
            unit='Bohr',
            verbose=0,
        )

        ep, _ = run_scf_energy(mol_p, cavity_coords_ref, dm0=dm_ref, verbose=0)
        em, _ = run_scf_energy(mol_m, cavity_coords_ref, dm0=dm_ref, verbose=0)

        g_num[ia, xyz] = (ep - em) / (2.0 * h)

print("\n=== Numerical gradient by central difference (Hartree/Bohr) ===")
print(g_num)

# =========================
# Comparison
# =========================
diff = g_anal - g_num

print("\n=== Difference: analytical - numerical (Hartree/Bohr) ===")
print(diff)

print("\n=== Summary ===")
print(f"Max abs diff   : {np.max(np.abs(diff)):.12e}")
print(f"RMS diff       : {np.sqrt(np.mean(diff**2)):.12e}")
print(f"Analytical norm: {np.linalg.norm(g_anal):.12e}")
print(f"Numerical norm : {np.linalg.norm(g_num):.12e}")

print("\n=== Per-component comparison ===")
for ia in range(mol0.natm):
    for xyz, lab in enumerate("xyz"):
        print(
            f"Atom {ia:2d} ({symbols[ia]:>2s}) {lab}: "
            f"anal = {g_anal[ia, xyz]: .12e}   "
            f"num = {g_num[ia, xyz]: .12e}   "
            f"diff = {diff[ia, xyz]: .12e}"
        )

from pyscf.solvent.grad.hsm import grad_qv, grad_nuc, grad_solver

dm = mf0.make_rdm1()
g_qv = grad_qv(mf0.with_solvent, dm)
g_nuc = grad_nuc(mf0.with_solvent, dm)
g_sol = grad_solver(mf0.with_solvent, dm)

print("grad_qv =\n", g_qv)
print("grad_nuc =\n", g_nuc)
print("grad_solver =\n", g_sol)
print("grad_solvent_total =\n", g_qv + g_nuc + g_sol)
