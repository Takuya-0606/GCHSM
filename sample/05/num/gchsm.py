#!/usr/bin/env python
from pyscf import gto, scf, dft, cc, solvent, mp
from pyscf.hessian import thermo
from pyscf.solvent import pcm, hsm
import numpy
import numpy as np
mol = gto.M(
    atom = '''
  O           0.00000000000000      0.00000000000000      0.06865240486852
  H           0.00000000000000      0.74760659813079     -0.54482630232708
  H           0.00000000000000     -0.74760659813079     -0.54482630232708
''',
    basis   = '6-31g',
    spin = 0,
    symmetry=True,
    unit    = 'angstrom',
    verbose = 4,
)

pcm_obj = hsm.PCM(mol)
pcm_obj.cavity_coords = mol.atom_coords(unit='B')
pcm_obj.method        = 'C-PCM'
pcm_obj.eps           = 80.1510
pcm_obj.vdw_scale     = 1.2
pcm_obj.lebedev_order = 17

# Calculation level
# Hartree-Fock
mymp = scf.RHF(mol).PCM(pcm_obj)
mymp.kernel()
# DFT
#mymp = dft.RKS(mol,xc='b3lypg').PCM(pcm_obj)
#mymp = dft.RKS(mol,xc='b3lypg')
#mymp.kernel()
# MP2
#mf = scf.RHF(mol)
#mf.kernel()
#mymp = mp.MP2(mf).PCM(pcm_obj)
#mymp.kernel()

def fd_stencil_centered(fd_order: int):
    stencils = {
        2: ([-1, 1],
            [-1/2, 1/2]),

        4: ([-2, -1,  1,  2],
            [ 1/12, -2/3, 2/3, -1/12]),

        6: ([-3, -2,  -1,  1,   2,   3],
            [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]),

        8: ([-4,  -3,   -2,   -1,   1,    2,    3,    4],
            [ 1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280]),

        10: ([-5,  -4,   -3,   -2,   -1,   1,    2,    3,    4,    5],
             [-1/1260, 5/504, -5/84, 5/21, -5/6, 5/6, -5/21, 5/84, -5/504, 1/1260]),

        12: ([-6,  -5,   -4,   -3,   -2,   -1,   1,    2,    3,    4,    5,    6],
             [ 1/5544, -1/385, 1/56, -5/63, 15/56, -6/7, 6/7, -15/56, 5/63, -1/56, 1/385, -1/5544])
    }
    if fd_order not in stencils:
        raise ValueError(f"fd_order must be one of {sorted(stencils.keys())}. Got {fd_order}")
    return stencils[fd_order]

def fd_hessian(mymp, step=5e-3, fd_order=2):

    mol = mymp.mol
    g = mymp.nuc_grad_method()
    g_scan = g.as_scanner()

    pmol = mol.copy()
    pmol.build()

    coords0 = pmol.atom_coords(unit='B')
    natm = pmol.natm
    hessian = np.zeros((natm, natm, 3, 3), dtype=float)

    offsets, coeffs = fd_stencil_centered(fd_order)

    for i in range(natm):
        for j in range(3):
            disp_unit = np.zeros_like(coords0)
            disp_unit[i, j] = 1.0

            acc = np.zeros((natm, 3), dtype=float)
            for k, ck in zip(offsets, coeffs):
                pmol.set_geom_(coords0 + (k * step) * disp_unit, unit='B')
                pmol.build()
                _, gk = g_scan(pmol)
                acc += ck * np.asarray(gk, dtype=float)

            hessian[i, :, j, :] = acc / step

    for a in range(natm):
        for b in range(a):
            hessian[b, a, :, :] = hessian[a, b, :, :].T

    pmol.set_geom_(coords0, unit='B')
    pmol.build()

    return hessian

# Analytical hessian

#g = mymp.nuc_grad_method()
#g.kernel()
#h = mymp.Hessian()
#hessian = h.kernel()

hessian = fd_hessian(mymp, step=5e-3, fd_order=2)
    # thermodynamics
results = thermo.harmonic_analysis(
        mol, hessian,
        exclude_trans=False, 
        exclude_rot=False,  
        imaginary_freq=False,
    )

freqs = results['freq_wavenumber'].real
norm_mode = results['norm_mode']

natm = mol.natm
print("Frequencies [cm^-1]:")
for i, w in enumerate(freqs):
    print(f"{i}: {w:.4f}")

thermo.dump_normal_mode(mol, results)
