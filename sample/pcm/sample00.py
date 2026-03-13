#!/usr/bin/env python
from pyscf import gto, scf, dft, cc, solvent, mp
from pyscf.hessian import thermo
from pyscf.solvent import pcm
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

pcm_obj = pcm.PCM(mol)
#pcm_obj.cavity_coords = mol.atom_coords(unit='B')
pcm_obj.method        = 'C-PCM'
pcm_obj.eps           = 80.1510
pcm_obj.vdw_scale     = 1.2
pcm_obj.lebedev_order = 17

# Calculation level
# Hartree-Fock
mymp = scf.RHF(mol).PCM(pcm_obj)
mymp.kernel()

# Analytical hessian
g = mymp.nuc_grad_method()
g.kernel()
h = mymp.Hessian()
hessian = h.kernel()

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
