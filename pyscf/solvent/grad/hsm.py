#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xiaojie Wu <wxj6000@gmail.com>
#

'''
Gradient of PCM family solvent models, copied from GPU4PySCF with modifications
'''

import numpy
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, df
from pyscf.solvent.pcm import PI, switch_h, PCM
from pyscf.grad import rhf as rhf_grad

def _sync_cavity_coords(pcmobj):
    if not hasattr(pcmobj, 'get_cavity_coords'):
        return
    cavity_coords = pcmobj.get_cavity_coords()
    if cavity_coords is None:
        return
    cavity_coords = numpy.asarray(cavity_coords, dtype=float)
    if cavity_coords.shape != (pcmobj.mol.natm, 3):
        raise ValueError('cavity_coords must have shape (natm, 3); '
                         f'got {cavity_coords.shape} for natm={pcmobj.mol.natm}')
    if pcmobj.surface is not None:
        pcmobj.surface['atom_coords'] = cavity_coords

def _ensure_q_sym(pcmobj, dm, q_sym):
    if q_sym is not None:
        return q_sym
    if not pcmobj._intermediates or 'q_sym' not in pcmobj._intermediates:
        pcmobj._get_vind(dm)
    try:
        return pcmobj._intermediates['q_sym']
    except KeyError as err:
        raise KeyError('q_sym is not available; ensure _get_vind has been called') from err

def grad_switch_h(x):
    ''' first derivative of h(x)'''
    dy = 30.0*x**2 - 60.0*x**3 + 30.0*x**4
    dy[x<0] = 0.0
    dy[x>1] = 0.0
    return dy

def get_dF_dA(surface):
    '''
    J. Chem. Phys. 133, 244111 (2010), Appendix C
    '''

    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    area        = surface['area']
    R_in_J      = surface['R_in_J']
    R_sw_J      = surface['R_sw_J']

    ngrids = grid_coords.shape[0]
    natom = atom_coords.shape[0]
    dF = numpy.zeros([ngrids, natom, 3])
    dA = numpy.zeros([ngrids, natom, 3])

    for ia in range(natom):
        p0,p1 = surface['gslice_by_atom'][ia]
        coords = grid_coords[p0:p1]
        ri_rJ = numpy.expand_dims(coords, axis=1) - atom_coords
        riJ = numpy.linalg.norm(ri_rJ, axis=-1)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        ri_rJ[:,ia,:] = 0.0
        ri_rJ[diJ < 1e-8] = 0.0

        fiJ = switch_h(diJ)
        dfiJ = grad_switch_h(diJ) / (fiJ * riJ * R_sw_J)
        dfiJ = numpy.expand_dims(dfiJ, axis=-1) * ri_rJ

        Fi = switch_fun[p0:p1]
        Ai = area[p0:p1]

        # grids response
        Fi = numpy.expand_dims(Fi, axis=-1)
        Ai = numpy.expand_dims(Ai, axis=-1)
        dFi_grid = numpy.sum(dfiJ, axis=1)

        dF[p0:p1,ia,:] += Fi * dFi_grid
        dA[p0:p1,ia,:] += Ai * dFi_grid

        # atom response
        Fi = numpy.expand_dims(Fi, axis=-2)
        Ai = numpy.expand_dims(Ai, axis=-2)
        dF[p0:p1,:,:] -= Fi * dfiJ
        dA[p0:p1,:,:] -= Ai * dfiJ

    return dF, dA

def get_dD_dS(surface, dF, with_S=True, with_D=False):
    '''
    derivative of D and S w.r.t grids, partial_i D_ij = -partial_j D_ij
    S is symmetric, D is not
    '''
    grid_coords = surface['grid_coords']
    exponents   = surface['charge_exp']
    norm_vec    = surface['norm_vec']
    switch_fun  = surface['switch_fun']

    xi_i, xi_j = numpy.meshgrid(exponents, exponents, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    ri_rj = numpy.expand_dims(grid_coords, axis=1) - grid_coords
    rij = numpy.linalg.norm(ri_rj, axis=-1)
    xi_r_ij = xi_ij * rij
    numpy.fill_diagonal(rij, 1)

    dS_dr = -(scipy.special.erf(xi_r_ij) - 2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2))/rij**2
    numpy.fill_diagonal(dS_dr, 0)

    dS_dr= numpy.expand_dims(dS_dr, axis=-1)
    drij = ri_rj/numpy.expand_dims(rij, axis=-1)
    dS = dS_dr * drij

    dD = None
    if with_D:
        nj_rij = numpy.sum(ri_rj * norm_vec, axis=-1)
        dD_dri = 4.0*xi_r_ij**2 * xi_ij / PI**0.5 * numpy.exp(-xi_r_ij**2) * nj_rij / rij**3
        numpy.fill_diagonal(dD_dri, 0.0)

        rij = numpy.expand_dims(rij, axis=-1)
        nj_rij = numpy.expand_dims(nj_rij, axis=-1)
        nj = numpy.expand_dims(norm_vec, axis=0)
        dD_dri = numpy.expand_dims(dD_dri, axis=-1)

        dD = dD_dri * drij + dS_dr * (-nj/rij + 3.0*nj_rij/rij**2 * drij)

    dSii_dF = -exponents * (2.0/PI)**0.5 / switch_fun**2
    dSii = numpy.expand_dims(dSii_dF, axis=(1,2)) * dF

    return dD, dS, dSii

def grad_nuc(pcmobj, dm, q_sym = None):
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and numpy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)

    mol = pcmobj.mol
    q_sym = _ensure_q_sym(pcmobj, dm, q_sym)
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords']
    exponents    = pcmobj.surface['charge_exp']

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=exponents**2)

    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')

    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)

    dv_g = numpy.einsum('g,xng->nx', q_sym, v_ng_ip1)
    de = -numpy.einsum('nx,n->nx', dv_g, atom_charges)

    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)

    dv_g = numpy.einsum('n,xgn->gx', atom_charges, v_ng_ip1)
    dv_g = numpy.einsum('gx,g->gx', dv_g, q_sym)

    de -= numpy.asarray([numpy.sum(dv_g[p0:p1], axis=0) for p0,p1 in gridslice])
    t1 = log.timer_debug1('grad nuc', *t1)
    return de

def grad_qv(pcmobj, dm, q_sym = None):
    '''
    contributions due to integrals
    '''
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and numpy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    nao = mol.nao
    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    gridslice    = pcmobj.surface['gslice_by_atom']
    q_sym = _ensure_q_sym(pcmobj, dm, q_sym)
    grid_coords  = pcmobj.surface['grid_coords']
    exponents    = pcmobj.surface['charge_exp']

    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2/3, 400))
    ngrids = q_sym.shape[0]
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
    dvj = numpy.zeros([3,nao])
    for p0, p1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)
        dvj += numpy.einsum('xijk,ij,k->xi', v_nj, dm, q_sym[p0:p1])

    int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
    dq = numpy.empty([3,ngrids])
    for p0, p1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
        q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)
        dq[:,p0:p1] = numpy.einsum('xijk,ij,k->xk', q_nj, dm, q_sym[p0:p1])

    aoslice = mol.aoslice_by_atom()
    dq = numpy.asarray([numpy.sum(dq[:,p0:p1], axis=1) for p0,p1 in gridslice])
    dvj= 2.0 * numpy.asarray([numpy.sum(dvj[:,p0:p1], axis=1) for p0,p1 in aoslice[:,2:]])
    de = dq + dvj
    t1 = log.timer_debug1('grad qv', *t1)
    return de


def grad_solver(pcmobj, dm):
    r'''Fixed-cavity solver contribution for GC-HSM.

    In the harmonic solvation model (HSM), the cavity is frozen during
    differentiation. Consequently, the PCM matrices K and R do not depend on the
    nuclear coordinates, so

        d/dR [K^{-1} R] = 0.

    The solvent gradient therefore reduces to q^T (dv/dR), which is already
    covered by :func:`grad_qv` and :func:`grad_nuc`. The solver term must vanish
    identically; keeping the generic PCM dK/dR and dR/dR contributions here is
    inconsistent with the fixed-cavity HSM formalism and reintroduces the
    spurious cavity-response terms that Eq. (30) removes.
    '''
    return numpy.zeros((pcmobj.mol.natm, 3))

def make_grad_object(base_method):
    '''Create nuclear gradients object with solvent contributions for the given
    solvent-attached method based on its gradients method in vaccum
    '''
    from pyscf.solvent._attach_solvent import _Solvation
    if isinstance(base_method, rhf_grad.GradientsBase):
        # For backward compatibility. The input argument is a gradient object in
        # previous implementations.
        base_method = base_method.base

    # Must be a solvent-attached method
    assert isinstance(base_method, _Solvation)
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    # create the Gradients in vacuum. Cannot call super().Gradients() here
    # because other dynamic corrections might be applied to the base_method.
    # Calling super().Gradients might discard these corrections.
    vac_grad = base_method.undo_solvent().Gradients()
    # The base method for vac_grad discards the with_solvent. Change its base to
    # the solvent-attached base method
    vac_grad.base = base_method
    name = with_solvent.__class__.__name__ + vac_grad.__class__.__name__
    return lib.set_class(WithSolventGrad(vac_grad),
                         (WithSolventGrad, vac_grad.__class__), name)

class WithSolventGrad:
    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventGrad, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_gpu(self):
        from pyscf.lib.misc import to_gpu
        from pyscf.tdscf.rhf import TDBase
        # Only PCM and SMD are available on GPU.
        # FIXME: The SMD class is a child class of PCM now. Additional check for
        # SMD should be made if SMD is refactored as an independent class
        assert isinstance(self.base.with_solvent, PCM)
        if isinstance(self, TDBase):
            raise NotImplementedError('.to_gpu() for PCM-TDDFT')
        return to_gpu(self, self.base.to_gpu().Gradients())

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        logger.debug(self, 'Compute gradients from solvents')
        self.de_solvent = self.base.with_solvent.grad(dm)
        logger.debug(self, 'Compute gradients from solutes')
        self.de_solute = super().kernel(*args, **kwargs)
        self.de = self.de_solute + self.de_solvent

        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
