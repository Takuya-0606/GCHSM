import inspect
import pyscf
import pyscf.solvent.hsm as hsm_mod
import pyscf.solvent.hessian.hsm as hsm_hess_mod

print("pyscf.__file__ =", pyscf.__file__)
print("hsm_mod.__file__ =", hsm_mod.__file__)
print("hsm_hess_mod.__file__ =", hsm_hess_mod.__file__)

from pyscf.solvent.hessian.hsm import get_dqsym_dx, analytical_hess_solver, analytical_grad_vmat

print("\n[get_dqsym_dx]")
print(inspect.getsource(get_dqsym_dx))

print("\n[analytical_hess_solver]")
print(inspect.getsource(analytical_hess_solver))

src = inspect.getsource(analytical_grad_vmat)
print("\n[int3c2e_ip2 in analytical_grad_vmat?] ", "int3c2e_ip2" in src)
print(src)
