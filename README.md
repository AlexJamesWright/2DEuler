# 2DEuler
------------

Evolves the two dimensional Euler equations for various models. Namely Special
Relativistic Hydrodynamics (SRHD), Special Relativistic Magnetohydrodynamics
(SRMHD) and the Two Fluid Electromagnetohydrodynamics model of Amano (TFEMHD).

Care may need to be taken with restecp to the type of the time integrator, e.g. 
the two fluid model requires implicitly solving for the source contribution so 
as to keep convergence to a solution for conventional timestep sizes when 
evolving stiff systems. For ideal fluids explicit integrators will suffice.