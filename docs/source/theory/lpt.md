# LPT initialization

Lagrangian perturbation theory initializes particle displacements and velocities from linear modes. PM++ generates white noise, maps it through linear transfer information, and passes the resulting modes to `pmpp.lpt.lpt` to build the starting particle state.

LPT is used before the N-body integrator so the later leapfrog steps start from physically meaningful initial displacements rather than from an unperturbed grid.
