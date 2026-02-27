Research Question

**Can morphology-aware local rewiring produce different synchronization outcomes compared to a global random rewiring rule?**

Model Description

•Each cell is represented as a phase oscillator.

•Oscillators interact only with their neighbors.

•The neighbor network can change over time.

**Three conditions are compared:**

•null_fixed – static neighbor network

•null_global – random global rewiring

•alt_local – local, geometry-aware rewiring

**Key Parameters**

•ρ (rewiring rate)

•ε (coupling strength)

•τ (interaction delay)

**Measurement**

Synchronization quality is evaluated using a defect rate,
which counts large phase differences between neighboring oscillators.

To compare models, I compute:

•Δdefect = defect(alt_local) − defect(null_global)

**Main Observation**

•The difference between local and global rewiring depends on the parameter region.

•For certain values of ρ and τ, the morphology-aware rule produces different synchronization behavior compared to the global null.

•At high rewiring rates, the differences tend to disappear.

**Technical Details**

•Implemented in Python

•Numerical phase updates

•Parameter sweeps over ρ and τ

•Multiple simulation seeds

**Limitations**

•Abstract phase oscillator model

•No molecular Notch-Delta dynamics

•Parameters are scanned, not emergent

•No direct experimental validation
