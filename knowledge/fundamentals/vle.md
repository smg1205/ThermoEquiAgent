# Vapor-liquid equilibrium

VLE requires equal component fugacity between coexisting phases. At low pressure with an ideal gas,
modified Raoult's law writes `y_i P = x_i gamma_i P_i^sat`. Ideal/Raoult sets `gamma_i = 1`.
Bubble calculations specify liquid composition; dew calculations specify vapor composition. A TP
flash specifies feed, temperature, and pressure and solves phase fraction plus phase compositions.
