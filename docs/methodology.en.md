# Methodology

The first backend uses modified Raoult's law with ideal activity coefficients and Antoine vapor
pressures. Bubble point solves `sum(x_i P_i^sat(T)) = P`; dew point solves
`sum(y_i P / P_i^sat(T)) = 1`. Isothermal curves evaluate bubble pressure and vapor composition.
TP flash solves Rachford-Rice using `K_i = P_i^sat/P` and explicitly handles single-phase limits.

Wilson, NRTL, UNIQUAC, and Peng-Robinson are represented by model cards and routing rules. They are
not silently approximated by the ideal adapter. Parameterized calls must resolve evidence-bearing
parameters before an implementation may execute.
