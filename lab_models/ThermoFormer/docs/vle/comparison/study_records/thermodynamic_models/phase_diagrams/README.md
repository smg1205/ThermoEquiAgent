# Phase-diagram case study

This exploratory analysis contains separate binary and ternary case studies.
Both select fixed-temperature and fixed-pressure test curves with at least six
composition points.

The binary study requires complete prediction coverage from ThermoFormer,
NRTL, Wilson, and UNIQUAC. The ternary Gibbs-triangle study covers the
available low- and high-temperature P-x-y extrapolation tests and requires
ThermoFormer, NRTL, and Wilson coverage; UNIQUAC is displayed only when its
parameter and solver coverage is available. The registered pressure protocols
contain no ternary T-x-y test samples.

The lowest- and highest-error ThermoFormer curve in each task is visualized.
Because test labels determine the cases, the figure is diagnostic rather than
confirmatory evidence. Complete candidate rankings are retained alongside the
selected plots.
