# Thermodynamic scope

Supported in 0.1: non-electrolyte molecular mixtures; Ideal/Raoult bubble point, dew point,
binary isobaric T-x-y, isothermal P-x-y, TP flash, numerical azeotrope candidates, basic stability
classification, and a typed LLE request/response boundary.

Model cards are available for Ideal, Wilson, NRTL, UNIQUAC, and Peng-Robinson. A card means the
router understands applicability; it does not claim that every model has a production solver.

Unsupported: electrolytes, reaction equilibrium, SLE, polymers, hydrates, petroleum pseudo-
components, polymorphs, VLLE, and automatic full-column design.
