# Linear_Programming_Algorithms
LP solvers (simplex, ellipsoid, and interior point methods)

## Simplex:
The simplex algorithm, developed by George Dantzig in 1947, solves LP problems by constructing a feasible solution at a vertex of the polytope and then walking along a path on the edges of the polytope to vertices with non-decreasing values of the objective function until an optimum is reached for sure. In many practical problems, "stalling" occurs: many pivots are made with no increase in the objective function In rare practical problems, the usual versions of the simplex algorithm may actually "cycle". To avoid cycles, researchers developed new pivoting rules.


## Interior Point Methods:
Interior-point methods move through the interior of the feasible region, unlike the simplex method which finds an optimal solution by traversing the edges between vertices on a polyhedral set.
