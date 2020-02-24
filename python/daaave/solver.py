'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import gurobipy
from scipy import sparse
from daaave.model_utils import convert_sbml_to_cobra
import numpy as np


LP_TOL = 1e-6


def optimize_cobra_model(sbml):
    '''Replicate Cobra command optimizeCbModel(model,[],'one').'''
    cobra = convert_sbml_to_cobra(sbml)

    return easy_lp(cobra['c'], cobra['S'], cobra['b'],
                   cobra['lb'], cobra['ub'], one=True)


def easy_lp(f, a, b, vlb, vub, one=False):
    '''Optimize lp using friends of Gurobi.'''

    # catch np arrays
    f, b, vlb, vub = list(f), list(b), list(vlb), list(vub)

    # create gurobi model
    lp = gurobipy.Model()
    lp.Params.OutputFlag = 0
    lp.Params.FeasibilityTol = 1e-9  # as per Cobra
    lp.Params.OptimalityTol = 1e-9  # as per Cobra
    rows, cols = a.shape
    # add variables to model
    for j in range(cols):
        LB = vlb[j]
        if LB == -np.inf:
            LB = -gurobipy.GRB.INFINITY
        UB = vub[j]
        if UB == np.inf:
            UB = gurobipy.GRB.INFINITY
        lp.addVar(lb=LB, ub=UB, obj=f[j])
    lp.update()
    lpvars = lp.getVars()
    # iterate over the rows of S adding each row into the model
    S = a.tocsr()
    for i in range(rows):
        start = S.indptr[i]
        end = S.indptr[i + 1]
        variables = [lpvars[j] for j in S.indices[start:end]]
        coeff = S.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        lp.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=b[i])
    lp.ModelSense = -1
    lp.update()
    lp.optimize()

    v = np.empty(len(f))
    v[:] = np.nan
    f_opt = np.nan
    conv = False
    if lp.Status == gurobipy.GRB.OPTIMAL:
        f_opt = lp.ObjVal
        conv = True
        v = [var.x for var in lp.getVars()]

    # remove model: better memory management?
    del lp

    if conv and one:
        # minimise one norm
        col = sparse.lil_matrix(f)
        a = sparse.vstack([a, f])
        b.append(f_opt)
        f = [0.] * len(f)
        nS, nR = a.shape
        for i in range(nR):
            col = sparse.lil_matrix((nS + i, 1))
            a = sparse.hstack([a, col, col])
            row = sparse.lil_matrix((1, nR + 2 * i + 2))
            row[0, i] = 1.
            row[0, nR + 2 * i] = 1.
            row[0, nR + 2 * i + 1] = -1.
            a = sparse.vstack([a, row])
            vlb.append(0.)
            vlb.append(0.)
            vub.append(np.inf)
            vub.append(np.inf)
            f.append(-1.)
            f.append(-1.)
            b.append(0.)
        v_sol = easy_lp(f, a, b, vlb, vub, one=False)[0]
        v = v_sol[:nR]

    return v, f_opt, conv


def easy_milp(f, a, b, vlb, vub, csense, vartype, ic=None):
    '''Optimize MILP using friends of Gurobi.'''
    return easy_milp_sos(f, a, b, vlb, vub, csense, vartype, ic)


def easy_milp_sos(f, a, b, vlb, vub, csense, vartype, ic=None, sos=None):
    '''Optimize MILP using friends of Gurobi.'''

    # catch np arrays
    f, b, vlb, vub = list(f), list(b), list(vlb), list(vub)

    if ic is not None:
        # check initial solution is feasible
        nS, nR = a.shape
        for index in range(nR):
            dL, dU = vlb[index] - ic[index], ic[index] - vub[index]
            if dL > LP_TOL:
                print('LB %g violated [%g]' % (index, dL))
            if dU > LP_TOL:
                print('UB %g violated [%g]' % (index, dU))
        b0 = a * ic
        for index in range(nS):
            if csense[index] == 'E':
                d = abs(b0[index] - b[index])
            elif csense[index] == 'G':
                d = b[index] - b0[index]
            elif csense[index] == 'L':
                d = b0[index] - b[index]
            if d > LP_TOL:
                print('constraint %g (%s) violated [%g]' %
                      (index, csense[index], d))

    # create gurobi model
    milp = gurobipy.Model()
    milp.Params.OutputFlag = 1
    milp.Params.FeasibilityTol = LP_TOL
    milp.Params.IntFeasTol = LP_TOL
    milp.Params.MIPGapAbs = LP_TOL
#
    milp.Params.MIPGap = 1e-3  # call time at 0.1% -> works for fidarestat data
#     milp.Params.MIPGap = 1e-4  # call time at 0.01%

#     milp.Params.timeLimit = 60.*60

    rows, cols = a.shape
    # add variables to model
    for j in range(cols):
        LB = vlb[j]
        if LB == -np.inf:
            LB = -gurobipy.GRB.INFINITY
        UB = vub[j]
        if UB == np.inf:
            UB = gurobipy.GRB.INFINITY
        milp.addVar(lb=LB, ub=UB, obj=f[j], vtype=vartype[j])
    milp.update()
    if ic is not None:
        milpvars = milp.getVars()
        for j in range(cols):
            var = milpvars[j]
            var.setAttr('Start', ic[j])
    if sos is not None:
        milpvars = milp.getVars()
        for X in sos:
            if len(X) == 2:
                x0, x1 = X
                v0, v1 = milpvars[x0], milpvars[x1]
                milp.addSOS(gurobipy.GRB.SOS_TYPE1, [v0, v1])
            elif len(X) == 3:
                x0, x1, x2 = X
                v0, v1, v2 = milpvars[x0], milpvars[x1], milpvars[x2]
                milp.addSOS(gurobipy.GRB.SOS_TYPE2, [v0, v1, v2])
    milp.update()
    milpvars = milp.getVars()
    # iterate over the rows of S adding each row into the model
    S = a.tocsr()
    for i in range(rows):
        start = S.indptr[i]
        end = S.indptr[i + 1]
        variables = [milpvars[j] for j in S.indices[start:end]]
        coeff = S.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        if csense[i] == 'E':
            csensei = gurobipy.GRB.EQUAL
        elif csense[i] == 'G':
            csensei = gurobipy.GRB.GREATER_EQUAL
        elif csense[i] == 'L':
            csensei = gurobipy.GRB.LESS_EQUAL
        milp.addConstr(lhs=expr, sense=csensei, rhs=b[i])
    milp.ModelSense = -1
    milp.update()
    milp.optimize()

    v = np.empty(len(f))
    v[:] = np.nan
    f_opt = np.nan
    conv = False
    if milp.Status in [gurobipy.GRB.OPTIMAL, gurobipy.GRB.TIME_LIMIT]:
        f_opt = milp.ObjVal
        conv = True
        v = [var.x for var in milp.getVars()]

    # remove model: better memory management?
    del milp

    return v, f_opt, conv
