'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import gurobipy
from daaave.data import genes_to_rxns
from daaave.model import convert_sbml_to_cobra, rescale_model
from daaave.solver import easy_lp
import numpy as np
import scipy.sparse as sparse


def daaave(sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale,
           exp_rxn_names, exp_flux, flux_to_scale,
           original_method=False):
    '''Original Daaave method.'''
    # gene data -> reaction data
    rxn_exp, rxn_exp_sd = genes_to_rxns(
        sbml, gene_names, gene_exp, gene_exp_sd)
    # rescale model so rxn_exp and flux = 1 for reaction gene_to_scale
    sbml, rxn_exp, rxn_exp_sd = rescale_model(
        sbml, rxn_exp, rxn_exp_sd, gene_to_scale
    )

    # Gene expression constraint FBA
    flux = _data_to_flux(sbml, rxn_exp, rxn_exp_sd, original_method)

    # rescale
    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
    flux = [value * flux_scale for value in flux]

    return flux


def _data_to_flux(sbml, rxn_exp, rxn_exp_sd, original_method=False):
    '''Daaaaave: predict flux by maximising correlation with data.'''

    model = sbml.getModel()
    nr_old = 0
    cobra = convert_sbml_to_cobra(sbml)
    v_sol = np.zeros(model.getNumReactions())

    while list(cobra['rev']).count(False) > nr_old:
        nr_old = list(cobra['rev']).count(False)

        # 1. fit to data
        N, L, U = cobra['S'].copy(), list(cobra['lb']), list(cobra['ub'])
        f, b = list(cobra['c']), list(cobra['b'])
        f = [0.] * len(f)
        for i in range(model.getNumReactions()):
            data, sd = rxn_exp[i], rxn_exp_sd[i]
            if (not cobra['rev'][i]) and (not np.isnan(data)) and (sd > 0):
                s1, s2 = N.shape
                col = sparse.lil_matrix((s1, 1))
                N = sparse.hstack([N, col, col])
                row = sparse.lil_matrix((1, s2 + 2))
                row[0, i] = 1.
                row[0, s2] = -1.
                row[0, s2 + 1] = 1.
                N = sparse.vstack([N, row])
                L.append(0.)
                L.append(0.)
                U.append(np.inf)
                U.append(np.inf)
                b.append(data)
                f.append(-1. / sd)
                f.append(-1. / sd)

        v, f_opt, conv = easy_lp(f, N, b, L, U)

        if conv:
            v_sol = v[:model.getNumReactions()]

            #  2. run FVA
            col = sparse.lil_matrix(f)
            N = sparse.vstack([N, f])
            b.append(f_opt)

            # create gurobi model
            lp = gurobipy.Model()
            lp.Params.OutputFlag = 0
            rows, cols = N.shape
            for j in range(cols):
                LB = L[j]
                if LB == -np.inf:
                    LB = -gurobipy.GRB.INFINITY
                UB = U[j]
                if UB == np.inf:
                    UB = gurobipy.GRB.INFINITY
                lp.addVar(lb=LB, ub=UB)
            lp.update()
            lpvars = lp.getVars()
            S = N.tocsr()
            for i in range(rows):
                start = S.indptr[i]
                end = S.indptr[i + 1]
                variables = [lpvars[j] for j in S.indices[start:end]]
                coeff = S.data[start:end]
                expr = gurobipy.LinExpr(coeff, variables)
                lp.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=b[i])
            lp.update()
            lp.ModelSense = -1

            for i in range(model.getNumReactions()):
                if cobra['rev'][i]:
                    f = [0.] * len(L)
                    f[i] = -1
                    if L[i] >= 0:
                        f_opt, conv = 0, True
                    else:
                        expr = gurobipy.LinExpr(f, lpvars)
                        lp.setObjective(expr)
                        lp.optimize()
                        conv = (lp.Status == gurobipy.GRB.OPTIMAL)
                        if conv:
                            f_opt = lp.ObjVal
                        else:
                            f_opt = np.nan
                    if conv and (-f_opt >= 0):  # irreversibly forward
                        cobra['lb'][i] = max(cobra['lb'][i], 0.)
                        cobra['rev'][i] = False
                    else:
                        if U[i] <= 0 and not original_method:
                            f_opt, conv = 0, True
                        else:
                            f[i] = 1
                            expr = gurobipy.LinExpr(f, lpvars)
                            lp.setObjective(expr)
                            lp.optimize()
                            conv = (lp.Status == gurobipy.GRB.OPTIMAL)
                            if conv:
                                f_opt = lp.ObjVal
                            else:
                                f_opt = np.nan
                        cond1 = (original_method) and (abs(f_opt) <= 0)
                        cond2 = (not original_method) and (f_opt <= 0)
                        if conv and (cond1 or cond2):  # irreversibly backward
                            cobra['ub'][i] = min(cobra['ub'][i], 0.)
                            cobra['rev'][i] = False
                            rxn_exp[i] = -rxn_exp[i]

    return v_sol
