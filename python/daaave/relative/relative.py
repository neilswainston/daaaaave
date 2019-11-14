'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=chained-comparison
# pylint: disable=consider-using-enumerate
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import os
import re

import gurobipy
from sklearn.metrics import r2_score

from daaave.data import load_flux_data, load_gene_data
from daaave.model import convert_sbml_to_cobra, get_gene_associations, \
    read_sbml
from daaave.runner import easy_milp
import numpy as np
import scipy.sparse as sparse


PATH = os.path.join(os.path.dirname(__file__), '../../data')


def test_comparison_daaave():
    '''Test ComparisonDaaave.'''
    # MODEL
    sbml = read_sbml(os.path.join(PATH, 'yeast_5.21_MCISB.xml'))

    # create some relative daaaaata
    gene_names, gene_exp, gene_exp_sd = [], [], []
    # 75%: condition 1
    gene_names_75, gene_exp_75, gene_exp_sd_75 = load_gene_data(
        os.path.join(PATH, 'genedata_75.txt')
    )
    # 8%: condition 2
    gene_names_85, gene_exp_85, gene_exp_sd_85 = load_gene_data(
        os.path.join(PATH, 'genedata_85.txt')
    )
#     # remove zero entries
#     for gene in [gene_exp_75, gene_exp_sd_75, gene_exp_85, gene_exp_sd_85]:
#         gene[gene == 0] = min(gene[gene != 0])/2
    # ratios: condition 2 / condition 1
    for index_85, gene in enumerate(gene_names_85):
        if gene in gene_names_75:
            index_75 = gene_names_75.index(gene)
            mean85, mean75 = gene_exp_85[index_85], gene_exp_75[index_75]
            sd85, sd75 = gene_exp_sd_85[index_85], gene_exp_sd_75[index_75]
            if 0 not in [mean85, mean75, sd85, sd75]:  # ignore zeros
                # formulae for mean and sd of ratios
                mean = mean85 / mean75
                sd = np.sqrt(mean85 / (sd75**2) +
                             ((sd85**2) / (sd75**4)) * mean75)

                # ignore big/small values
                if 1e-3 <= mean <= 1e3:
                    gene_names.append(gene)
                    gene_exp.append(mean)
                    gene_exp_sd.append(sd)

    # fix glucose inputs
    exp_flux = {'r_1714': (-16.5, -11.0)}

    fluxes = call_comparison_daaave(
        sbml, gene_names, gene_exp, gene_exp_sd, exp_flux)

    # format results
    model = sbml.getModel()
    for ind, percent in enumerate(['75', '85']):
        fluxes_file = 'experimental_fluxes_' + percent + '.txt'
        exp_flux, exp_rxn_names = load_flux_data(
            os.path.join(PATH, fluxes_file))
        mod_ComparisonDaaaaave = np.zeros(len(exp_rxn_names))
        mod_rxn_names = [
            reaction.getName()
            for reaction in model.getListOfReactions()
        ]
        for i in range(len(exp_rxn_names)):
            j = mod_rxn_names.index(exp_rxn_names[i])
            mod_ComparisonDaaaaave[i] = abs(fluxes[j][ind])
        mod_ComparisonDaaaaave[abs(mod_ComparisonDaaaaave) < 1e-6] = 0
        print('%s\t%s\t%s' % ("rxn", "exp", "C'Dave"))
        for i in range(len(exp_rxn_names)):
            print('%s\t%.3g\t%.3g' % (
                exp_rxn_names[i], exp_flux[i], mod_ComparisonDaaaaave[i]))
        print('%s\t%.3g\t%.3g\n' % (
            'R2', 1, r2_score(exp_flux[1:], mod_ComparisonDaaaaave[1:])))

    # rxn    exp    C'Dave
    # D-glucose exchange    16.5    16.5
    # ethanol exchange    23.8    0.0226
    # carbon dioxide exchange    22.7    8.4
    # glycerol exchange    3.54    0
    # acetate exchange    0.311    2.68
    # alpha,alpha-trehalose exchange    0.0356    0.000116
    # (R)-lactate exchange    0.00873    2.21e-06
    # R2    1    -0.174
    #
    # rxn    exp    C'Dave
    # D-glucose exchange    11    11
    # ethanol exchange    13    0.022
    # carbon dioxide exchange    21    6.17
    # glycerol exchange    2.17    0
    # acetate exchange    0.239    1.53
    # alpha,alpha-trehalose exchange    0.0215    1.7e-06
    # (R)-lactate exchange    0.00609    1.7e-06
    # R2    1    -0.00347


def call_comparison_daaave(sbml_in, gene_names, gene_exp, _, exp_flux):
    '''Call ComparisonDaaave.'''
    sbml = sbml_in.clone()
    model = sbml.getModel()

    # which reactions can go?
    bounds = FVA(sbml)  # this is S L O W
    dead = []
    for index in range(len(bounds)):
        LB, UB = bounds[index]
        if (LB >= 0) and (UB <= 0):
            dead.append(index)

    # remove dead reactions, so we can force some to be non-zero later
    # this also shrinks the problem size
    for index in reversed(dead):
        model.removeReaction(index)

    # replace characters in gene identifiers
    gene_names = [gene.replace('-', '_') for gene in gene_names]
    model_grRules = get_gene_associations(sbml)
    model_grRules = [gene.replace('-', '_') for gene in model_grRules]
    model_rxns = [reaction.getId() for reaction in model.getListOfReactions()]

    # get matrices and vectors
    cobra = convert_sbml_to_cobra(sbml)
    S = cobra['S']
    nS, nR = S.shape
    L = np.array(cobra['lb'])
    U = np.array(cobra['ub'])
    fMaxGrowth = np.array(cobra['c'])
    f = np.zeros(len(fMaxGrowth))
    b = np.array(cobra['b'])
    csense = 'E' * nS
    vartype = 'C' * nR

    # remove any pseudo-infinites
    L[L < -500] = -np.inf
    U[U > 500] = np.inf

    # add in experimental fluxes
    L1, L2 = L.copy(), L.copy()
    U1, U2 = U.copy(), U.copy()
    for rxn_id in exp_flux:
        if rxn_id in model_rxns:
            exp_ind = model_rxns.index(rxn_id)
            bound1, bound2 = exp_flux[rxn_id]
            L1[exp_ind] = bound1
            L2[exp_ind] = bound2
            U1[exp_ind] = bound1
            U2[exp_ind] = bound2

    # create dual problem
    S = S = sparse.vstack([
        sparse.hstack([S, sparse.lil_matrix((nS, nR))]),
        sparse.hstack([sparse.lil_matrix((nS, nR)), S])
    ])
    L = np.append(L1, L2)
    U = np.append(U1, U2)
    b = np.append(b, b)
    f = np.append(f, f)
    csense = csense + csense
    vartype = vartype + vartype

    NS, NR = 2 * nS, 2 * nR
    # create positive FBA problem
    S = sparse.vstack([
        sparse.hstack([S, sparse.lil_matrix((NS, 2 * NR))]),
        sparse.hstack([sparse.eye(NR), -sparse.eye(NR), sparse.eye(NR)])
    ])
    L = np.append(L, np.zeros(2 * NR))
    U = np.append(U, np.inf * np.ones(2 * NR))
    b = np.append(b, np.zeros(NR))
    f = np.append(f, np.zeros(2 * NR))
    csense = csense + 'E' * NR
    vartype = vartype + 'C' * 2 * NR

    # only allow positive or negative flux
    M = 1e3 * max(abs(np.hstack([U[np.isfinite(U)], L[np.isfinite(L)]])))
    S = sparse.hstack([S, sparse.lil_matrix((NS + NR, 2 * NR))])
    L = np.append(L, -np.inf * np.ones(2 * NR))
    U = np.append(U, np.inf * np.ones(2 * NR))
    f = np.append(f, np.zeros(2 * NR))
    vartype = vartype + 'B' * 2 * NR

    # p <= M * kP -> -p + M*kP >= 0
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((NR, NR)),
                       -sparse.eye(NR),
                       sparse.lil_matrix((NR, NR)),
                       M * sparse.eye(NR),
                       sparse.lil_matrix((NR, NR))])
    ])
    b = np.append(b, np.zeros(NR))
    csense = csense + 'G' * NR

    # n <= M * kN -> -n + M*kN >= 0
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((NR, NR)),
                       sparse.lil_matrix((NR, NR)),
                       -sparse.eye(NR),
                       sparse.lil_matrix((NR, NR)),
                       M * sparse.eye(NR)])
    ])
    b = np.append(b, np.zeros(NR))
    csense = csense + 'G' * NR

    # kP + kN = 1
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((NR, NR)),
                       sparse.lil_matrix((NR, NR)),
                       sparse.lil_matrix((NR, NR)),
                       sparse.eye(NR),
                       sparse.eye(NR)])
    ])
    b = np.append(b, np.ones(NR))
    csense = csense + 'E' * NR

    # abs(v) variables
    S = sparse.hstack([S, sparse.lil_matrix((NS + 4 * NR, NR))])
    L = np.append(L, -np.inf * np.ones(NR))
    U = np.append(U, np.inf * np.ones(NR))
    f = np.append(f, np.zeros(NR))
    vartype = vartype + 'C' * NR
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((NR, NR)), -sparse.eye(NR),
                       -sparse.eye(NR),
                       sparse.lil_matrix((NR, NR)),
                       sparse.lil_matrix((NR, NR)),
                       sparse.eye(NR)])
    ])
    b = np.append(b, np.zeros(NR))
    csense = csense + 'E' * NR
    abs_flux_index = {}
    for index in range(nR):
        abs_flux_index[model_rxns[index]] = (
            5 * NR + index, 5 * NR + index + nR)

    # add gene fluxes
    nS_all, nR_all = S.shape
    nG = len(gene_names)
    S = sparse.hstack([S, sparse.lil_matrix((nS_all, 2 * nG))])
    L = np.append(L, np.zeros(2 * nG))
    U = np.append(U, np.inf * np.ones(2 * nG))
    f = np.append(f, np.zeros(2 * nG))
    vartype = vartype + 'C' * (2 * nG)
    gene_index = {}
    for index in range(nG):
        gene_index[gene_names[index]] = (nR_all + index, nR_all + index + nG)
    # add genes
    S = sparse.vstack([
        S,
        sparse.hstack(
            [sparse.lil_matrix((2 * nG, nR_all)), -sparse.eye(2 * nG)])
    ])
    b = np.append(b, np.zeros(2 * nG))
    csense = csense + 'E' * (2 * nG)
    gene_sum_index = {}
    for index in range(nG):
        gene_sum_index[gene_names[index]] = (
            nS_all + index, nS_all + index + nG)

    # add data
    S = S.todok()
    for index in range(nG):
        nS_all, nR_all = S.shape
        S.resize((nS_all + 1, nR_all + 2))
        gene = gene_names[index]
        ratio = gene_exp[index]
        # std = gene_exp_sd[index]
        cond_1, cond_2 = gene_index[gene]
        if ratio < 1:
            # ratio = cond_2 / cond_1 -> ratio * cond_1 - cond_2 = 0
            S[nS_all, cond_1] = ratio
            S[nS_all, cond_2] = -1
        else:
            # only use ratios < 1 by swapping
            S[nS_all, cond_2] = 1 / ratio
            S[nS_all, cond_1] = -1
            # std = something
        b = np.append(b, 0)
        csense = csense + 'E'
        S[nS_all, nR_all] = 1
        S[nS_all, nR_all + 1] = -1
        L = np.append(L, [0, 0])
        U = np.append(U, [np.inf, np.inf])
#         f = np.append(f, [-1/std, -1/std])
        f = np.append(f, [-1, -1])
        vartype = vartype + 'CC'

    gene_exp_inv = [1. / gene for gene in gene_exp]
    M = 1e-3 * \
        min(abs(
            np.hstack([U[np.nonzero(U)], L[np.nonzero(L)], gene_exp,
                       gene_exp_inv])))

    for ind_rxn in range(len(model_rxns)):
        association = model_grRules[ind_rxn].strip()

        if association:
            rxn_id = model_rxns[ind_rxn]
            abs_flux_index_1, abs_flux_index_2 = abs_flux_index[rxn_id]

            # get list of genes
            genes = re.findall(r'\b\S+\b', association)
            genes = list(set(genes).intersection(gene_index.keys()))

            for gene in genes:
                gene_index_1, gene_index_2 = gene_index[gene]
                gene_sum_index_1, gene_sum_index_2 = gene_sum_index[gene]
                S[gene_sum_index_1, abs_flux_index_1] = 1
                S[gene_sum_index_2, abs_flux_index_2] = 1
                L[gene_index_1] = max([L[gene_index_1], M])
                L[gene_index_2] = max([L[gene_index_2], M])

    soln, _, _ = easy_milp(f, S, b, L, U, csense, vartype)
#     print 'MILP solution:\t%g\n' %solution_obj

    fluxes = [(soln[i], soln[i + nR]) for i in range(nR)]

    # put back in dead reactions as zero
    all_fluxes = []
    n = 0
    while fluxes:
        if n in dead:
            flux = (0., 0.)
        else:
            flux = fluxes.pop(0)
        all_fluxes.append(flux)
        n += 1

    return all_fluxes


def FVA(sbml):
    '''FVA.'''
    cobra = convert_sbml_to_cobra(sbml)
    a = cobra['S']
    rows, cols = a.shape
    vlb = np.array(cobra['lb'])
    vub = np.array(cobra['ub'])
    b = np.array(cobra['b'])

    # create gurobi model
    lp = gurobipy.Model()
    lp.Params.OutputFlag = 0
    lp.Params.FeasibilityTol = 1e-9  # as per Cobra
    lp.Params.OptimalityTol = 1e-9  # as per Cobra

    # add variables to model
    for j in range(cols):
        LB = vlb[j]
        if LB == -np.inf:
            LB = -gurobipy.GRB.INFINITY
        UB = vub[j]
        if UB == np.inf:
            UB = gurobipy.GRB.INFINITY
        lp.addVar(lb=LB, ub=UB, obj=0.)
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
    lp.update()
    lp.ModelSense = -1

    bounds = []

    for var in lp.getVars():
        # max bound
        #         print dir(var)
        var.setAttr("Obj", 1.)
        lp.update()
        lp.optimize()
        if lp.Status == gurobipy.GRB.OPTIMAL:
            UB = lp.ObjVal
        else:
            UB = np.inf
        # min bound
        var.setAttr("Obj", -1.)
        lp.update()
        lp.optimize()
        if lp.Status == gurobipy.GRB.OPTIMAL:
            LB = -lp.ObjVal
        else:
            LB = -np.inf
        var.setAttr("Obj", 0.)
        lp.update()
        # silly -0.0
        if UB == 0.:
            UB = 0.
        if LB == 0.:
            LB = 0.
        bounds.append((LB, UB))

    return bounds
