"""Implementation of the Daaaaave algorithm.

15-Apr-14:

The Daaaaave algorithm is described in
Lee D, Smallbone K, Dunn WB, Murabito E, Winder CL, Kell DB, Mendes P,
Swainston N (2012) "Improving metabolic flux predictions using absolute
gene expression data" BMC Syst Biol 6:73
http://dx.doi.org/10.1186/1752-0509-6-73

results() will reproduce Tables 1 and 2 of that paper, which compares
Daaaaave with alternatives FBA, Gimme and Shlomi, and with experimental
measurements.

Flagging original_method to True will use a warts-and-all implementation
to ensure identical results to the original Matlab algorithm.

29-May-15:

Implementation of all-improved SuperDaaaaave, translated from the Matlab
function "call_SuperDaaaaave.m" available at
http://github.com/u003f/transcript2flux
"""

# pylint --generate-rcfile
# http://legacy.python.org/dev/peps/pep-0008/

# pylint: disable=broad-except
# pylint: disable=chained-comparison
# pylint: disable=consider-using-enumerate
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import os
import re

import gurobipy
from sklearn.metrics import r2_score
from sympy.logic import boolalg

import numpy as np
from python.data import genes_to_rxns, load_flux_data, load_gene_data
from python.model import get_gene_associations, read_sbml
import scipy.sparse as sparse


# http://www.gurobi.com/documentation/6.0/quickstart_mac/py_python_interface.html
# http://frank-fbergmann.blogspot.co.uk/2014/05/libsbml-python-bindings-5101.html
PATH = os.path.join(os.path.dirname(__file__), 'data')
EPS = 2.**(-52.)

LP_TOL = 1e-6


def test_ComparisonDaaaaave():

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
                if (mean >= 1e-3) and (mean <= 1e3):
                    gene_names.append(gene)
                    gene_exp.append(mean)
                    gene_exp_sd.append(sd)

    # fix glucose inputs
    exp_flux = {'r_1714': (-16.5, -11.0)}

    fluxes = call_ComparisonDaaaaave(
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

    # rxn	exp	C'Dave
    # D-glucose exchange	16.5	16.5
    # ethanol exchange	23.8	0.0226
    # carbon dioxide exchange	22.7	8.4
    # glycerol exchange	3.54	0
    # acetate exchange	0.311	2.68
    # alpha,alpha-trehalose exchange	0.0356	0.000116
    # (R)-lactate exchange	0.00873	2.21e-06
    # R2	1	-0.174
    #
    # rxn	exp	C'Dave
    # D-glucose exchange	11	11
    # ethanol exchange	13	0.022
    # carbon dioxide exchange	21	6.17
    # glycerol exchange	2.17	0
    # acetate exchange	0.239	1.53
    # alpha,alpha-trehalose exchange	0.0215	1.7e-06
    # (R)-lactate exchange	0.00609	1.7e-06
    # R2	1	-0.00347


def call_ComparisonDaaaaave(sbml_in, gene_names, gene_exp, _, exp_flux):

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

    # model = sbml.getModel()

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


def call_SuperDaaaaave_SOS(sbml, gene_names, gene_exp, gene_exp_sd,
                           MaxGrowth=False, UseSD=True, FixScaling=0,
                           TargetFlux=None):
    """
    call_SuperDaaaaave, but replacing the binary variables with Special
    Ordered Set (SOS) parameters
    """

    model = sbml.getModel()

    # replace characters in gene identifiers
    gene_names = [gene.replace('-', '_') for gene in gene_names]
    model_grRules = get_gene_associations(sbml)
    model_grRules = [gene.replace('-', '_') for gene in model_grRules]

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

    # find a feasible solution
    feas0, _, feas_stat = easy_lp(f, S, b, L, U, one=True)
    if not feas_stat:
        print('error:\tmodel infeasible')
    feas0 = np.array(feas0)
    if FixScaling:
        feas0 = feas0 * FixScaling
    feas = np.array(feas0)

    solution_full, solution_obj, statMaxGrowth = easy_lp(
        fMaxGrowth, S, b, L, U, one=True)
    solnMaxGrowth = np.zeros(nR)
    objMaxGrowth = 0.
    if statMaxGrowth == 1.:
        # use max growth if SuperDaaaaave does not converge
        solnMaxGrowth = solution_full
#         objMaxGrowth = np.floor(solution_obj/EPS)*EPS # round down a touch
        if abs(solution_obj) < LP_TOL:
            objMaxGrowth = 0.
        else:
            objMaxGrowth = solution_obj - LP_TOL  # round down a touch

    # create positive FBA problem
    S = sparse.vstack([
        sparse.hstack([S, sparse.lil_matrix((nS, 2 * nR))]),
        sparse.hstack([sparse.eye(nR), -sparse.eye(nR), sparse.eye(nR)])
    ])
    L = np.append(L, np.zeros(2 * nR))
    U = np.append(U, np.inf * np.ones(2 * nR))
    b = np.append(b, np.zeros(nR))
    f = np.append(f, np.zeros(2 * nR))
    csense = csense + 'E' * nR
    vartype = vartype + 'C' * 2 * nR
    x = np.append(feas0, -feas0)
    x[x < 0] = 0
    feas = np.append(feas, x)

    # create SOS relationships
    sos_list = [(index + nR, index + 2 * nR) for index in range(nR)]

    # abs(v) variables
    S = sparse.hstack([S, sparse.lil_matrix((nS + nR, nR))])
    L = np.append(L, -np.inf * np.ones(nR))
    U = np.append(U, np.inf * np.ones(nR))
    f = np.append(f, np.zeros(nR))
    vartype = vartype + 'C' * nR
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)), -
                       sparse.eye(nR), -sparse.eye(nR), sparse.eye(nR)])
    ])
    b = np.append(b, np.zeros(nR))
    csense = csense + 'E' * nR
    model_rxns = [reaction.getId() for reaction in model.getListOfReactions()]
    abs_flux_index = dict(zip(model_rxns, range(3 * nR, 4 * nR)))
    x = abs(feas0)
    feas = np.append(feas, x)

    # add scaling parameter a
    scale_index = 4 * nR
    nS_all, nR_all = S.shape

    S = sparse.hstack([S, sparse.lil_matrix((nS_all, 1))])
    if FixScaling:
        L = np.append(L, FixScaling)
        U = np.append(U, FixScaling)
        feas = np.append(feas, FixScaling)
    else:
        #         L = np.append(L, 0)
        L = np.append(L, 1)  # transcript > flux; avoids a=0 issues
        U = np.append(U, np.inf)
        feas = np.append(feas, 1)
    f = np.append(f, 0)
    vartype = vartype + 'C'

    # v >= a L -> v - a L >= 0
    n = 0
    for i, x in enumerate(L[:nR]):
        if x not in [-np.inf, 0, np.inf]:
            nS_all, nR_all = S.shape
            row = np.zeros(nR_all)
            row[i] = 1
            row[scale_index] = -x
            n += 1
            L[i] = -np.inf
            S = sparse.vstack([S, row])
    b = np.append(b, np.zeros(n))
    csense = csense + 'G' * n

    # v <= a U -> - v + a U >= 0
    n = 0
    for i, x in enumerate(U[:nR]):
        if x not in [-np.inf, 0, np.inf]:
            nS_all, nR_all = S.shape
            row = np.zeros(nR_all)
            row[i] = -1
            row[scale_index] = x
            n += 1
            U[i] = np.inf
            S = sparse.vstack([S, row])
    b = np.append(b, np.zeros(n))
    csense = csense + 'G' * n

    # add slack variables for genes
    nS_all, nR_all = S.shape
    nG = len(gene_names)
    S = sparse.hstack([S, sparse.lil_matrix((nS_all, nG))])
    L = np.append(L, np.zeros(nG))
    U = np.append(U, np.inf * np.ones(nG))
    f = np.append(f, np.zeros(nG))
    vartype = vartype + 'C' * nG
    feas = np.append(feas, gene_exp)

    # add genes
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nG, nR_all)), sparse.eye(nG)])
    ])
    b = np.append(b, gene_exp)
    csense = csense + 'E' * nG
    gene_index = dict(zip(gene_names, range(nS_all, nS_all + nG)))

    # add gene associations
    gene_exp_sd = dict(zip(gene_names, gene_exp_sd))

    rxn_index = {}
    rxn_std = {}

    S = S.todok()

    for ind_rxn in range(len(model_rxns)):
        association = model_grRules[ind_rxn].strip()
        if association:

            rxn_id = model_rxns[ind_rxn]
            nS_all, nR_all = S.shape
            rxn_row = nS_all
            rxn_col = nR_all

            # add reaction
            S.resize((nS_all + 1, nR_all + 1))
            S[rxn_row, rxn_col] = -1
            L = np.append(L, -np.inf)
            U = np.append(U, np.inf)
            f = np.append(f, 1)
            vartype = vartype + 'C'
            b = np.append(b, 0)
            csense = csense + 'E'
            feas = np.append(feas, 0)

            rxn_index[rxn_id] = rxn_col

            # convert to DNF (or of ands)
            association = to_dnf(association)
            list_of_ors = re.split(' or ', association)

            std_out = 0

            for association_or in list_of_ors:

                # add complex
                nS_all, nR_all = S.shape
                complex_col = nR_all
                S.resize((nS_all, nR_all + 1))
                S[rxn_row, complex_col] = 1
                L = np.append(L, 0)
                U = np.append(U, np.inf)
                f = np.append(f, 0)
                vartype = vartype + 'C'
                feas = np.append(feas, 0)

                if association_or[0] == '(':
                    association_or = association_or[1:-1]

                list_of_ands = re.split(' and ', association_or)

                std_in = np.inf

                for gene in list_of_ands:

                    nS_all, nR_all = S.shape
                    gene_col = nR_all
                    ineq_row = nS_all
                    # complex < gene -> gene - complex > 0
                    S.resize((nS_all + 1, nR_all + 1))
                    S[ineq_row, gene_col] = 1
                    S[ineq_row, complex_col] = -1
                    L = np.append(L, 0)
                    U = np.append(U, np.inf)
                    f = np.append(f, 0)
                    vartype = vartype + 'C'
                    feas = np.append(feas, 0)
                    b = np.append(b, 0)
                    csense = csense + 'G'

                    index = gene_index[gene]
                    S[index, gene_col] = 1

                    std = gene_exp_sd[gene]
                    std_in = min([std_in, std])

                std_out = np.sqrt(std_out**2 + std_in**2)

            rxn_std[rxn_id] = std_out

    if statMaxGrowth and MaxGrowth:
        # set max growth as constraint
        fMaxGrowthBig = np.zeros((1, len(f)))
        fMaxGrowthBig[0, :len(fMaxGrowth)] = fMaxGrowth
        fMaxGrowthBig[0, scale_index] = -objMaxGrowth
        S = sparse.vstack([S, fMaxGrowthBig])
        b = np.append(b, 0)
        csense = csense + 'E'

    print('\n%s\n' % ('maximising total reaction gene content'))
    soln, obj, _ = easy_milp_sos(
        f, S, b, L, U, csense, vartype, ic=feas, sos=sos_list)

    # set objective as constraint
#     obj = np.floor(obj/EPS)*EPS # round down a touch
    if abs(obj) < LP_TOL:
        obj = 0.
    else:
        obj = obj - LP_TOL  # round down a touch
    S = sparse.vstack([S, f])
    b = np.append(b, obj)
    csense = csense + 'G'

    # minimise distance from data to flux
    f = np.zeros(len(f))
    S = S.todok()
    init = np.array(soln)
    for ind in range(nR):
        rxn_id = model_rxns[ind]
        if rxn_id in rxn_index.keys():
            nS_all, nR_all = S.shape
            # R - D = P - N -> R - D - P + N = 0
            S.resize((nS_all + 1, nR_all + 2))
            rxn_ind = nS_all
            S[rxn_ind, abs_flux_index[rxn_id]] = 1
            S[rxn_ind, rxn_index[rxn_id]] = -1
            S[rxn_ind, nR_all] = -1
            S[rxn_ind, nR_all + 1] = 1
            L = np.append(L, [0, 0])
            U = np.append(U, [np.inf, np.inf])
            if UseSD:
                std = rxn_std[rxn_id]
            else:
                std = 1.
            f = np.append(f, [-1. / std, -1. / std])
            vartype = vartype + 'C' * 2
            b = np.append(b, 0)
            csense = csense + 'E'

            # start from feasible solution
            R = soln[abs_flux_index[rxn_id]]
            D = soln[rxn_index[rxn_id]]
            X = R - D
            if X >= 0:
                p, n = X, 0
            else:
                p, n = 0, -X
            init = np.append(init, [p, n])

    print('\n%s\n' % ('minimising distance gene to flux'))
    soln, obj, conv = easy_milp_sos(
        f, S, b, L, U, csense, vartype, ic=init, sos=sos_list)

    if conv:
        #         obj = np.floor(obj/EPS)*EPS # round down a touch
        if abs(obj) < LP_TOL:
            obj = 0.
        else:
            obj = obj - LP_TOL  # round down a touch
        S = sparse.vstack([S, f])
        b = np.append(b, obj)
        csense = csense + 'G'
        f = np.zeros(len(f))

        # minimise distance to target flux
        if not TargetFlux:
            TargetFlux = np.zeros(nR)
        TargetFlux = np.array(TargetFlux)

        # fix rescaling
        scaling_factor = soln[scale_index]
        L[scale_index] = scaling_factor
        U[scale_index] = scaling_factor

        TargetFlux = TargetFlux * scaling_factor

        # v - TargetFlux = P - N -> v - P + N = TargetFlux
        nS_all, nR_all = S.shape
        S = sparse.vstack([
            sparse.hstack([S, sparse.lil_matrix((nS_all, 2 * nR))]),
            sparse.hstack([sparse.eye(nR), sparse.lil_matrix(
                (nR, nR_all - nR)), -sparse.eye(nR), sparse.eye(nR)])
        ])
        L = np.append(L, np.zeros(2 * nR))
        U = np.append(U, np.inf * np.ones(2 * nR))
        b = np.append(b, TargetFlux)
        f = np.append(f, -np.ones(2 * nR))
        csense = csense + 'E' * nR
        vartype = vartype + 'C' * 2 * nR

        # start from feasible solution
        p0 = soln[:nR] - TargetFlux
        p0[p0 < 0] = 0.
        n0 = -(soln[:nR] - TargetFlux)
        n0[n0 < 0] = 0.
        init = np.append(soln, np.append(p0, n0))

        print('\n%s\n' % ('minimising distance to target flux'))
        soln_min, solution_obj_min, conv_min = easy_milp_sos(
            f, S, b, L, U, csense, vartype, ic=init, sos=sos_list)
        if conv_min:
            soln, solution_obj, conv = soln_min, solution_obj_min, conv_min

    # rescale
    if conv:
        scaling_factor = soln[scale_index]
        fluxes = [soln[i] / scaling_factor for i in range(nR)]
    else:
        scaling_factor = np.nan
        fluxes = solnMaxGrowth

    return fluxes, scaling_factor


def call_SuperDaaaaave(sbml, gene_names, gene_exp, gene_exp_sd,
                       MaxGrowth=False, UseSD=True, FixScaling=0,
                       TargetFlux=None):
    """
    Implementation of all-improved SuperDaaaaave, translated from the Matlab
    function
    call_SuperDaaaaave.m available at http://github.com/u003f/transcript2flux
    29 May 2015
    MaxGrowth: [1/0] find solution that maximises growth
    FixScaling: fixed rescaling between transcript and flux data
    """

    BIG_NUMBER = 1e3  # -> works for fidarestat
#
    model = sbml.getModel()
#
    # replace characters in gene identifiers
    gene_names = [gene.replace('-', '_') for gene in gene_names]
    model_grRules = get_gene_associations(sbml)
    model_grRules = [gene.replace('-', '_') for gene in model_grRules]
#
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
#
    # remove any pseudo-infinites
    L[L < -500] = -np.inf
    U[U > 500] = np.inf
#
    # find a feasible solution
    feas0, _, feas_stat = easy_lp(f, S, b, L, U, one=True)
    if not feas_stat:
        print('error:\tmodel infeasible')
    feas0 = np.array(feas0)
    if FixScaling:
        feas0 = feas0 * FixScaling
    feas = np.array(feas0)
#
    solution_full, solution_obj, statMaxGrowth = easy_lp(
        fMaxGrowth, S, b, L, U, one=True)
    solnMaxGrowth = np.zeros(nR)
    objMaxGrowth = 0.
    if statMaxGrowth == 1.:
        # use max growth if SuperDaaaaave does not converge
        solnMaxGrowth = solution_full
#         objMaxGrowth = np.floor(solution_obj/EPS)*EPS # round down a touch
        if abs(solution_obj) < LP_TOL:
            objMaxGrowth = 0.
        else:
            objMaxGrowth = solution_obj - LP_TOL  # round down a touch
#
    # create positive FBA problem
    S = sparse.vstack([
        sparse.hstack([S, sparse.lil_matrix((nS, 2 * nR))]),
        sparse.hstack([sparse.eye(nR), -sparse.eye(nR), sparse.eye(nR)])
    ])
    L = np.append(L, np.zeros(2 * nR))
    U = np.append(U, np.inf * np.ones(2 * nR))
    b = np.append(b, np.zeros(nR))
    f = np.append(f, np.zeros(2 * nR))
    csense = csense + 'E' * nR
    vartype = vartype + 'C' * 2 * nR
    x = np.append(feas0, -feas0)
    x[x < 0] = 0
    feas = np.append(feas, x)
#
    # only allow positive or negative flux
    M = BIG_NUMBER * \
        max(abs(np.hstack([U[np.isfinite(U)], L[np.isfinite(L)], gene_exp])))
    S = sparse.hstack([S, sparse.lil_matrix((nS + nR, 2 * nR))])
    L = np.append(L, -np.inf * np.ones(2 * nR))
    U = np.append(U, np.inf * np.ones(2 * nR))
    f = np.append(f, np.zeros(2 * nR))
    vartype = vartype + 'B' * 2 * nR
    x = np.append(feas0 >= 0, feas0 < 0)
    feas = np.append(feas, x)
#
    # p <= M * kP -> -p + M*kP >= 0
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)),
                       -sparse.eye(nR),
                       sparse.lil_matrix((nR, nR)),
                       M * sparse.eye(nR),
                       sparse.lil_matrix((nR, nR))])
    ])
    b = np.append(b, np.zeros(nR))
    csense = csense + 'G' * nR
#
    # n <= M * kN -> -n + M*kN >= 0
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)),
                       sparse.lil_matrix((nR, nR)),
                       -sparse.eye(nR),
                       sparse.lil_matrix((nR, nR)),
                       M * sparse.eye(nR)])
    ])
    b = np.append(b, np.zeros(nR))
    csense = csense + 'G' * nR
#
    # kP + kN = 1
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)),
                       sparse.lil_matrix((nR, nR)),
                       sparse.lil_matrix((nR, nR)),
                       sparse.eye(nR),
                       sparse.eye(nR)])
    ])
    b = np.append(b, np.ones(nR))
    csense = csense + 'E' * nR
#
    # abs(v) variables
    S = sparse.hstack([S, sparse.lil_matrix((nS + 4 * nR, nR))])
    L = np.append(L, -np.inf * np.ones(nR))
    U = np.append(U, np.inf * np.ones(nR))
    f = np.append(f, np.zeros(nR))
    vartype = vartype + 'C' * nR
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)), -sparse.eye(nR),
                       -sparse.eye(nR),
                       sparse.lil_matrix((nR, nR)),
                       sparse.lil_matrix((nR, nR)),
                       sparse.eye(nR)])
    ])
    b = np.append(b, np.zeros(nR))
    csense = csense + 'E' * nR
    model_rxns = [reaction.getId() for reaction in model.getListOfReactions()]
    abs_flux_index = dict(zip(model_rxns, range(5 * nR, 6 * nR)))
    x = abs(feas0)
    feas = np.append(feas, x)
#
    # add scaling parameter a
    scale_index = 6 * nR
    nS_all, nR_all = S.shape
#
    S = sparse.hstack([S, sparse.lil_matrix((nS_all, 1))])
    if FixScaling:
        L = np.append(L, FixScaling)
        U = np.append(U, FixScaling)
        feas = np.append(feas, FixScaling)
    else:
        #         L = np.append(L, 0)
        L = np.append(L, 1)  # transcript > flux; avoids a=0 issues
        U = np.append(U, np.inf)
        feas = np.append(feas, 1)
    f = np.append(f, 0)
    vartype = vartype + 'C'
#
    # v >= a L -> v - a L >= 0
    n = 0
    for i, x in enumerate(L[:nR]):
        if x not in [-np.inf, 0, np.inf]:
            nS_all, nR_all = S.shape
            row = np.zeros(nR_all)
            row[i] = 1
            row[scale_index] = -x
            n += 1
            L[i] = -np.inf
            S = sparse.vstack([S, row])
    b = np.append(b, np.zeros(n))
    csense = csense + 'G' * n
#
    # v <= a U -> - v + a U >= 0
    n = 0
    for i, x in enumerate(U[:nR]):
        if x not in [-np.inf, 0, np.inf]:
            nS_all, nR_all = S.shape
            row = np.zeros(nR_all)
            row[i] = -1
            row[scale_index] = x
            n += 1
            U[i] = np.inf
            S = sparse.vstack([S, row])
    b = np.append(b, np.zeros(n))
    csense = csense + 'G' * n
#
    # add slack variables for genes
    nS_all, nR_all = S.shape
    nG = len(gene_names)
    S = sparse.hstack([S, sparse.lil_matrix((nS_all, nG))])
    L = np.append(L, np.zeros(nG))
    U = np.append(U, np.inf * np.ones(nG))
    f = np.append(f, np.zeros(nG))
    vartype = vartype + 'C' * nG
    feas = np.append(feas, gene_exp)
#
    # add genes
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nG, nR_all)), sparse.eye(nG)])
    ])
    b = np.append(b, gene_exp)
    csense = csense + 'E' * nG
    gene_index = dict(zip(gene_names, range(nS_all, nS_all + nG)))
#
    # add gene associations
    gene_exp_sd = dict(zip(gene_names, gene_exp_sd))
#
    rxn_index = {}
    rxn_std = {}
#
    S = S.todok()
#
    for ind_rxn in range(len(model_rxns)):
        association = model_grRules[ind_rxn].strip()
        if association:
            #
            rxn_id = model_rxns[ind_rxn]
            nS_all, nR_all = S.shape
            rxn_row = nS_all
            rxn_col = nR_all
#
            # add reaction
            S.resize((nS_all + 1, nR_all + 1))
            S[rxn_row, rxn_col] = -1
            L = np.append(L, -np.inf)
            U = np.append(U, np.inf)
            f = np.append(f, 1)
            vartype = vartype + 'C'
            b = np.append(b, 0)
            csense = csense + 'E'
            feas = np.append(feas, 0)
#
            rxn_index[rxn_id] = rxn_col
#
            # convert to DNF (or of ands)
            association = to_dnf(association)
            list_of_ors = re.split(' or ', association)
#
            std_out = 0
#
            for association_or in list_of_ors:
                #
                # add complex
                nS_all, nR_all = S.shape
                complex_col = nR_all
                S.resize((nS_all, nR_all + 1))
                S[rxn_row, complex_col] = 1
                L = np.append(L, 0)
                U = np.append(U, np.inf)
                f = np.append(f, 0)
                vartype = vartype + 'C'
                feas = np.append(feas, 0)
#
                if association_or[0] == '(':
                    association_or = association_or[1:-1]
#
                list_of_ands = re.split(' and ', association_or)
#
                std_in = np.inf
#
                for gene in list_of_ands:
                    #
                    nS_all, nR_all = S.shape
                    gene_col = nR_all
                    ineq_row = nS_all
                    # complex < gene -> gene - complex > 0
                    S.resize((nS_all + 1, nR_all + 1))
                    S[ineq_row, gene_col] = 1
                    S[ineq_row, complex_col] = -1
                    L = np.append(L, 0)
                    U = np.append(U, np.inf)
                    f = np.append(f, 0)
                    vartype = vartype + 'C'
                    feas = np.append(feas, 0)
                    b = np.append(b, 0)
                    csense = csense + 'G'
#
                    index = gene_index[gene]
                    S[index, gene_col] = 1
#
                    std = gene_exp_sd[gene]
                    std_in = min([std_in, std])
#
                std_out = np.sqrt(std_out**2 + std_in**2)
#
            rxn_std[rxn_id] = std_out
#
    if statMaxGrowth and MaxGrowth:
        # set max growth as constraint
        fMaxGrowthBig = np.zeros((1, len(f)))
        fMaxGrowthBig[0, :len(fMaxGrowth)] = fMaxGrowth
        fMaxGrowthBig[0, scale_index] = -objMaxGrowth
        S = sparse.vstack([S, fMaxGrowthBig])
        b = np.append(b, 0)
        csense = csense + 'E'
#
    print('\n%s\n' % ('maximising total reaction gene content'))
    soln, obj, _ = easy_milp(f, S, b, L, U, csense, vartype, ic=feas)
#
    # set objective as constraint
#     obj = np.floor(obj/EPS)*EPS # round down a touch
    if abs(obj) < LP_TOL:
        obj = 0.
    else:
        obj = obj - LP_TOL  # round down a touch
    S = sparse.vstack([S, f])
    b = np.append(b, obj)
    csense = csense + 'G'
#

    # minimise distance from data to flux
    f = np.zeros(len(f))
    S = S.todok()
    init = np.array(soln)
    for ind in range(nR):
        rxn_id = model_rxns[ind]
        if rxn_id in rxn_index.keys():
            nS_all, nR_all = S.shape
            # R - D = P - N -> R - D - P + N = 0
            S.resize((nS_all + 1, nR_all + 2))
            rxn_ind = nS_all
            S[rxn_ind, abs_flux_index[rxn_id]] = 1
            S[rxn_ind, rxn_index[rxn_id]] = -1
            S[rxn_ind, nR_all] = -1
            S[rxn_ind, nR_all + 1] = 1
            L = np.append(L, [0, 0])
            U = np.append(U, [np.inf, np.inf])
            if UseSD:
                std = rxn_std[rxn_id]
            else:
                std = 1.
            f = np.append(f, [-1. / std, -1. / std])
            vartype = vartype + 'C' * 2
            b = np.append(b, 0)
            csense = csense + 'E'

            # start from feasible solution
            R = soln[abs_flux_index[rxn_id]]
            D = soln[rxn_index[rxn_id]]
            X = R - D
            if X >= 0:
                p, n = X, 0
            else:
                p, n = 0, -X
            init = np.append(init, [p, n])

    print('\n%s\n' % ('minimising distance gene to flux'))
    soln, obj, conv = easy_milp_sos(f, S, b, L, U, csense, vartype, ic=init)
#
    if conv:
        #         obj = np.floor(obj/EPS)*EPS # round down a touch
        if abs(obj) < LP_TOL:
            obj = 0.
        else:
            obj = obj - LP_TOL  # round down a touch
        S = sparse.vstack([S, f])
        b = np.append(b, obj)
        csense = csense + 'G'
        f = np.zeros(len(f))
#
        # minimise distance to target flux
        if not TargetFlux:
            TargetFlux = np.zeros(nR)
        TargetFlux = np.array(TargetFlux)
#
        # fix rescaling
        scaling_factor = soln[scale_index]
        L[scale_index] = scaling_factor
        U[scale_index] = scaling_factor
#
        TargetFlux = TargetFlux * scaling_factor
#
        # v - TargetFlux = P - N -> v - P + N = TargetFlux
        nS_all, nR_all = S.shape
        S = sparse.vstack([
            sparse.hstack([S, sparse.lil_matrix((nS_all, 2 * nR))]),
            sparse.hstack([sparse.eye(nR), sparse.lil_matrix(
                (nR, nR_all - nR)), -sparse.eye(nR), sparse.eye(nR)])
        ])
        L = np.append(L, np.zeros(2 * nR))
        U = np.append(U, np.inf * np.ones(2 * nR))
        b = np.append(b, TargetFlux)
        f = np.append(f, -np.ones(2 * nR))
        csense = csense + 'E' * nR
        vartype = vartype + 'C' * 2 * nR
#
        # start from feasible solution
        p0 = soln[:nR] - TargetFlux
        p0[p0 < 0] = 0.
        n0 = -(soln[:nR] - TargetFlux)
        n0[n0 < 0] = 0.
        init = np.append(soln, np.append(p0, n0))
#
        print('\n%s\n' % ('minimising distance to target flux'))
        soln_min, solution_obj_min, conv_min = easy_milp(
            f, S, b, L, U, csense, vartype, ic=init)
        if conv_min:
            soln, solution_obj, conv = soln_min, solution_obj_min, conv_min
#
    # rescale
    if conv:
        scaling_factor = soln[scale_index]
        fluxes = [soln[i] / scaling_factor for i in range(nR)]
    else:
        scaling_factor = np.nan
        fluxes = solnMaxGrowth
#
    return fluxes, scaling_factor


def to_dnf(association):

    # A and B and (C or D) or E
    association = association.replace(' AND ', ' & ').replace(
        ' OR ', ' | ').replace(' and ', ' & ').replace(' or ', ' | ')
    # -> A & B & (C | D) | E
    association = str(boolalg.to_dnf(association))
    # -> Or(And(A, B, C), And(A, B, D), E)
    for and_old in re.findall(r'And\([^)]+\)', association):
        and_new = and_old
        and_new = and_new.replace('And(', '(')
        and_new = and_new.replace(', ', ' and ')
        association = association.replace(and_old, and_new)
    # -> Or((A and B and C), (A and B and D), E)
    association = association.replace(', ', ' or ')
    if association[:3] == 'Or(':
        association = association[3:-1]
    # .. -> (A and B) or (A and C) or D

    if association == 'False':
        association = ''
    return association


def SuperDaaaaave(model_file, genes_file, fluxes_file, flux_to_scale):

    # MODEL
    # sbml = read_sbml(os.path.join(PATH, model_file))

    # gene data
    gene_names, gene_exp, gene_exp_sd = load_gene_data(
        os.path.join(PATH, genes_file)
    )
    # flux data
    exp_flux, exp_rxn_names = load_flux_data(os.path.join(PATH, fluxes_file))

    # set SBML input to experimental value
    sbml = read_sbml(os.path.join(PATH, model_file))
    rescale_SBML(sbml, exp_rxn_names, exp_flux, flux_to_scale)

    flux, _ = call_SuperDaaaaave(
        sbml, gene_names, gene_exp, gene_exp_sd, MaxGrowth=False)

    return flux


def results():
    """Print tables 1 and 2 of Daaaaave et al."""

    print_results(
        'example.xml',
        'genedata_example_1.txt', 'experimental_fluxes_example_1.txt',
        'rA', 'rA'
    )

#     print_results(
#         'example.xml',
#         'genedata_example_2.txt',
#         'experimental_fluxes_example_2.txt',
#         'rA', 'rA'
#         )
#
#     start = time.clock()
    print_results(
        'yeast_5.21_MCISB.xml',
        'genedata_75.txt', 'experimental_fluxes_75.txt',
        'glucose transport', 'D-glucose exchange',
        original_method=True
    )
#     print_results(
#         'yeast_5.21_MCISB.xml',
#         'genedata_85.txt', 'experimental_fluxes_85.txt',
#         'glucose transport', 'D-glucose exchange',
#         original_method=True
#         )
#     print "%g seconds elapsed\n" % (time.clock() - start)

#     start = time.clock()
#     print_results(
#         'yeast_7.5_cobra.xml',
#         'genedata_75.txt', 'experimental_fluxes_75.txt',
#         'glucose transport', 'D-glucose exchange',
#         original_method=True
#         )
#     print_results(
#         'yeast_7.5_cobra.xml',
#         'genedata_85.txt', 'experimental_fluxes_85.txt',
#         'glucose transport', 'D-glucose exchange',
#         original_method=True
#         )
#     print "%g seconds elapsed\n" % (time.clock() - start)


def print_results(
        model_file, genes_file, fluxes_file,
        gene_to_scale, flux_to_scale, original_method=False):
    """Format output as per Daaaaave et al."""

    (rxn_names, exp_flux, mod_SuperDaaaaave, mod_daaaaave,
     mod_fba, mod_fba_best, mod_gimme) = analysis(
         model_file, genes_file, fluxes_file,
         gene_to_scale, flux_to_scale, original_method)
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
        "rxn", "exp", "S'Dave", "Dave",
        "FBA", "fFBA", "Gimme"))
    for i in range(len(rxn_names)):
        print('%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g' % (
            rxn_names[i], exp_flux[i], mod_SuperDaaaaave[i], mod_daaaaave[i],
            mod_fba[i], mod_fba_best[i], mod_gimme[i]))
    print('%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n' % (
        'R2',
        1,
        r2_score(exp_flux, mod_SuperDaaaaave),
        r2_score(exp_flux, mod_daaaaave),
        r2_score(exp_flux, mod_fba),
        r2_score(exp_flux, mod_fba_best),
        r2_score(exp_flux, mod_gimme)))


def OriginalDaaaaave(sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale,
                     exp_rxn_names, exp_flux, flux_to_scale,
                     original_method=False):

    # gene data -> reaction data
    rxn_exp, rxn_exp_sd = genes_to_rxns(
        sbml, gene_names, gene_exp, gene_exp_sd)
    # rescale model so rxn_exp and flux = 1 for reaction gene_to_scale
    sbml, rxn_exp, rxn_exp_sd = rescale_model(
        sbml, rxn_exp, rxn_exp_sd, gene_to_scale
    )

    # Gene expression constraint FBA
    flux = data_to_flux(sbml, rxn_exp, rxn_exp_sd, original_method)

    # rescale
    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
    flux = [value * flux_scale for value in flux]

    return flux


def GimmeGimmeGimme(sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale,
                    exp_rxn_names, exp_flux, flux_to_scale,
                    original_method=False):

    cutoff = 0.25  # set threshold at lower quartile
    req_fun = 0.9  # force 90% growth

    # gene data -> reaction data
    rxn_exp, rxn_exp_sd = genes_to_rxns(
        sbml, gene_names, gene_exp, gene_exp_sd)

    # rescale model so rxn_exp and flux = 1 for reaction gene_to_scale
    sbml, rxn_exp, rxn_exp_sd = rescale_model(
        sbml, rxn_exp, rxn_exp_sd, gene_to_scale
    )

    # gimme
    flux = gimme(sbml, rxn_exp, cutoff, req_fun, original_method)

    # rescale
    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
    flux = [value * flux_scale for value in flux]

    return flux


def OriginalFBA(sbml, exp_rxn_names, exp_flux, flux_to_scale,
                original_method=False):

    flux, _ = optimize_cobra_model(sbml, original_method)

    # rescale
    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
    flux = [value * flux_scale for value in flux]

    return flux


def FittedFBA(sbml, gene_to_scale, exp_rxn_names, exp_flux, flux_to_scale,
              original_method=False):

    data = create_data_array(
        sbml, exp_flux[:], exp_rxn_names[:], gene_to_scale)

    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
    flux = fba_fitted(sbml, data / flux_scale, original_method)

    # rescale
    flux = [value * flux_scale for value in flux]

    return flux


def rescale_SBML(sbml, exp_rxn_names, exp_flux, flux_to_scale):

    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]

    model = sbml.getModel()
    for reaction in model.getListOfReactions():
        name = reaction.getName()
        if name == flux_to_scale:
            kineticLaw = reaction.getKineticLaw()
            LB = kineticLaw.getParameter('LOWER_BOUND')
            UB = kineticLaw.getParameter('UPPER_BOUND')

            if (LB.getValue() > -500) and (LB.getValue() < 0):
                flux_scale = - abs(flux_scale)
            elif UB.getValue() < 0:
                flux_scale = - abs(flux_scale)
            else:
                flux_scale = abs(flux_scale)

            LB.setValue(flux_scale)
            UB.setValue(flux_scale)

#    # test
#    print optimize_cobra_model(sbml, original_method=True)[1]


def analysis(
        model_file, genes_file, fluxes_file,
        gene_to_scale, flux_to_scale, original_method=False):
    """Run all analyses on input files."""

    # SuperDaaaaave!
    # v_SuperDaaaaave = SuperDaaaaave(
    #    model_file, genes_file, fluxes_file, flux_to_scale)

    # MODEL
    sbml = read_sbml(os.path.join(PATH, model_file))
    v_SuperDaaaaave = [0] * sbml.getModel().getNumReactions()

    # gene data
    gene_names, gene_exp, gene_exp_sd = load_gene_data(
        os.path.join(PATH, genes_file)
    )
    # flux data
    exp_flux, exp_rxn_names = load_flux_data(os.path.join(PATH, fluxes_file))

    # OriginalDaaaaave
    v_gene_exp = OriginalDaaaaave(
        sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale,
        exp_rxn_names, exp_flux, flux_to_scale, original_method
    )

    # GimmeGimmeGimme
    v_gimme = GimmeGimmeGimme(
        sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale,
        exp_rxn_names, exp_flux, flux_to_scale, original_method
    )

    # OriginalFBA
    v_fba = OriginalFBA(sbml, exp_rxn_names, exp_flux,
                        flux_to_scale, original_method)

    # find best fit from standard FBA solution
    v_fba_best = FittedFBA(sbml, gene_to_scale, exp_rxn_names,
                           exp_flux, flux_to_scale, original_method)

    # compare
    mod_SuperDaaaaave, mod_daaaaave, mod_fba, mod_gimme, mod_fba_best = \
        format_results(
            sbml, exp_rxn_names, v_SuperDaaaaave,
            v_gene_exp, v_fba, v_gimme, v_fba_best
        )

    return exp_rxn_names, exp_flux, mod_SuperDaaaaave, mod_daaaaave, \
        mod_fba, mod_fba_best, mod_gimme


def rescale_model(sbml, rxn_exp, rxn_exp_sd, gene_to_scale):

    model = sbml.getModel()
    rxn_names = [
        reaction.getName()
        for reaction in model.getListOfReactions()
    ]
    uptake = rxn_names.index(gene_to_scale)
    rxn_exp_sd = rxn_exp_sd / rxn_exp[uptake]
    rxn_exp = rxn_exp / rxn_exp[uptake]
    reaction = model.getReaction(uptake)
    kinetic_law = reaction.getKineticLaw()
    kinetic_law.getParameter('LOWER_BOUND').setValue(1)
    kinetic_law.getParameter('UPPER_BOUND').setValue(1)

    return sbml, rxn_exp, rxn_exp_sd


def create_data_array(sbml, exp_flux, exp_rxn_names, gene_to_scale):

    exp_rxn_names.append(gene_to_scale)
    exp_flux.append(1.)
    model = sbml.getModel()
    mod_rxn_names = [
        reaction.getName()
        for reaction in model.getListOfReactions()
    ]
    data = np.empty(model.getNumReactions())
    data[:] = np.nan
    for i in range(len(exp_rxn_names)):
        j = mod_rxn_names.index(exp_rxn_names[i])
        data[j] = exp_flux[i]
    return data


def format_results(
        sbml, exp_rxn_names, v_SuperDaaaaave,
        v_gene_exp, v_fba, v_gimme, v_fba_best):

    mod_SuperDaaaaave = np.zeros(len(exp_rxn_names))
    mod_daaaaave = np.zeros(len(exp_rxn_names))
    mod_fba = np.zeros(len(exp_rxn_names))
    mod_gimme = np.zeros(len(exp_rxn_names))
    mod_fba_best = np.zeros(len(exp_rxn_names))

    model = sbml.getModel()
    mod_rxn_names = [
        reaction.getName()
        for reaction in model.getListOfReactions()
    ]

    for i in range(len(exp_rxn_names)):
        j = mod_rxn_names.index(exp_rxn_names[i])
        mod_SuperDaaaaave[i] = abs(v_SuperDaaaaave[j])
        mod_daaaaave[i] = abs(v_gene_exp[j])
        mod_fba[i] = abs(v_fba[j])
        mod_gimme[i] = abs(v_gimme[j])
        mod_fba_best[i] = abs(v_fba_best[j])

    mod_SuperDaaaaave[abs(mod_SuperDaaaaave) < 1e-6] = 0
    mod_daaaaave[abs(mod_daaaaave) < 1e-6] = 0
    mod_fba[abs(mod_fba) < 1e-6] = 0
    mod_gimme[abs(mod_gimme) < 1e-6] = 0
    mod_fba_best[abs(mod_fba_best) < 1e-6] = 0

    return mod_SuperDaaaaave, mod_daaaaave, mod_fba, mod_gimme, mod_fba_best


def data_to_flux(sbml, rxn_exp, rxn_exp_sd, original_method=False):
    """Daaaaave: predict flux by maximising correlation with data."""

    model = sbml.getModel()
    nr_old = 0
    if original_method:
        bound = 1000
    else:
        bound = np.inf
    cobra = convert_sbml_to_cobra(sbml, bound)
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


def convert_sbml_to_cobra(sbml, bound=np.inf):
    """Get Cobra matrices from SBML model."""
    model = sbml.getModel()
    S = sparse.lil_matrix((model.getNumSpecies(), model.getNumReactions()))
    lb, ub, c, b, rev, sIDs = [], [], [], [], [], []
    for species in model.getListOfSpecies():
        sIDs.append(species.getId())
        b.append(0.)
    sIDs = [species.getId() for species in model.getListOfSpecies()]
    for j, reaction in enumerate(model.getListOfReactions()):
        for reactant in reaction.getListOfReactants():
            sID = reactant.getSpecies()
            s = reactant.getStoichiometry()
            if not model.getSpecies(sID).getBoundaryCondition():
                i = sIDs.index(sID)
                S[i, j] = S[i, j] - s
        for product in reaction.getListOfProducts():
            sID = product.getSpecies()
            s = product.getStoichiometry()
            if not model.getSpecies(sID).getBoundaryCondition():
                i = sIDs.index(sID)
                S[i, j] = S[i, j] + s
        kinetic_law = reaction.getKineticLaw()
        rxn_lb = kinetic_law.getParameter('LOWER_BOUND').getValue()
        rxn_ub = kinetic_law.getParameter('UPPER_BOUND').getValue()
        rxn_c = kinetic_law.getParameter('OBJECTIVE_COEFFICIENT').getValue()
        rxn_rev = reaction.getReversible()
        if rxn_lb < -bound:
            rxn_lb = -bound
        if rxn_ub > bound:
            rxn_ub = bound
        if rxn_lb < 0:
            rxn_rev = True
        lb.append(rxn_lb)
        ub.append(rxn_ub)
        c.append(rxn_c)
        rev.append(rxn_rev)
    lb, ub, c, b = np.array(lb), np.array(ub), np.array(c), np.array(b)
    rev = np.array(rev)
    cobra = {'S': S, 'lb': lb, 'ub': ub, 'c': c, 'b': b, 'rev': rev}
    return cobra


def optimize_cobra_model(sbml, original_method=False):
    """Replicate Cobra command optimizeCbModel(model,[],'one')."""

    if original_method:
        bound = 1000
    else:
        bound = np.inf
    cobra = convert_sbml_to_cobra(sbml, bound)

    N, L, U = cobra['S'], list(cobra['lb']), list(cobra['ub'])
    f, b = list(cobra['c']), list(cobra['b'])
    v_sol, f_opt, _ = easy_lp(f, N, b, L, U, one=True)

    return v_sol, f_opt


def gimme(
        sbml, gene_exp,
        cutoff_threshold=0.25, req_fun=0.9, original_method=False):
    """Gimme method."""
    model = sbml.getModel()

    # set "required metabolic functionalities"
    f_opt = optimize_cobra_model(sbml, original_method)[1]
    c = [
        reaction.getKineticLaw()
        .getParameter('OBJECTIVE_COEFFICIENT').getValue()
        for reaction in model.getListOfReactions()
    ]
    biomass = model.getReaction(c.index(1))
    biomass.getKineticLaw().getParameter('LOWER_BOUND').setValue(
        req_fun * f_opt)

    cutoff_percent = 100. * cutoff_threshold
    if original_method:
        cutoff = prctile(gene_exp, cutoff_percent)
        bound = 1000
    else:
        cutoff = np.percentile(gene_exp, cutoff_percent)
        bound = np.inf

    cobra = convert_sbml_to_cobra(sbml, bound=bound)
    S, L, U = cobra['S'], list(cobra['lb']), list(cobra['ub'])
    f, b = list(cobra['c']), list(cobra['b'])
    f = [0.] * len(f)
    for i in range(len(gene_exp)):
        if gene_exp[i] < cutoff:
            c = cutoff - gene_exp[i]
            n1, n2 = S.shape
            col = sparse.lil_matrix((n1, 1))
            S = sparse.hstack([S, col, col])
            row = sparse.lil_matrix((1, n2 + 2))
            row[0, i] = 1.
            row[0, n2] = -1.
            row[0, n2 + 1] = 1.
            S = sparse.vstack([S, row])
            L.append(0.)
            L.append(0.)
            U.append(np.inf)
            U.append(np.inf)
            f.append(-c)
            f.append(-c)
            b.append(0.)
    solution = easy_lp(f, S, b, L, U, one=True)[0]
    v_sol = solution[:model.getNumReactions()]

    return v_sol


def prctile(x, p):
    """Implementation of MatLab percentile function."""
    x = np.array(x)
    x = x[~np.isnan(x)]
    x.sort()
    nr = len(x)
    q = 100 * (np.array(range(nr)) + 0.5) / nr
    v = np.interp(p, q, x)
    return v


def shlomi(sbml):
    """[Shlomi method is not implemented.]"""
    model = sbml.getModel()
    return np.zeros(model.getNumReactions())


def fba_fitted(sbml, data, original_method=False):
    """FBA solution that best fits data."""
    f_opt = optimize_cobra_model(sbml, original_method)[1]
    model = sbml.getModel()
    c = [
        reaction.getKineticLaw()
        .getParameter('OBJECTIVE_COEFFICIENT').getValue()
        for reaction in model.getListOfReactions()
    ]
    biomass = model.getReaction(c.index(1))
    biomass.getKineticLaw().getParameter('LOWER_BOUND').setValue(f_opt)
    if original_method:
        bound = 1000
    else:
        bound = np.inf
    cobra = convert_sbml_to_cobra(sbml, bound)
    N, L, U = cobra['S'].copy(), list(cobra['lb']), list(cobra['ub'])
    f, b = list(cobra['c']), list(cobra['b'])
    f = [0.] * len(f)
    for i in range(model.getNumReactions()):
        flux = data[i]
        if not np.isnan(flux):
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
            b.append(flux)
            f.append(-1.)
            f.append(-1.)
    v_sol = easy_lp(f, N, b, L, U, one=True)[0]
    v_sol = v_sol[:model.getNumReactions()]

    return v_sol


if __name__ == '__main__':
    results()
#     test_ComparisonDaaaaave()
    print('DONE!')
