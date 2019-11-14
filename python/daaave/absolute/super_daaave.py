'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=consider-using-enumerate
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import os
import re
from sympy.logic import boolalg
from daaave.data import load_flux_data, load_gene_data
from daaave.model import convert_sbml_to_cobra, get_gene_associations, \
    read_sbml
from daaave.solver import easy_lp, easy_milp, easy_milp_sos
import numpy as np
import scipy.sparse as sparse


PATH = os.path.join(os.path.dirname(__file__), '../../data')
LP_TOL = 1e-6


def SuperDaaaaave(model_file, genes_file, fluxes_file, flux_to_scale):
    '''Super Daaave.'''
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
    _rescale_SBML(sbml, exp_rxn_names, exp_flux, flux_to_scale)

    flux, _ = call_SuperDaaaaave(
        sbml, gene_names, gene_exp, gene_exp_sd, MaxGrowth=False)

    return flux


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
            association = _to_dnf(association)
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

    solution_full, solution_obj, statMaxGrowth = easy_lp(
        fMaxGrowth, S, b, L, U, one=True)
    solnMaxGrowth = np.zeros(nR)
    objMaxGrowth = 0.
    if statMaxGrowth == 1.:
        # use max growth if SuperDaaaaave does not converge
        solnMaxGrowth = solution_full

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
            association = _to_dnf(association)
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


def _to_dnf(association):

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


def _rescale_SBML(sbml, exp_rxn_names, exp_flux, flux_to_scale):

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
