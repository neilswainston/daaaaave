'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order

from daaave.data import genes_to_rxns
from daaave.model import convert_sbml_to_cobra
from daaave.runner import rescale_model
from daaave.solver import easy_lp, optimize_cobra_model
import numpy as np
import scipy.sparse as sparse


def gimme(sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale,
          exp_rxn_names, exp_flux, flux_to_scale):
    '''Run Gimme.'''
    # gene data -> reaction data
    rxn_exp, rxn_exp_sd = genes_to_rxns(
        sbml, gene_names, gene_exp, gene_exp_sd)

    # rescale model so rxn_exp and flux = 1 for reaction gene_to_scale
    sbml, rxn_exp, rxn_exp_sd = rescale_model(
        sbml, rxn_exp, rxn_exp_sd, gene_to_scale)

    # gimme
    flux = _gimme(sbml, rxn_exp)

    # rescale
    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]

    return [value * flux_scale for value in flux]


def _gimme(sbml, gene_exps, cutoff_threshold=0.25, req_fun=0.9):
    '''Gimme method.'''
    model = sbml.getModel()

    # set 'required metabolic functionalities'
    f_opt = optimize_cobra_model(sbml)[1]

    c = [
        reaction.getKineticLaw()
        .getParameter('OBJECTIVE_COEFFICIENT').getValue()
        for reaction in model.getListOfReactions()
    ]

    biomass = model.getReaction(c.index(1))

    biomass.getKineticLaw().getParameter('LOWER_BOUND').setValue(
        req_fun * f_opt)

    cutoff_percent = 100. * cutoff_threshold
    cutoff = _prctile(gene_exps, cutoff_percent)

    cobra = convert_sbml_to_cobra(sbml)
    S, L, U = cobra['S'], cobra['lb'], cobra['ub']
    f, b = cobra['c'], cobra['b']
    f = [0.] * len(f)

    for i, gene_exp in enumerate(gene_exps):
        if gene_exp < cutoff:
            c = cutoff - gene_exp
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

    return solution[:model.getNumReactions()]


def _prctile(x, p):
    '''Implementation of MatLab percentile function.'''
    x = np.array(x)
    x = x[~np.isnan(x)]
    x.sort()
    nr = len(x)
    q = 100 * (np.array(range(nr)) + 0.5) / nr
    return np.interp(p, q, x)
