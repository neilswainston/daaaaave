'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order
from daaave.model_utils import convert_sbml_to_cobra
from daaave.solver import easy_lp, optimize_cobra_model
import numpy as np
import scipy.sparse as sparse


def original_fba(sbml, exp_rxn_names, exp_flux, flux_to_scale):
    '''Original FBA.'''
    flux, _, _ = optimize_cobra_model(sbml)

    # rescale
    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
    return [value * flux_scale for value in flux]


def fitted_fba(sbml, gene_to_scale, exp_rxn_names, exp_flux, flux_to_scale):
    '''Fitted FBA.'''
    data = _create_data_array(
        sbml, exp_flux[:], exp_rxn_names[:], gene_to_scale)

    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
    flux = _fitted_fba(sbml, data / flux_scale)

    # rescale
    return [value * flux_scale for value in flux]


def _create_data_array(sbml, exp_flux, exp_rxn_names, gene_to_scale):
    '''Create data array.'''
    exp_rxn_names.append(gene_to_scale)
    exp_flux.append(1.)
    model = sbml.getModel()
    mod_rxn_names = [reaction.getName()
                     for reaction in model.getListOfReactions()]

    data = np.empty(model.getNumReactions())
    data[:] = np.nan

    for i, exp_rxn_name in enumerate(exp_rxn_names):
        j = mod_rxn_names.index(exp_rxn_name)
        data[j] = exp_flux[i]

    return data


def _fitted_fba(sbml, data):
    '''FBA solution that best fits data.'''
    f_opt = optimize_cobra_model(sbml)[1]
    model = sbml.getModel()
    c = [
        reaction.getKineticLaw()
        .getParameter('OBJECTIVE_COEFFICIENT').getValue()
        for reaction in model.getListOfReactions()
    ]
    biomass = model.getReaction(c.index(1))
    biomass.getKineticLaw().getParameter('LOWER_BOUND').setValue(f_opt)
    cobra = convert_sbml_to_cobra(sbml)
    N, L, U = cobra['S'].copy(), cobra['lb'], cobra['ub']
    f, b = cobra['c'], cobra['b']
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
    return v_sol[:model.getNumReactions()]
