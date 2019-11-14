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

import gurobipy
from sklearn.metrics import r2_score

from daaave.absolute import fba, gimme
from daaave.data import genes_to_rxns, load_flux_data, load_gene_data
from daaave.model import convert_sbml_to_cobra, read_sbml
from daaave.solver import easy_lp
import numpy as np
import scipy.sparse as sparse


PATH = os.path.join(os.path.dirname(__file__), '../data')


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
    v_gimme = gimme.gimme(
        sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale,
        exp_rxn_names, exp_flux, flux_to_scale)

    # OriginalFBA
    v_fba = fba.original_fba(sbml, exp_rxn_names, exp_flux, flux_to_scale)

    # find best fit from standard FBA solution
    v_fba_best = fba.fitted_fba(sbml, gene_to_scale, exp_rxn_names,
                                exp_flux, flux_to_scale)

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


if __name__ == '__main__':
    results()
