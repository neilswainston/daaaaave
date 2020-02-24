'''Implementation of the Daaaaave algorithm.

15-Apr-14:

The Daaaaave algorithm is described in
Lee D, Smallbone K, Dunn WB, Murabito E, Winder CL, Kell DB, Mendes P,
Swainston N (2012) 'Improving metabolic flux predictions using absolute
gene expression data' BMC Syst Biol 6:73
http://dx.doi.org/10.1186/1752-0509-6-73

results() will reproduce Tables 1 and 2 of that paper, which compares
Daaaaave with alternatives FBA, Gimme and Shlomi, and with experimental
measurements.

Flagging original_method to True will use a warts-and-all implementation
to ensure identical results to the original Matlab algorithm.

29-May-15:

Implementation of all-improved SuperDaaaaave, translated from the Matlab
function 'call_SuperDaaaaave.m' available at
http://github.com/u003f/transcript2flux
'''
# pylint: disable=consider-using-enumerate
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order

import os

from sklearn.metrics import r2_score

from daaave.absolute import fba, gimme, orig_daaave
from daaave.data import load_flux_data, load_gene_data
from daaave.model import read_sbml
import numpy as np


PATH = os.path.join(os.path.dirname(__file__), '../data')


def run():
    '''Print tables 1 and 2 of Daaaaave et al.'''

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
        'sookie/model_CBS1513.xml',
        'sookie/genedata_30aa.txt',
        'experimental_fluxes_75.txt',
        'glucose transport',
        'D-glucose exchange',
        original_method=True
    )
#     print_results(
#         'yeast_5.21_MCISB.xml',
#         'genedata_85.txt', 'experimental_fluxes_85.txt',
#         'glucose transport', 'D-glucose exchange',
#         original_method=True
#         )
#     print '%g seconds elapsed\n' % (time.clock() - start)

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
#     print '%g seconds elapsed\n' % (time.clock() - start)


def print_results(
        model_file, genes_file, fluxes_file,
        gene_to_scale, flux_to_scale, original_method=False):
    '''Format output as per Daaaaave et al.'''

    (rxn_names, exp_flux, mod_SuperDaaaaave, mod_daaaaave,
     mod_fba, mod_fba_best, mod_gimme) = analysis(
         model_file, genes_file, fluxes_file,
         gene_to_scale, flux_to_scale, original_method)
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
        'rxn', 'exp', 'SDave', 'Dave', 'FBA', 'fFBA', 'Gimme'))
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


def analysis(
        model_file, genes_file, fluxes_file,
        gene_to_scale, flux_to_scale, original_method=False):
    '''Run all analyses on input files.'''

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

    # Original Daaave:
    v_gene_exp = orig_daaave.daaave(
        sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale,
        exp_rxn_names, exp_flux, flux_to_scale, original_method
    )

    # Gimme:
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


def format_results(
        sbml, exp_rxn_names, v_SuperDaaaaave,
        v_gene_exp, v_fba, v_gimme, v_fba_best):
    '''Format results.'''
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


if __name__ == '__main__':
    run()
