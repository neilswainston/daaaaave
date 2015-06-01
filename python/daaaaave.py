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

Implementation of all-improved SuperDaaaaave, translated from the Matlab function
"call_SuperDaaaaave.m" available at http://github.com/u003f/transcript2flux
"""

# pylint --generate-rcfile
# http://legacy.python.org/dev/peps/pep-0008/

from sympy.logic import boolalg
import csv
import numpy as np
import os
import re
import scipy.sparse as sparse
import time

# http://www.gurobi.com/documentation/6.0/quickstart_mac/py_python_interface.html
import gurobipy
# http://frank-fbergmann.blogspot.co.uk/2014/05/libsbml-python-bindings-5101.html
import libsbml

NAN = np.nan
INF = np.inf
PATH = os.path.dirname(__file__)


def call_SuperDaaaaave(sbml, gene_names, gene_exp, gene_exp_sd, MaxGrowth=False):

    """
    Implementation of all-improved SuperDaaaaave, translated from the Matlab function
    call_SuperDaaaaave.m available at http://github.com/u003f/transcript2flux
    29 May 2015
    """

    model = sbml.getModel()
    EPS = 2.**(-52.)

#     % fill in zero entries
#     unmeasured = setdiff(model.genes, gene_names);
#     gene_names = [gene_names; unmeasured];
#     gene_exp = [gene_exp; zeros(length(unmeasured), 1)];
#     if isempty(gene_exp_sd)
#         gene_exp_sd = zeros(size(gene_exp));
#         disp('warning: no gene expression std given')
#     else
#         gene_exp_sd = [gene_exp_sd; zeros(length(unmeasured), 1)];
#     end
#     gene_min = min(gene_exp_sd(gene_exp_sd>0));
#     if isempty(gene_min)
#         gene_min = min(gene_exp(gene_exp>0));
#     end
#     gene_exp_sd(gene_exp_sd == 0) = gene_min/2;

    # replace characters in gene identifiers
    gene_names = [gene.replace('-', '_') for gene in gene_names]
    model_grRules = get_list_of_genes(sbml)
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
    L[L<-500] = -INF
    U[U>500] = INF

    solution_full, solution_obj, statMaxGrowth = easy_lp(fMaxGrowth, S, b, L, U, one=True)
    solnMaxGrowth = np.zeros(nR)
    objMaxGrowth = 0.
    if statMaxGrowth == 1.:
        # use max growth if SuperDaaaaave does not converge
        solnMaxGrowth = solution_full
        objMaxGrowth = np.floor(solution_obj/EPS)*EPS # round down a touch

    # create positive FBA problem
    S = sparse.vstack([
        sparse.hstack([S, sparse.lil_matrix((nS, 2*nR))]),
        sparse.hstack([sparse.eye(nR), -sparse.eye(nR), sparse.eye(nR)])
        ])
    L = np.append(L,np.zeros(2*nR))
    U = np.append(U,INF*np.ones(2*nR))
    b = np.append(b, np.zeros(nR))
    f = np.append(f, np.zeros(2*nR))
    csense = csense + 'E'*nR
    vartype = vartype + 'C'*2*nR

    # only allow positive or negative flux
    M = 1e3 * max(abs( np.hstack([U[np.isfinite(U)], L[np.isfinite(L)], gene_exp]) ))
    S = sparse.hstack([S, sparse.lil_matrix((nS+nR, 2*nR))])
    L = np.append(L,-INF*np.ones(2*nR))
    U = np.append(U,INF*np.ones(2*nR))
    f = np.append(f, np.zeros(2*nR))
    vartype = vartype + 'B'*2*nR

    # p <= M * kP -> -p + M*kP >= 0
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)), -sparse.eye(nR), sparse.lil_matrix((nR, nR)), M*sparse.eye(nR), sparse.lil_matrix((nR, nR))])
        ])
    b = np.append(b, np.zeros(nR))
    csense = csense + 'G'*nR

    # n <= M * kN -> -n + M*kN >= 0
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)), sparse.lil_matrix((nR, nR)), -sparse.eye(nR), sparse.lil_matrix((nR, nR)), M*sparse.eye(nR)])
        ])
    b = np.append(b, np.zeros(nR))
    csense = csense + 'G'*nR

    # kP + kN = 1
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)), sparse.lil_matrix((nR, nR)), sparse.lil_matrix((nR, nR)), sparse.eye(nR), sparse.eye(nR)])
        ])
    b = np.append(b, np.ones(nR))
    csense = csense + 'E'*nR

    # abs(v) variables
    S = sparse.hstack([S, sparse.lil_matrix((nS+4*nR, nR))])
    L = np.append(L,-INF*np.ones(nR))
    U = np.append(U,INF*np.ones(nR))
    f = np.append(f, np.zeros(nR))
    vartype = vartype + 'C'*nR
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nR, nR)), -sparse.eye(nR), -sparse.eye(nR), sparse.lil_matrix((nR, nR)), sparse.lil_matrix((nR, nR)), sparse.eye(nR)])
        ])
    b = np.append(b, np.zeros(nR))
    csense = csense + 'E'*nR
    model_rxns = [reaction.getId() for reaction in model.getListOfReactions()]
    abs_flux_index = dict(zip(model_rxns, range(5*nR,6*nR)))

    # add scaling parameter a
    scale_index = 6*nR
    nS_all, nR_all = S.shape

    S = sparse.hstack([S, sparse.lil_matrix((nS_all, 1))])
    L = np.append(L, 0)
    U = np.append(U, INF)
    f = np.append(f, 0)
    vartype = vartype + 'C'

    # v >= a L -> v - a L >= 0
    n = 0
    for i, x in enumerate(L[:nR]):
        if x not in [-INF, 0, INF]:
            nS_all, nR_all = S.shape
            row = np.zeros(nR_all)
            row[i] = 1
            row[scale_index] = -x
            n += 1
            L[i] = -INF
            S = sparse.vstack([S, row])
    b = np.append(b, np.zeros(n))
    csense = csense + 'G'*n

    # v <= a U -> - v + a U >= 0
    n = 0
    for i, x in enumerate(U[:nR]):
        if x not in [-INF, 0, INF]:
            nS_all, nR_all = S.shape
            row = np.zeros(nR_all)
            row[i] = -1
            row[scale_index] = x
            n += 1
            U[i] = INF
            S = sparse.vstack([S, row])
    b = np.append(b, np.zeros(n))
    csense = csense + 'G'*n

    # add slack variables for genes
    nS_all, nR_all = S.shape
    nG = len(gene_names)
    S = sparse.hstack([S, sparse.lil_matrix((nS_all, nG))])
    L = np.append(L, np.zeros(nG))
    U = np.append(U, INF*np.ones(nG))
    f = np.append(f, np.zeros(nG))
    vartype = vartype + 'C'*nG

    # add genes
    S = sparse.vstack([
        S,
        sparse.hstack([sparse.lil_matrix((nG, nR_all)), sparse.eye(nG)])
        ])
    b = np.append(b, gene_exp)
    csense = csense + 'E'*nG
    gene_index = dict(zip(gene_names, range(nS_all,nS_all+nG)))

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
            S.resize((nS_all+1, nR_all+1))
            S[rxn_row, rxn_col] = -1
            L = np.append(L, -INF)
            U = np.append(U, INF)
            f = np.append(f, 1)
            vartype = vartype + 'C'
            b = np.append(b, 0)
            csense = csense + 'E'

            rxn_index[rxn_id] = rxn_col

            # convert to DNF (or of ands)
            association = to_dnf(association)
            list_of_ors = re.split(' or ', association)

            std_out = 0

            for association_or in list_of_ors:

                # add complex
                nS_all, nR_all = S.shape
                complex_col = nR_all
                S.resize((nS_all, nR_all+1))
                S[rxn_row, complex_col] = 1
                L = np.append(L, 0)
                U = np.append(U, INF)
                f = np.append(f, 0)
                vartype = vartype + 'C'

                if association_or[0] == '(':
                    association_or = association_or[1:-1]

                list_of_ands = re.split(' and ', association_or)

                std_in = INF

                for gene in list_of_ands:

                    nS_all, nR_all = S.shape
                    gene_col = nR_all
                    ineq_row = nS_all
                    # complex > gene -> gene - complex > 0
                    S.resize((nS_all+1, nR_all+1))
                    S[ineq_row, gene_col] = 1
                    S[ineq_row, complex_col] = -1
                    L = np.append(L, 0)
                    U = np.append(U, INF)
                    f = np.append(f, 0)
                    vartype = vartype + 'C'
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
        fMaxGrowthBig = np.zeros((1,len(f)))
        fMaxGrowthBig[0,:len(fMaxGrowth)] = fMaxGrowth
        fMaxGrowthBig[0,scale_index] = -objMaxGrowth
        S = sparse.vstack([S, fMaxGrowthBig])
        b = np.append(b, 0)
        csense = csense + 'E'

    v, obj, conv = easy_milp(f, S, b, L, U, csense, vartype)

    # set objective as constraint
    obj = np.floor(obj/EPS)*EPS # round down a touch
    S = sparse.vstack([S, f])
    b = np.append(b, obj)
    csense = csense + 'E'

    # minimise distance from data to flux
    f = np.zeros(len(f))
    S = S.todok()
    for ind in range(nR):
        rxn_id = model_rxns[ind]
        if rxn_id in rxn_index.keys():
            nS_all, nR_all = S.shape
            # R - D = P - N -> R - D - P + N = 0
            S.resize((nS_all+1, nR_all+2))
            rxn_ind = nS_all
            S[rxn_ind, abs_flux_index[rxn_id]] = 1
            S[rxn_ind, rxn_index[rxn_id]] = -1
            S[rxn_ind, nR_all] = -1
            S[rxn_ind, nR_all+1] = 1
            L = np.append(L,[0,0])
            U = np.append(U,[INF,INF])
#             std = rxn_std[rxn_id]
#             f = np.append(f, [-1/std, -1/std])
            f = np.append(f, [-1, -1])
            vartype = vartype + 'C'*2
            b = np.append(b,0)
            csense = csense + 'E'

    soln, solution_obj, conv = easy_milp(f, S, b, L, U, csense, vartype)
#     print 'MILP solution:\t%g\n' %solution_obj

    # rescale
    if soln:
        fluxes = [soln[i]/soln[scale_index] for i in range(nR)]
    else:
        fluxes = solnMaxGrowth

    return fluxes


def to_dnf(association):

    # A and B and (C or D) or E
    association = association.replace(' AND ',' & ').replace(' OR ',' | ').replace(' and ',' & ').replace(' or ',' | ')
    # -> A & B & (C | D) | E
    association = str(boolalg.to_dnf(association))
    # -> Or(And(A, B, C), And(A, B, D), E)
    for and_old in re.findall(r'And\([^)]+\)',association):
        and_new = and_old
        and_new = and_new.replace('And(','(')
        and_new = and_new.replace(', ',' and ')
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
    sbml = read_sbml(os.path.join(PATH, model_file))

    # gene data
    gene_names, gene_exp, gene_exp_sd = parse_gene_data(
        os.path.join(PATH, genes_file)
        )
    # flux data
    exp_flux, exp_rxn_names = load_flux_data(os.path.join(PATH, fluxes_file))

    # set SBML input to experimental value
    sbml = read_sbml(os.path.join(PATH, model_file))
    rescale_SBML(sbml, exp_rxn_names, exp_flux, flux_to_scale)

    flux = call_SuperDaaaaave(sbml, gene_names, gene_exp, gene_exp_sd, MaxGrowth=False)

    return flux


def results():
    """Print tables 1 and 2 of Daaaaave et al."""

    print_results(
        'example.xml',
        'genedata_example_1.txt', 'experimental_fluxes_example_1.txt',
        'rA', 'rA'
        )
    print_results(
        'example.xml',
        'genedata_example_2.txt',
        'experimental_fluxes_example_2.txt',
        'rA', 'rA'
        )
#
    start = time.clock()
    print_results(
        'yeast_5.21_MCISB.xml',
        'genedata_75.txt', 'experimental_fluxes_75.txt',
        'glucose transport', 'D-glucose exchange',
        original_method=True
        )
    print_results(
        'yeast_5.21_MCISB.xml',
        'genedata_85.txt', 'experimental_fluxes_85.txt',
        'glucose transport', 'D-glucose exchange',
        original_method=True
        )
    print "%g seconds elapsed\n" % (time.clock() - start)

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
    print '%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
        "rxn", "exp", "S'Dave", "Dave",
        "FBA", "fFBA", "Gimme")
    for i in xrange(len(rxn_names)):
        print '%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g' % (
            rxn_names[i], exp_flux[i], mod_SuperDaaaaave[i], mod_daaaaave[i],
            mod_fba[i], mod_fba_best[i], mod_gimme[i])
    print '%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n' % (
        'R2', 1, r_squared(mod_SuperDaaaaave, exp_flux), r_squared(mod_daaaaave, exp_flux),
        r_squared(mod_fba, exp_flux), r_squared(mod_fba_best, exp_flux),
        r_squared(mod_gimme, exp_flux))


def r_squared(f, y):
    """The coefficient of determination of data y with model f."""
    # See http://en.wikipedia.org/wiki/Coefficient_of_determination
    ss_res = sum((y-f)**2)
    ss_tot = sum((y-np.mean(y))**2)
    return 1 - ss_res/ss_tot


def OriginalDaaaaave(sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale, exp_rxn_names, exp_flux, flux_to_scale, original_method=False):

    # gene data -> reaction data
    rxn_exp, rxn_exp_sd = genes_to_rxns(
        sbml, gene_names, gene_exp, gene_exp_sd, original_method
        )
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


def GimmeGimmeGimme(sbml, gene_names, gene_exp, gene_exp_sd, gene_to_scale, exp_rxn_names, exp_flux, flux_to_scale, original_method=False):

    cutoff = 0.25  # set threshold at lower quartile
    req_fun = 0.9  # force 90% growth

    # gene data -> reaction data
    rxn_exp, rxn_exp_sd = genes_to_rxns(
        sbml, gene_names, gene_exp, gene_exp_sd, original_method
        )
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


def OriginalFBA(sbml, exp_rxn_names, exp_flux, flux_to_scale, original_method=False):

	flux, f_opt = optimize_cobra_model(sbml, original_method)

	# rescale
	flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
	flux = [value * flux_scale for value in flux]

	return flux


def FittedFBA(sbml, gene_to_scale, exp_rxn_names, exp_flux, flux_to_scale, original_method=False):

    data = create_data_array(sbml, exp_flux[:], exp_rxn_names[:], gene_to_scale)

    flux_scale = exp_flux[exp_rxn_names.index(flux_to_scale)]
    flux = fba_fitted(sbml, data/flux_scale, original_method)

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
            elif (UB.getValue() < 0):
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
    v_SuperDaaaaave = SuperDaaaaave(model_file, genes_file, fluxes_file, flux_to_scale)

    # MODEL
    sbml = read_sbml(os.path.join(PATH, model_file))

    # gene data
    gene_names, gene_exp, gene_exp_sd = parse_gene_data(
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
    v_fba = OriginalFBA(sbml, exp_rxn_names, exp_flux, flux_to_scale, original_method)

    # find best fit from standard FBA solution
    v_fba_best = FittedFBA(sbml, gene_to_scale, exp_rxn_names, exp_flux, flux_to_scale, original_method)

    # compare
    mod_SuperDaaaaave, mod_daaaaave, mod_fba, mod_gimme, mod_fba_best = \
        format_results(
            sbml, exp_rxn_names, v_SuperDaaaaave,
            v_gene_exp, v_fba, v_gimme, v_fba_best
            )

    return exp_rxn_names, exp_flux, mod_SuperDaaaaave, mod_daaaaave, \
        mod_fba, mod_fba_best, mod_gimme


def read_sbml(filename):
    """Read an SBML file from specified path."""
    reader = libsbml.SBMLReader()
    sbml = reader.readSBMLFromFile(filename)
    return sbml


def rescale_model(sbml, rxn_exp, rxn_exp_sd, gene_to_scale):

    model = sbml.getModel()
    rxn_names = [
        reaction.getName()
        for reaction in model.getListOfReactions()
        ]
    uptake = rxn_names.index(gene_to_scale)
    rxn_exp_sd = rxn_exp_sd/rxn_exp[uptake]
    rxn_exp = rxn_exp/rxn_exp[uptake]
    reaction = model.getReaction(uptake)
    kinetic_law = reaction.getKineticLaw()
    kinetic_law.getParameter('LOWER_BOUND').setValue(1)
    kinetic_law.getParameter('UPPER_BOUND').setValue(1)

    return sbml, rxn_exp, rxn_exp_sd


def load_flux_data(fluxes_file):

    rxn_names, exp_flux = [], []
    with open(fluxes_file, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            rxn_names.append(row[0])
            exp_flux.append(float(row[1]))
    return exp_flux, rxn_names


def create_data_array(sbml, exp_flux, exp_rxn_names, gene_to_scale):

    exp_rxn_names.append(gene_to_scale)
    exp_flux.append(1.)
    model = sbml.getModel()
    mod_rxn_names = [
        reaction.getName()
        for reaction in model.getListOfReactions()
        ]
    data = np.empty(model.getNumReactions())
    data[:] = NAN
    for i in xrange(len(exp_rxn_names)):
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

    for i in xrange(len(exp_rxn_names)):
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


def parse_gene_data(genes_file):
    """Translate gene expression data file to arrays."""
    gene_names, gene_exp, gene_exp_sd = [], [], []
    with open(genes_file, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            gene_names.append(row["gene"])
            gene_exp.append(float(row["mean"]))
            gene_exp_sd.append(float(row["std"]))
    gene_exp, gene_exp_sd = np.array(gene_exp), np.array(gene_exp_sd)
    return gene_names, gene_exp, gene_exp_sd


def get_list_of_genes(sbml):

    gene_list = []
    model = sbml.getModel()
    for reaction in model.getListOfReactions():
        notes = reaction.getNotesString()
        match = re.search(
            r'<p>GENE_ASSOCIATION:' + '([^<]*)' + r'</p>', notes
            )
        gene_assn = match.group(1)
        gene_list.append(gene_assn)

    return gene_list


def genes_to_rxns(
        sbml, gene_names, gene_exp, gene_exp_sd,
        original_method=False):
    """Match gene-level data to reaction-level data."""

    model = sbml.getModel()
    rxn_exp, rxn_exp_sd = [], []
    for i in xrange(len(gene_names)):
        gene_names[i] = gene_names[i].replace('-', '_')

    list_of_genes = get_list_of_genes(sbml)

    for i in xrange(model.getNumReactions()):
        reaction = model.getReaction(i)

        gene_assn = list_of_genes[i]

        gene_assn = gene_assn.replace('-', '_')
        gene_assn = gene_assn.replace(' AND ', ' and ')
        gene_assn = gene_assn.replace(' OR ', ' or ')
        gene_assn = gene_assn.replace(' )', ')')
        gene_assn = gene_assn.replace('( ', '(')
        gene_list = re.findall(r'\b([\w]*)\b', gene_assn)
        gene_list = set_diff(gene_list, ['and', 'or', ''])
        for gene in gene_list:
            j = gene_names.index(gene)
            ng, ng_sd = gene_exp[j], gene_exp_sd[j]
            if original_method:
                # use Matlab number to string converter
                str_ng, str_ng_sd = num2str(ng), num2str(ng_sd)
            else:
                # use high-precision python number to string converter
                str_ng, str_ng_sd = repr(ng), repr(ng_sd)
            gene_assn = re.sub(
                r'\b' + gene + r'\b', str_ng + '~' + str_ng_sd, gene_assn
                )
        nr, nr_sd = map_gene_data(gene_assn, original_method)
        rxn_exp.append(nr)
        rxn_exp_sd.append(nr_sd)

    rxn_exp, rxn_exp_sd = np.array(rxn_exp), np.array(rxn_exp_sd)
    # sds 0 -> small
    rxn_exp_sd[rxn_exp_sd == 0] = min(rxn_exp_sd[rxn_exp_sd != 0])/2
    return rxn_exp, rxn_exp_sd


def num2str(x):
    """Implementation of Matlab num2str number to string converter."""
    max_field_width = 12
    float_width_offset = 4
    float_field_extra = 7
    xmax = float(abs(x))
    if xmax == 0:
        d = 1
    else:
        d = min(max_field_width, max(1, np.floor(np.log10(xmax))+1)) + \
            float_width_offset
    f = '%%%.0f.%.0fg' % (d+float_field_extra, d)
    s = f % x
    s = s.strip()
    return s


def set_diff(a, b):
    """Return the set difference of the two arrays."""
    return list(set(a).difference(set(b)))


def map_gene_data(gene_assn, original_method=False):
    """Map string '(x1~x1SD) and (x2~x2SD) or (x3~x3SD)' to string y~ySD."""
    nr, nr_sd = NAN, NAN
    a_pm_b = r'[0-9\.]+~[0-9\.]+'
    if gene_assn:
        while np.isnan(nr):
            try:
                match_expr = r'\A([0-9\.]+)~([0-9\.]+)\Z'
                match = re.search(match_expr, gene_assn)
                str_nr, str_nr_sd = match.group(1), match.group(2)
                nr, nr_sd = float(str_nr), float(str_nr_sd)
            except:
                # replace brackets
                match_expr = r'\((' + a_pm_b + r')\)'
                for match in re.findall(match_expr, gene_assn):
                    gene_assn = re.sub(r'\(' + match + r'\)', match, gene_assn)
                # replace ANDs
                match_expr = '(' + a_pm_b + ') and (' + a_pm_b + ')'
                for ind, match in enumerate(re.findall(match_expr, gene_assn)):
                    if ind == 0:
                        lhs = match[0]
                        rhs = match[1]
                        replace_expr = a_and_b(lhs, rhs, original_method)
                        gene_assn = re.sub(
                            r'\b' + lhs + ' and ' + rhs + r'\b',
                            replace_expr, gene_assn
                            )
                # replace ORs
                match_expr = r'(' + a_pm_b + r') or (' + a_pm_b + r')'
                match = re.search(match_expr, gene_assn)
                for ind, match in enumerate(re.findall(match_expr, gene_assn)):
                    if ind == 0:
                        lhs = match[0]
                        rhs = match[1]
                        replace_expr = a_or_b(lhs, rhs, original_method)
                        gene_assn = re.sub(
                            r'\b' + lhs + ' or ' + rhs + r'\b',
                            replace_expr, gene_assn
                            )
    return nr, nr_sd


def a_and_b(str1, str2, original_method=False):

    a_pm_b = r'\A([0-9\.]+)~([0-9\.]+)'
    match_expr = a_pm_b
    match1 = re.search(match_expr, str1)
    ng1 = float(match1.group(1))
    ng1_sd = float(match1.group(2))
    match2 = re.search(match_expr, str2)
    ng2 = float(match2.group(1))
    ng2_sd = float(match2.group(2))
    ng12, ng12_sd = [ng1, ng2], [ng1_sd, ng2_sd]
    j = np.argmin(ng12)
    ng, ng_sd = ng12[j], ng12_sd[j]
    if original_method:
        str_ng, str_ng_sd = num2str(ng), num2str(ng_sd)
    else:
        str_ng, str_ng_sd = repr(ng), repr(ng_sd)
    return str_ng + '~' + str_ng_sd


def a_or_b(str1, str2, original_method=False):

    a_pm_b = r'\A([0-9\.]+)~([0-9\.]+)'
    match_expr = a_pm_b
    match1 = re.search(match_expr, str1)
    ng1 = float(match1.group(1))
    ng1_sd = float(match1.group(2))
    match2 = re.search(match_expr, str2)
    ng2 = float(match2.group(1))
    ng2_sd = float(match2.group(2))
    ng = ng1 + ng2
    ng_sd = np.sqrt(ng1_sd**2. + ng2_sd**2.)
    if original_method:
        str_ng, str_ng_sd = num2str(ng), num2str(ng_sd)
    else:
        str_ng, str_ng_sd = repr(ng), repr(ng_sd)
    return str_ng + '~' + str_ng_sd


def data_to_flux(sbml, rxn_exp, rxn_exp_sd, original_method=False):
    """Daaaaave: predict flux by maximising correlation with data."""

    model = sbml.getModel()
    nr_old = 0
    if original_method:
        bound = 1000
    else:
        bound = INF
    cobra = convert_sbml_to_cobra(sbml, bound)
    v_sol = np.zeros(model.getNumReactions())

    while list(cobra['rev']).count(False) > nr_old:
        nr_old = list(cobra['rev']).count(False)

        # 1. fit to data
        N, L, U = cobra['S'].copy(), list(cobra['lb']), list(cobra['ub'])
        f, b = list(cobra['c']), list(cobra['b'])
        f = [0.] * len(f)
        for i in xrange(model.getNumReactions()):
            data, sd = rxn_exp[i], rxn_exp_sd[i]
            if (not cobra['rev'][i]) and (not np.isnan(data)) and (sd > 0):
                s1, s2 = N.shape
                col = sparse.lil_matrix((s1, 1))
                N = sparse.hstack([N, col, col])
                row = sparse.lil_matrix((1, s2+2))
                row[0, i] = 1.
                row[0, s2] = -1.
                row[0, s2+1] = 1.
                N = sparse.vstack([N, row])
                L.append(0.)
                L.append(0.)
                U.append(INF)
                U.append(INF)
                b.append(data)
                f.append(-1./sd)
                f.append(-1./sd)
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
            for j in xrange(cols):
                LB = L[j]
                if LB == -INF:
                    LB = -gurobipy.GRB.INFINITY
                UB = U[j]
                if UB == INF:
                    UB = gurobipy.GRB.INFINITY
                lp.addVar(lb=LB, ub=UB)
            lp.update()
            lpvars = lp.getVars()
            S = N.tocsr()
            for i in xrange(rows):
                start = S.indptr[i]
                end = S.indptr[i+1]
                variables = [lpvars[j] for j in S.indices[start:end]]
                coeff = S.data[start:end]
                expr = gurobipy.LinExpr(coeff, variables)
                lp.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=b[i])
            lp.update()
            lp.ModelSense = -1

            for i in xrange(model.getNumReactions()):
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
                            f_opr = NAN
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
                                f_opr = NAN
                        cond1 = (original_method) and (abs(f_opt) <= 0)
                        cond2 = (not original_method) and (f_opt <= 0)
                        if conv and (cond1 or cond2):  # irreversibly backward
                            cobra['ub'][i] = min(cobra['ub'][i], 0.)
                            cobra['rev'][i] = False
                            rxn_exp[i] = -rxn_exp[i]

    return v_sol


def easy_milp(f, a, b, vlb, vub, csense, vartype):
    '''Optimize MILP using friends of Gurobi.'''

    # catch np arrays
    f, b, vlb, vub = list(f), list(b), list(vlb), list(vub)

    # create gurobi model
    milp = gurobipy.Model()
    milp.Params.OutputFlag = 0
    milp.Params.FeasibilityTol = 1e-9  # as per Cobra
    milp.Params.OptimalityTol = 1e-9  # as per Cobra

    milp.Params.timeLimit = 5*60  # max 5 mins / solve
#     milp.Params.OutputFlag = 1  # display all

    rows, cols = a.shape
    # add variables to model
    for j in xrange(cols):
        LB = vlb[j]
        if LB == -INF:
            LB = -gurobipy.GRB.INFINITY
        UB = vub[j]
        if UB == INF:
            UB = gurobipy.GRB.INFINITY
        milp.addVar(lb=LB, ub=UB, obj=f[j], vtype=vartype[j])
    milp.update()
    lpvars = milp.getVars()
    # iterate over the rows of S adding each row into the model
    S = a.tocsr()
    for i in xrange(rows):
        start = S.indptr[i]
        end = S.indptr[i+1]
        variables = [lpvars[j] for j in S.indices[start:end]]
        coeff = S.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        if csense[i] == 'E':
            csensei = gurobipy.GRB.EQUAL
        elif csense[i] == 'G':
            csensei = gurobipy.GRB.GREATER_EQUAL
        elif csense[i] == 'L':
            csensei = gurobipy.GRB.LESS_EQUAL
        milp.addConstr(lhs=expr, sense=csensei, rhs=b[i])
    milp.update()
    milp.ModelSense = -1
    milp.optimize()

    v = np.empty(len(f))
    v[:] = NAN
    f_opt = NAN
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
    for j in xrange(cols):
        LB = vlb[j]
        if LB == -INF:
            LB = -gurobipy.GRB.INFINITY
        UB = vub[j]
        if UB == INF:
            UB = gurobipy.GRB.INFINITY
        lp.addVar(lb=LB, ub=UB, obj=f[j])
    lp.update()
    lpvars = lp.getVars()
    # iterate over the rows of S adding each row into the model
    S = a.tocsr()
    for i in xrange(rows):
        start = S.indptr[i]
        end = S.indptr[i+1]
        variables = [lpvars[j] for j in S.indices[start:end]]
        coeff = S.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        lp.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=b[i])
    lp.update()
    lp.ModelSense = -1
    lp.optimize()

    v = np.empty(len(f))
    v[:] = NAN
    f_opt = NAN
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
        for i in xrange(nR):
            col = sparse.lil_matrix((nS + i, 1))
            a = sparse.hstack([a, col, col])
            row = sparse.lil_matrix((1, nR+2*i+2))
            row[0, i] = 1.
            row[0, nR+2*i] = 1.
            row[0, nR+2*i+1] = -1.
            a = sparse.vstack([a, row])
            vlb.append(0.)
            vlb.append(0.)
            vub.append(INF)
            vub.append(INF)
            f.append(-1.)
            f.append(-1.)
            b.append(0.)
        v_sol = easy_lp(f, a, b, vlb, vub, one=False)[0]
        v = v_sol[:nR]

    return v, f_opt, conv


def convert_sbml_to_cobra(sbml, bound=INF):
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
        bound = INF
    cobra = convert_sbml_to_cobra(sbml, bound)

    N, L, U = cobra['S'], list(cobra['lb']), list(cobra['ub'])
    f, b = list(cobra['c']), list(cobra['b'])
    v_sol, f_opt, conv = easy_lp(f, N, b, L, U, one=True)

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
    biomass.getKineticLaw().getParameter('LOWER_BOUND').setValue(req_fun*f_opt)

    cutoff_percent = 100.*cutoff_threshold
    if original_method:
        cutoff = prctile(gene_exp, cutoff_percent)
        bound = 1000
    else:
        cutoff = np.percentile(gene_exp, cutoff_percent)
        bound = INF

    cobra = convert_sbml_to_cobra(sbml, bound=bound)
    S, L, U = cobra['S'], list(cobra['lb']), list(cobra['ub'])
    f, b = list(cobra['c']), list(cobra['b'])
    f = [0.] * len(f)
    for i in xrange(len(gene_exp)):
        if gene_exp[i] < cutoff:
            c = cutoff - gene_exp[i]
            n1, n2 = S.shape
            col = sparse.lil_matrix((n1, 1))
            S = sparse.hstack([S, col, col])
            row = sparse.lil_matrix((1, n2+2))
            row[0, i] = 1.
            row[0, n2] = -1.
            row[0, n2+1] = 1.
            S = sparse.vstack([S, row])
            L.append(0.)
            L.append(0.)
            U.append(INF)
            U.append(INF)
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
    q = 100*(np.array(xrange(nr))+0.5)/nr
    v = np.interp(p, q, x)
    return v


def shlomi(sbml, rxn_exp, original_method=False):
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
        bound = INF
    cobra = convert_sbml_to_cobra(sbml, bound)
    N, L, U = cobra['S'].copy(), list(cobra['lb']), list(cobra['ub'])
    f, b = list(cobra['c']), list(cobra['b'])
    f = [0.] * len(f)
    for i in xrange(model.getNumReactions()):
        flux = data[i]
        if not np.isnan(flux):
            s1, s2 = N.shape
            col = sparse.lil_matrix((s1, 1))
            N = sparse.hstack([N, col, col])
            row = sparse.lil_matrix((1, s2+2))
            row[0, i] = 1.
            row[0, s2] = -1.
            row[0, s2+1] = 1.
            N = sparse.vstack([N, row])
            L.append(0.)
            L.append(0.)
            U.append(INF)
            U.append(INF)
            b.append(flux)
            f.append(-1.)
            f.append(-1.)
    v_sol = easy_lp(f, N, b, L, U, one=True)[0]
    v_sol = v_sol[:model.getNumReactions()]

    return v_sol


if __name__ == '__main__':
    results()
