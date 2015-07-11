import csv
import numpy as np
import os
import sys
import xlrd

if __name__ == '__main__':
    if __package__ is None:
        sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
        from python import daaaaave as om
    else:
        from ..python import daaaaave as om

NAN = np.nan
INF = np.inf
PATH = os.path.dirname(__file__)


RAMBO_NAME = 'rambo.xml'
# RAMBO_NAME = 'rambo_150602.xml'


import shutil
shutil.copy(
    '/Users/kieran/Dropbox/k/code/git/recon2/rats/rambo.xml',
    '/Users/kieran/Dropbox/k/code/git/daaaaave/rats/rambo.xml'
    )


def parse_fidarestat_data():
    """
    Convert fidarestat data into simple text file, with columns
    gene = Ensembl
    C = non-diabetic controls
    D = diabetic untreated
    F = diabetic + fidarestat
    """

    book = xlrd.open_workbook(os.path.join(PATH, 'fidarestat.xls'))
    sbml = om.read_sbml(os.path.join(PATH, RAMBO_NAME))
    gene_list = om.get_list_of_genes(sbml)

    sheet = book.sheet_by_name("maxd1antilog2")
    file = os.path.join(PATH, 'fidarestat.txt')
    with open(file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['gene', 'C', 'D', 'F'])
        done = []
        for row in range(1, sheet.nrows):
            F = strip_string(sheet.cell(row, 3).value)
            C = strip_string(sheet.cell(row, 4).value)
            D = strip_string(sheet.cell(row, 5).value)
            gene = strip_string(sheet.cell(row, 17).value)
            if (gene in gene_list) and (gene not in done):
                done.append(gene)
                writer.writerow([gene, C, D, F])
        for gene in gene_list:
            if gene not in done:
                done.append(gene)
                writer.writerow([gene, 0, 0, 0])


def fidarestat_to_flux():
    """
    Convert fidarestat data into flux using SuperDaaaaave
    """
    sbml = om.read_sbml(os.path.join(PATH, RAMBO_NAME))

    # set minimal glucose medium
    media = [
        'EX_ca2(e)', 'EX_cl(e)', 'EX_fe2(e)', 'EX_fe3(e)', 'EX_h(e)', 'EX_h2o(e)',
        'EX_k(e)', 'EX_na1(e)', 'EX_nh4(e)', 'EX_so4(e)', 'EX_pi(e)', 'EX_o2(e)'
        ]
    block_all_imports(sbml)
#     set_import_bounds(sbml, 'EX_glc(e)', 1)
    sbml.getModel().getReaction('R_EX_glc_LPAREN_e_RPAREN_').getKineticLaw().getParameter('LOWER_BOUND').setValue(1.)
    sbml.getModel().getReaction('R_EX_glc_LPAREN_e_RPAREN_').getKineticLaw().getParameter('UPPER_BOUND').setValue(1.)
    set_import_bounds(sbml, media, INF)
    change_objective(sbml, 'DM_atp_c_')
#     v, atp_max = om.optimize_cobra_model(sbml)
    atp_max = 31.
    # force at least 50% of max atp generation
    sbml.getModel().getReaction('R_DM_atp_c_').getKineticLaw().getParameter('LOWER_BOUND').setValue(atp_max/2.)

    # limit sorbitol dehydrogenase
    sbml.getModel().getReaction('R_SBTD_D2').getKineticLaw().getParameter('UPPER_BOUND').setValue(1.)
    # allow sorbitol export to sink
    sbml.getModel().getReaction('R_SBTle').getKineticLaw().getParameter('LOWER_BOUND').setValue(-INF)

    # create gene data
    gene_names, gene_exp_C, gene_exp_D, gene_exp_T, = [], [], [], []
    file = os.path.join(PATH, 'fidarestat.txt')
    with open(file, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            gene = row['gene']
            C = row['C']
            D = row['D']
            T = row['F']
            gene_names.append(gene)
            gene_exp_C.append(float(C))
            gene_exp_D.append(float(D))
            gene_exp_T.append(float(T))
    gene_exp_C, gene_exp_D, gene_exp_T = np.array(gene_exp_C), np.array(gene_exp_D), np.array(gene_exp_T)

    # no sds given, assume equal to mean
    gene_exp_sd_C, gene_exp_sd_D, gene_exp_sd_T = gene_exp_C.copy(), gene_exp_D.copy(), gene_exp_T.copy()
    # but set zero means as small sd
    for gene_exp_sd_X in [gene_exp_sd_C, gene_exp_sd_D, gene_exp_sd_T]:
#         gene_exp_sd_X[gene_exp_sd_X == 0] = min(gene_exp_sd_X[gene_exp_sd_X != 0])/1e3
        gene_exp_sd_X[gene_exp_sd_X == 0] = min(gene_exp_sd_X[gene_exp_sd_X != 0])/2

    USE_SD = False # ignore SDs: assume all data has equal weighting

    flux_C, scaling_factor_C = om.call_SuperDaaaaave(sbml, gene_names, gene_exp_C, gene_exp_sd_C, MaxGrowth=False, UseSD=USE_SD, FixScaling=0)

    # fix glucose uptake in diabetic & treated at 6x level in control
#     set_import_bounds(sbml, 'EX_glc(e)', 6.)
    sbml.getModel().getReaction('R_EX_glc_LPAREN_e_RPAREN_').getKineticLaw().getParameter('LOWER_BOUND').setValue(10.)
    sbml.getModel().getReaction('R_EX_glc_LPAREN_e_RPAREN_').getKineticLaw().getParameter('UPPER_BOUND').setValue(10.)

    flux_D, scaling_factor_D = om.call_SuperDaaaaave(
        sbml, gene_names, gene_exp_D, gene_exp_sd_D, MaxGrowth=False,
        UseSD=USE_SD, FixScaling=scaling_factor_C, TargetFlux=flux_C
        )

    flux_T, scaling_factor_T = om.call_SuperDaaaaave(
        sbml, gene_names, gene_exp_T, gene_exp_sd_T, MaxGrowth=False,
        UseSD=USE_SD, FixScaling=scaling_factor_C, TargetFlux=flux_D
        )

    # remove small fluxes
    flux_C, flux_D, flux_T = np.array(flux_C), np.array(flux_D), np.array(flux_T)
    for X in [flux_C, flux_D, flux_T]:
        X[abs(X) < 1e-6] = 0

    # write results
    file = os.path.join(PATH, 'om_fidarestat.txt')
    with open(file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['rxn_id', 'C', 'D', 'T'])
        for index, reaction in enumerate(sbml.getModel().getListOfReactions()):
            rID = reaction.getId()
            C = flux_C[index]
            D = flux_D[index]
            T = flux_T[index]
            writer.writerow([rID, C, D, T])


def strip_string(txt):
    try:
        txt = '%g'%txt
    except:
        txt = str(txt)
    txt = txt.strip()
    if txt:
        if txt[0] == "'": txt = txt[1:]
    if txt:
        if txt[-1] == "'": txt = txt[:-1]
    return txt


def block_all_imports(sbml):
    """
    Written to mimic neilswainston matlab function blockAllImports
    """
    model = sbml.getModel()

    for rID in get_source_reactions(sbml):
        reaction = model.getReaction(rID)
        nR, nP = 0, 0
        for reactant in reaction.getListOfReactants():
            sID = reactant.getSpecies()
            if not model.getSpecies(sID).getBoundaryCondition():
                nR += 1
        for product in reaction.getListOfProducts():
            sID = product.getSpecies()
            if not model.getSpecies(sID).getBoundaryCondition():
                nP += 1
        kineticLaw = reaction.getKineticLaw()
        if (nR == 1) and (nP == 0):
            kineticLaw.getParameter('LOWER_BOUND').setValue(0)
        if (nR == 0) and (nP == 1):
            kineticLaw.getParameter('UPPER_BOUND').setValue(0)


def set_import_bounds(sbml, rxn_name_list, value):
    model = sbml.getModel()
    # convert single entries to lists
    if isinstance(rxn_name_list, str):
        rxn_name_list = [rxn_name_list]
    if isinstance(value, (int, float, long, complex)):
        value = [value] * len(rxn_name_list)
    for index, rID in enumerate(rxn_name_list):
        reaction = get_reaction_by_id(sbml, rID)
        if not reaction:
            print 'reaction %s not found'%rID
        else:
            nR, nP = 0, 0
            for reactant in reaction.getListOfReactants():
                sID = reactant.getSpecies()
                if not model.getSpecies(sID).getBoundaryCondition():
                    nR += 1
            for product in reaction.getListOfProducts():
                sID = product.getSpecies()
                if not model.getSpecies(sID).getBoundaryCondition():
                    nP += 1
            kineticLaw = reaction.getKineticLaw()
            val = abs(value[index])
            if (nR == 0) and (nP == 1):
                kineticLaw.getParameter('UPPER_BOUND').setValue(val)
            elif (nR == 1) and (nP == 0):
                kineticLaw.getParameter('LOWER_BOUND').setValue(-val)
            else:
                print 'reaction %s not import'%rID


def get_reaction_by_id(sbml, rID):
    model = sbml.getModel()
    reaction = model.getReaction(rID)
    if not reaction:
        # try cobra replacements
        rID = format_for_SBML_ID(rID)
        reaction = model.getReaction(rID)
    if not reaction:
        # try removing trailing underscore
        if rID[-1] == '_':
            rID = rID[:-1]
        reaction = model.getReaction(rID)
    if not reaction:
        # try adding "_in"
        reaction = model.getReaction(rID + '_in')
    if not reaction:
        # try known alternatives
        rID_map = {
            'R_DM_atp_c': 'R_HKt',  # alternative ATPase
            'R_EX_HC02175_LPAREN_e_RPAREN': 'R_EX_dca_LPAREN_e_RPAREN_',  # alternative C10:0
            'R_EX_HC02176_LPAREN_e_RPAREN': 'R_EX_ddca_LPAREN_e_RPAREN_',  # alternative C12:0
            'R_EX_docosac': 'R_EX_docosac_LPAREN_e_RPAREN_',  # alternative C22:0
            }
        if rID in rID_map:
            rID = rID_map[rID]
            reaction = get_reaction_by_id(sbml, rID)
    return reaction


def change_objective(sbml, rxn_name_list, objective_coeff=1):
    """
    Written to mimic the matlab function changeObjective from http://opencobra.sf.net/
    """
    model = sbml.getModel()
    for reaction in model.getListOfReactions():
        kineticLaw = reaction.getKineticLaw()
        kineticLaw.getParameter('OBJECTIVE_COEFFICIENT').setValue(0)
    # convert single entries to lists
    if isinstance(rxn_name_list, str):
        rxn_name_list = [rxn_name_list]
    if isinstance(objective_coeff, (int, float, long, complex)):
        objective_coeff = [objective_coeff] * len(rxn_name_list)
    for index, rID in enumerate(rxn_name_list):
        reaction = get_reaction_by_id(sbml, rID)
        if not reaction:
            print 'reaction %s not found'%rID
        else:
            kineticLaw = reaction.getKineticLaw()
            kineticLaw.getParameter('OBJECTIVE_COEFFICIENT').setValue(objective_coeff[index])


def get_source_reactions(sbml):
    """Determine source and sink reactions"""
    model = sbml.getModel()

    rID_list = []

    # strip out format used in recon 2.1
    species = model.getSpecies('M_carbon_e')
    if species:
        species.setBoundaryCondition(True)

    for reaction in model.getListOfReactions():
        nS, nP = 0, 0
        for reactant in reaction.getListOfReactants():
            sID = reactant.getSpecies()
            if not model.getSpecies(sID).getBoundaryCondition():
                nS += 1
        for product in reaction.getListOfProducts():
            sID = product.getSpecies()
            if not model.getSpecies(sID).getBoundaryCondition():
                nP += 1
        if (nS == 0) or (nP == 0):
            rID_list.append(reaction.getId())

    return rID_list


def format_for_SBML_ID(txt):
    """
    Written to mimic the matlab function formatForSBMLID from http://opencobra.sf.net/
    """
    txt = 'R_' + txt
    for symbol, replacement in [
            ('-', '_DASH_'),
            ('/', '_FSLASH_'),
            ('\\', '_BSLASH_'),
            ('(', '_LPAREN_'),
            (')', '_RPAREN_'),
            ('[', '_LSQBKT_'),
            (']', '_RSQBKT_'),
            (',', '_COMMA_'),
            ('.', '_PERIOD_'),
            ('\'', '_APOS_'),
            ('&', '&amp'),
            ('<', '&lt'),
            ('>', '&gt'),
            ('"', '&quot')]:
        txt = txt.replace(symbol, replacement)
    return txt


if __name__ == '__main__':
    parse_fidarestat_data()
    fidarestat_to_flux()

    print 'DONE!'
