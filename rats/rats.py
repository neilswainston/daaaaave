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


def parse_fidarestat_data():
    """
    Convert fidarestat data into simple text file, with columns
    gene = Ensembl
    C = non-diabetic controls
    D = diabetic untreated
    F = diabetic + fidarestat
    """

    book = xlrd.open_workbook(os.path.join(PATH, 'fidarestat.xls'))
    sbml = om.read_sbml(os.path.join(PATH, 'rambo.xml'))
    gene_list = om.get_list_of_genes(sbml)
    print gene_list

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
            print gene
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
    sbml = om.read_sbml(os.path.join(PATH, 'rambo.xml'))

    # create gene data
    gene_names, gene_exp_C, gene_exp_D, gene_exp_F, = [], [], [], []

    file = os.path.join(PATH, 'fidarestat.txt')
    with open(file, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            gene = row['gene']
            C = row['C']
            D = row['D']
            F = row['F']
            gene_names.append(gene)
            gene_exp_C.append(float(C))
            gene_exp_D.append(float(D))
            gene_exp_F.append(float(F))
    gene_exp_C, gene_exp_D, gene_exp_F = np.array(gene_exp_C), np.array(gene_exp_D), np.array(gene_exp_F)

    # no sds given, assume equal to mean
    gene_exp_sd_C, gene_exp_sd_D, gene_exp_sd_F = gene_exp_C.copy(), gene_exp_D.copy(), gene_exp_F.copy()
    # but set zero means as small sd
    for gene_exp_sd_X in [gene_exp_sd_C, gene_exp_sd_D, gene_exp_sd_F]:
#         gene_exp_sd_X[gene_exp_sd_X == 0] = min(gene_exp_sd_X[gene_exp_sd_X != 0])/1e3
        gene_exp_sd_X[gene_exp_sd_X == 0] = min(gene_exp_sd_X[gene_exp_sd_X != 0])/2

    # turn off constraints
    for reaction in sbml.getModel().getListOfReactions():
    	kineticLaw = reaction.getKineticLaw()
    	LB = kineticLaw.getParameter('LOWER_BOUND')
    	if LB.getValue() < 0:
    		LB.setValue(-INF)
    	elif LB.getValue() > 0:
    		LB.setValue(0)
    	UB = kineticLaw.getParameter('UPPER_BOUND')
    	if UB.getValue() < 0:
    		UB.setValue(0)
    	elif UB.getValue() > 0:
    		UB.setValue(INF)

    # fix glucose uptake = 1 in control
    reaction = sbml.getModel().getReaction('R_EX_glc_LPAREN_e_RPAREN_')
    reaction.getKineticLaw().getParameter('LOWER_BOUND').setValue(1)
    reaction.getKineticLaw().getParameter('UPPER_BOUND').setValue(1)

    USE_SD = False # ignore SDs: assume all data has equal weighting

    flux_C, scaling_factor_C = om.call_SuperDaaaaave(sbml, gene_names, gene_exp_C, gene_exp_sd_C, MaxGrowth=False, UseSD=USE_SD, FixScaling=0)

    # unfix glucose uptake in diabetic & treated, but fix scaling factor
    reaction = sbml.getModel().getReaction('R_EX_glc_LPAREN_e_RPAREN_')
    reaction.getKineticLaw().getParameter('LOWER_BOUND').setValue(0)
    reaction.getKineticLaw().getParameter('UPPER_BOUND').setValue(INF)
    flux_D, scaling_factor_D = om.call_SuperDaaaaave(
        sbml, gene_names, gene_exp_D, gene_exp_sd_D, MaxGrowth=False,
        UseSD=USE_SD, FixScaling=scaling_factor_C, TargetFlux=flux_C
        )
    flux_F, scaling_factor_F = om.call_SuperDaaaaave(
        sbml, gene_names, gene_exp_F, gene_exp_sd_F, MaxGrowth=False,
        UseSD=USE_SD, FixScaling=scaling_factor_C, TargetFlux=flux_C
        )

    # remove small fluxes
    flux_C, flux_D, flux_F = np.array(flux_C), np.array(flux_D), np.array(flux_F)
    for X in [flux_C, flux_D, flux_F]:
        X[abs(X) < 1e-6] = 0

    # write results
    file = os.path.join(PATH, 'om_fidarestat.txt')
    with open(file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['rxn_id', 'C', 'D', 'F'])
        for index, reaction in enumerate(sbml.getModel().getListOfReactions()):
            rID = reaction.getId()
            C = flux_C[index]
            D = flux_D[index]
            F = flux_F[index]
            writer.writerow([rID, C, D, F])


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


if __name__ == '__main__':
#     parse_fidarestat_data()
    fidarestat_to_flux()

    print 'DONE!'