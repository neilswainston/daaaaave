'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=broad-except
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
import csv
import re

from daaave.model import get_gene_associations
from daaave.utils import set_diff
import numpy as np


def load_gene_data(filename):
    '''Load gene data.'''
    gene_names = []
    gene_exp = []
    gene_exp_sd = []

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')

        for row in reader:
            gene_names.append(row['gene'])
            gene_exp.append(float(row['mean']))
            gene_exp_sd.append(float(row['std']))

    return gene_names, np.array(gene_exp), np.array(gene_exp_sd)


def load_flux_data(filename):
    '''Load flux data.'''
    exp_flux = []
    rxn_names = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')

        for row in reader:
            rxn_names.append(row[0])
            exp_flux.append(float(row[1]))

    return exp_flux, rxn_names


def genes_to_rxns(sbml, gene_names, gene_exp, gene_exp_sd):
    '''Match gene-level data to reaction-level data.'''

    rxn_exp = []
    rxn_exp_sd = []

    gene_names = [gene_name.replace('-', '_')
                  for gene_name in gene_names]

    gene_assoc = get_gene_associations(sbml)

    for gene_assn in gene_assoc:
        gene_assn = gene_assn.replace('-', '_')
        gene_assn = gene_assn.replace(' AND ', ' and ')
        gene_assn = gene_assn.replace(' OR ', ' or ')
        gene_assn = gene_assn.replace(' )', ')')
        gene_assn = gene_assn.replace('( ', '(')

        gene_list = re.findall(r'\b([\w]*)\b', gene_assn)
        gene_list = set_diff(gene_list, ['and', 'or', ''])

        for gene in gene_list:
            j = gene_names.index(gene)
            str_ng = repr(gene_exp[j])
            str_ng_sd = repr(gene_exp_sd[j])

            gene_assn = re.sub(
                r'\b' + gene + r'\b', str_ng + '~' + str_ng_sd, gene_assn)

        nr, nr_sd = _map_gene_data(gene_assn)
        rxn_exp.append(nr)
        rxn_exp_sd.append(nr_sd)

    rxn_exp = np.array(rxn_exp)
    rxn_exp_sd = np.array(rxn_exp_sd)

    # sds 0 -> small
    rxn_exp_sd[rxn_exp_sd == 0] = min(rxn_exp_sd[rxn_exp_sd != 0]) / 2

    return rxn_exp, rxn_exp_sd


def _map_gene_data(gene_assn):
    '''Map string '(x1~x1SD) and (x2~x2SD) or (x3~x3SD)' to string y~ySD.'''
    nr, nr_sd = np.nan, np.nan
    a_pm_b = r'[0-9\.]+~[0-9\.]+'
    if gene_assn:
        while np.isnan(nr):
            try:
                match_expr = r'\A([0-9\.]+)~([0-9\.]+)\Z'
                match = re.search(match_expr, gene_assn)
                str_nr, str_nr_sd = match.group(1), match.group(2)
                nr, nr_sd = float(str_nr), float(str_nr_sd)
            except Exception:
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
                        replace_expr = _a_and_b(lhs, rhs)
                        gene_assn = re.sub(
                            r'\b' + lhs + ' and ' + rhs + r'\b',
                            replace_expr, gene_assn)
                # replace ORs
                match_expr = r'(' + a_pm_b + r') or (' + a_pm_b + r')'
                match = re.search(match_expr, gene_assn)
                for ind, match in enumerate(re.findall(match_expr, gene_assn)):
                    if ind == 0:
                        lhs = match[0]
                        rhs = match[1]
                        replace_expr = _a_or_b(lhs, rhs)
                        gene_assn = re.sub(
                            r'\b' + lhs + ' or ' + rhs + r'\b',
                            replace_expr, gene_assn)
    return nr, nr_sd


def _a_and_b(str1, str2):
    '''A AND B.'''
    ng1, ng1_sd = _parse_a_b(str1)
    ng2, ng2_sd = _parse_a_b(str2)

    ng12, ng12_sd = [ng1, ng2], [ng1_sd, ng2_sd]
    j = np.argmin(ng12)
    ng, ng_sd = ng12[j], ng12_sd[j]

    return repr(ng) + '~' + repr(ng_sd)


def _a_or_b(str1, str2):
    '''A OR B.'''
    ng1, ng1_sd = _parse_a_b(str1)
    ng2, ng2_sd = _parse_a_b(str2)

    ng = ng1 + ng2
    ng_sd = np.sqrt(ng1_sd**2.0 + ng2_sd**2.0)

    return repr(ng) + '~' + repr(ng_sd)


def _parse_a_b(term):
    '''Parse A B term.'''
    match_expr = r'\A([0-9\.]+)~([0-9\.]+)'
    match = re.search(match_expr, term)
    return float(match.group(1)), float(match.group(2))
