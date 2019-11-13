'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import re

import libsbml
from scipy.sparse import lil_matrix

import numpy as np


def read_sbml(filename):
    '''Read an SBML file from specified path.'''
    reader = libsbml.SBMLReader()
    return reader.readSBMLFromFile(filename)


def convert_sbml_to_cobra(sbml):
    '''Get Cobra matrices from SBML model.'''
    model = sbml.getModel()

    S = lil_matrix((model.getNumSpecies(), model.getNumReactions()))

    lb = []
    ub = []
    c = []
    b = [0.0] * S.shape[0]
    rev = []
    spec_ids = [species.getId() for species in model.getListOfSpecies()]

    for j, reaction in enumerate(model.getListOfReactions()):
        _update_s_matrix(model, reaction, S, j, spec_ids)

        kinetic_law = reaction.getKineticLaw()
        rxn_lb = kinetic_law.getParameter('LOWER_BOUND').getValue()
        rxn_rev = reaction.getReversible()

        if rxn_lb < 0:
            rxn_rev = True

        lb.append(rxn_lb)
        ub.append(kinetic_law.getParameter('UPPER_BOUND').getValue())
        c.append(kinetic_law.getParameter('OBJECTIVE_COEFFICIENT').getValue())
        rev.append(rxn_rev)

    return {'S': S,
            'lb': np.array(lb),
            'ub': np.array(ub),
            'c': np.array(c),
            'b': np.array(b),
            'rev': np.array(rev)}


def get_gene_associations(sbml):
    '''Get gene associations.'''
    gene_assoc = []

    model = sbml.getModel()

    for reaction in model.getListOfReactions():
        match = re.search(r'<p>GENE_ASSOCIATION:([^<]*)</p>',
                          reaction.getNotesString())

        gene_assoc.append(match.group(1))

    return gene_assoc


def _update_s_matrix(model, reaction, S, j, spec_ids):
    '''Update S-matrix.'''
    for reactant in reaction.getListOfReactants():
        spec_id = reactant.getSpecies()

        if not model.getSpecies(spec_id).getBoundaryCondition():
            i = spec_ids.index(spec_id)
            S[i, j] = S[i, j] - reactant.getStoichiometry()

    for product in reaction.getListOfProducts():
        spec_id = product.getSpecies()

        if not model.getSpecies(spec_id).getBoundaryCondition():
            i = spec_ids.index(spec_id)
            S[i, j] = S[i, j] + product.getStoichiometry()
