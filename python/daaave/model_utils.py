'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import re

import libsbml
from scipy.sparse import lil_matrix


def read_sbml(filename):
    '''Read an SBML file from specified path.'''
    reader = libsbml.SBMLReader()
    return reader.readSBMLFromFile(filename)


def convert_sbml_to_cobra(sbml, bound=1000):
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

        rxn_lb = max(get_parameter(reaction, 'LOWER_BOUND'), -bound)

        rxn_rev = reaction.getReversible()

        if rxn_lb < -bound:
            rxn_lb = -bound
        if rxn_lb < 0:
            rxn_rev = True

        lb.append(rxn_lb)
        ub.append(min(get_parameter(reaction, 'UPPER_BOUND'), bound))
        c.append(get_parameter(reaction, 'OBJECTIVE_COEFFICIENT'))
        rev.append(rxn_rev)

    return {'S': S,
            'lb': lb,
            'ub': ub,
            'c': c,
            'b': b,
            'rev': rev}


def get_gene_associations(sbml):
    '''Get gene associations.'''
    gene_assocs = []

    model = sbml.getModel()

    for reaction in model.getListOfReactions():
        match = re.search(r'<p>GENE_ASSOCIATION:([^<]*)</p>',
                          reaction.getNotesString())

        if match:
            gene_assoc = match.group(1)
        else:
            # Support SBML v3:
            fbc = reaction.getPlugin('fbc')
            gpa = fbc.getGeneProductAssociation()

            if gpa:
                gene_assoc = _process_assoc(gpa.getAssociation())
            else:
                gene_assoc = ''

        gene_assocs.append(gene_assoc)

    return gene_assocs


def rescale_model(sbml, rxn_exp, rxn_exp_sd, gene_to_scale):
    '''Rescale model.'''
    model = sbml.getModel()
    rxn_names = [
        reaction.getName()
        for reaction in model.getListOfReactions()
    ]
    uptake = rxn_names.index(gene_to_scale)
    rxn_exp_sd = rxn_exp_sd / rxn_exp[uptake]
    rxn_exp = rxn_exp / rxn_exp[uptake]
    reaction = model.getReaction(uptake)
    set_parameter(reaction, 'LOWER_BOUND', 1)
    set_parameter(reaction, 'UPPER_BOUND', 1)

    return sbml, rxn_exp, rxn_exp_sd


def set_parameter(reaction, param_id, value):
    '''Set parameter.'''
    kinetic_law = reaction.getKineticLaw()

    if not kinetic_law:
        kinetic_law = reaction.createKineticLaw()

    parameter = kinetic_law.getParameter(param_id)

    if not parameter:
        parameter = kinetic_law.createParameter()
        parameter.setId(param_id)

    parameter.setValue(value)


def get_parameter(reaction, param_id):
    '''Get parameter.'''
    kinetic_law = reaction.getKineticLaw()

    if not kinetic_law or not kinetic_law.getParameter(param_id):
        if param_id == 'LOWER_BOUND':
            return float('-inf')

        if param_id == 'UPPER_BOUND':
            return float('inf')

        if param_id == 'OBJECTIVE_COEFFICIENT':
            return 0

    return kinetic_law.getParameter(param_id).getValue()


def _process_assoc(ass):
    ''' Recursively convert gpr association to a gpr string.
    Defined as inline functions to not pass the replacement dict around.
    '''
    if ass.isFbcOr():
        return ' '.join(
            ['(', ' or '.join(_process_assoc(c)
                              for c in ass.getListOfAssociations()), ')']
        )

    if ass.isFbcAnd():
        return ' '.join(
            ['(', ' and '.join(_process_assoc(c)
                               for c in ass.getListOfAssociations()), ')'])

    if ass.isGeneProductRef():
        gid = ass.getGeneProduct()

        if gid.startswith('G_'):
            return gid[2:]

        return gid

    return None


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
