
import numpy as np
from sys import path
from aiida.orm.data.base import Str, Bool, Int, Float
from aiida.orm import DataFactory
from aiida.work.workchain import WorkChain, if_, while_, return_

path.insert(0, 'glosim2')
from libmatch.soap import get_soap
from libmatch.utils import ase2qp, get_spkit, get_spkitMax

StructureData = DataFactory('structure')
ArrayData = DataFactory('array')


class SOAPWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(SOAP, cls).define(spec)
        spec.input('aiida_structure', valid_type=StructureData)
        spec.input('spkit_max', valid_type=ParameterData)
        spec.input('aononymize', valid_type=Bool, default=Bool(True))
        spec.input('scale', valid_type=Bool, default=Bool(True))
        spec.input('scale_per', valid_type=Str, default=Str('site'))
        spec.input('soapargs', valid_type=ParameterData, required=False)
        spec.outline(
            cls.get_quippy_atoms,
            if_(cls.check_anonymize)(
                cls.anonymize_structure,
            )
            if_(cls.check_scale)(
                cls.scale_volume,
            )
            cls.get_soap_fingerprint
        )
        spec.output('soap', valid_type=ArrayData)

    def check_anonymize(self):
        return self.inputs.anonymize

    def check_scale(self):
        return self.inputs.scale

    def validate_inputs(self):
        pass

    def get_quippy_atoms(self):
        ase_atoms = self.inputs.anonymous_aiida_structure.get_ase()
        quippy_atoms = ase2qp(ase_atoms)
        self.ctx.quippy_atoms = quippy_atoms

    def anonymize_structure(self):
        qp = self.ctx.quippy_atoms
        n_atoms = qp.n
        qp.set_atomic_numbers([1] * n_atoms)
        qp.set_chemical_symbols(['H'] * n_atoms)

        self.ctx.quippy_atoms = qp

    def scale_volume(self):
        qp = self.ctx.quippy_atoms
        if self.inputs.scale_per == 'site':
            n_atoms = qp.n
        elif self.inputs.scale_per == 'cell':
            n_atoms = 1  # scaling volume to 1/cell is the same as having 1 atom
        new_cell = qp.get_cell() / np.cbrt(qp.cell_volume() / n_atoms)
        qp.set_cell(new_cell)
        new_pos = qp.get_positions() / np.linalg.norm(qp.get_cell(), axis=1) * \
            np.linalg.norm(new_cell, axis=1)
        qp.set_positions(new_pos)
        self.ctx.quippy_atoms = qp

    def get_soap_fingerprint(self):
        qp = self.ctx.quippy_atoms
        soapargs = self.inputs.soapargs.get_dict()
        spkit = get_spkit(qp)
        soap = get_soap(qp, spkit=spkit,
                        spkitMax=self.inputs.spkit_max.get_dict(), **soapargs)
        soap_array = np.array(soap.values())
        self.out('soap', ArraData(soap_array))
