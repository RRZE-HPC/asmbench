#!/usr/bin/env python3
import copy
import itertools


# TODO use abc to force implementation of interface requirements

# TODO Should Register contain a state (i.e. register name)? I think not. That should only come when
# synthesizing ir and evaluating joined(?) registers...

# NEEDS CONCEPTUAL WORK!



class Operand:
    def __init__(self, llvm_type):
        self.llvm_type = llvm_type
    
    def get_constraint_char(self):
        raise NotImplemented()


class Immediate(Operand):
    def __init__(self, llvm_type, value):
        Operand.__init__(self, llvm_type)
        self.value = value
    
    def get_constraint_char(self):
        return 'i'

class MemoryReference(Operand):
    '''
    offset + base + index*width
    
    OFFSET(BASE, INDEX, WIDTH) in AT&T assembly
    
    Possible operand values:
        offset: immediate integer (+/-)
        base: register
        index: register
        width: immediate 1,2,4 or 8
    '''
    def __init__(self, llvm_type, offset=None, base=None, index=None, width=None):
        self.offset = offset
        self.base = base
        self.index = index
        self.width = width
        self.destination = destination
        self.parallel = parallel
        
        # Sanity checks:
        if bool(index) ^ bool(width):
            raise ValueError("Index and width both need to be set, or None.")
        elif index and width:
            if not (isinstance(width, Immediate) and int(width.value) in [1,2,4,8]):
                raise ValueError("Width may only be immediate 1,2,4 or 8.")
            if not isinstance(index, Register):
                raise ValueError("Index must be a register.")

        if offset and not isinstance(offset, Immediate):
            raise ValueError("Offset must be an immediate.")
        if base and not isinstance(base, Register):
            raise ValueError("Offset must be a register.")

        if not index and not width and not offset and not base:
            raise ValueError("Must provide at least an offset or base.")

    def get_constraint_char(self):
        return 'm'


class Register(Operand):
    # Persistent storage of register names
    _REGISTER_NAMES_IN_USE = []
    
    def __init__(self, llvm_type, constraint_char='r', name='reg'):
        self.llvm_type = llvm_type
        self.constraint_char = constraint_char
        self.source = None
        self.destinations = []
        assert len(name) > 0, "name needs to be at least of length 1."
        self.name = name
    
    def set_source(self, source):
        if source is None:
            raise ValueError("source is already set. Only single source registers are allowed.")
        self.source = source
    
    def add_destination(self, destination):
        self.destinations.append(destination)
    
    def get_name(self):
        # Check if name is already in use and append integer
        name = self.name
        if name in self._REGISTER_NAMES_IN_USE:
            i = 0
            name_test = name
            while name_test in self._REGISTER_NAMES_IN_USE:
                name_test = '{}.{}'.format(name, i)
                i += 1
            name = name_test
        self._REGISTER_NAMES_IN_USE.append(name)
        return '%"{}"'.format(name)
    
    def get_constraint_char(self):
        return self.constraint_char


class Synthable:
    def __init__(self):
        pass
    
    def build_ir(self):
        raise NotImplemented()
    
    def get_source_registers(self):
        raise NotImplemented()
    
    def get_destination_registers(self):
        raise NotImplemente()

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={!r}'.format(k,v) for k,v in self.__dict__.items()
                       if not k.startswith('_')]))


class Operation(Synthable):
    '''Base class for operations.'''
    pass


class Instruction(Operation):
    def __init__(self, instruction, destination_operand, source_operands):
        self.instruction = instruction
        self.destination_operand = destination_operand
        assert isinstance(destination_operand, Register), "Destination needs to be a register."
        self.source_operands = source_operands
    
    def get_source_registers(self):
        return [sop for sop in self.source_operands if isinstance(sop, Register)]
    
    def get_destination_registers(self):
        if isinstance(self.destination_operand, Register):
            return [self.destination_operand]
        else:
            return []

    def build_ir(self):
        '''
        Build IR string based on in and out operand names and types.
        '''        
        # Build constraint string from operands
        constraints = ','.join(
            ['='+self.destination_operand.get_constraint_char()] +
            [sop.get_constraint_char() for sop in self.source_operands])
        
        # Build argument string from operands and register names
        operands = []
        for sop in self.source_operands:
            if isinstance(sop, Immediate):
                operands.append('{type} {val}'.format(type=sop.llvm_type, val=sop.value))
            elif isinstance(sop, Register):
                # Assuming register
                operands.append('{type} {name}'.format(type=sop.llvm_type, name=sop.get_name()))
            else:
                raise NotImplemente("Only register and immediate operands are supported.")
        args = ', '.join(operands)
        
        # Build instruction from instruction and operands
        return ('{dst_reg} = call {dst_type} asm sideeffect'
                ' "{instruction}", "{constraints}" ({args})\n').format(
                    dst_reg=self.destination_operand.get_name(),
                    dst_type=self.destination_operand.llvm_type,
                    instruction=self.instruction,
                    constraints=constraints,
                    args=args)


class Load(Operation):
    def __init__(self, chain_length, structure='linear'):
        '''
        *chain_length* is the number of pointers to place in memory.
        *structure* may be 'linear' (1-offsets) or 'random'.
        '''
        self.chain_length = chain_length
        self.structure = structure
    # TODO


class AddressGeneration(Operation):
    def __init__(self, offset, base, index, width, destination='base'):
        self.offset = offset
        self.base = base
        self.index = index
        self.width = width
        self.destination = destination
    # TODO


class Serialized(Synthable):
    def __init__(self, synths):
        self.synths = synths
        assert all([isinstance(s, Synthable) for s in synths]), "All elements need to be Sythable"


class Parallelized(Synthable):
    def __init__(self, synths):
        self.synths = synths
        assert all([isinstance(s, Synthable) for s in synths]), "All elements need to be Sythable"


if __name__ == '__main__':
    i = Instruction(
        instruction='add $2, $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r'), Immediate('i64', '1')])
    print(i.build_ir())