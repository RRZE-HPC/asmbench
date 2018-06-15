#!/usr/bin/env python3
import copy
import itertools
import collections
from pprint import pprint


# TODO use abc to force implementation of interface requirements

class Operand:
    def __init__(self, llvm_type):
        self.llvm_type = llvm_type
    
    def get_constraint_char(self):
        raise NotImplementedError()

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={!r}'.format(k,v) for k,v in self.__dict__.items()
                       if not k.startswith('_')]))


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
    def __init__(self, llvm_type, constraint_char='r'):
        self.llvm_type = llvm_type
        self.constraint_char = constraint_char
    
    def get_constraint_char(self):
        return self.constraint_char


class Synthable:
    def __init__(self):
        pass
    
    def build_ir(self, dst_reg_names, src_reg_names, used_registers):
        raise NotImplementedError()
    
    def get_source_registers(self):
        raise NotImplementedError()
    
    def get_destination_registers(self):
        raise NotImplementeError()

    def _get_unused_reg_name(self, used_registers):
        name = None
        i = 0
        while name in used_registers or name is None:
            name = '%"reg.{}"'.format(i)
            i += 1
        used_registers.add(name)
        return name

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={!r}'.format(k,v) for k,v in self.__dict__.items()
                       if not k.startswith('_')]))


class Operation(Synthable):
    '''Base class for operations.'''
    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={!r}'.format(k,v) for k,v in self.__dict__.items()
                       if not k.startswith('_')]))


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

    def build_ir(self, dst_reg_names, src_reg_names, used_registers):
        '''
        Build IR string based on in and out operand names and types.
        '''
        # Build constraint string from operands
        constraints = ','.join(
            ['='+self.destination_operand.get_constraint_char()] +
            [sop.get_constraint_char() for sop in self.source_operands])
        
        # Build argument string from operands and register names
        operands = []
        i = 0
        for sop in self.source_operands:
            if isinstance(sop, Immediate):
                operands.append('{type} {repr}'.format(
                    type=sop.llvm_type,
                    repr=sop.value))
            elif isinstance(sop, Register):
                operands.append('{type} {repr}'.format(
                    type=sop.llvm_type,
                    repr=src_reg_names[i]))
                i += 1
            else:
                raise NotImplemente("Only register and immediate operands are supported.")
        args = ', '.join(operands)
        
        # Build instruction from instruction and operands
        return ('{dst_reg} = call {dst_type} asm '
                ' "{instruction}", "{constraints}" ({args})').format(
                    dst_reg=dst_reg_names[0],
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
        raise NotImplementedError()


class Serialized(Synthable):
    def __init__(self, synths):
        self.synths = synths
        assert all([isinstance(s, Synthable) for s in synths]), "All elements need to be Sythable"
    
    def get_source_registers(self):
        if self.synths:
            return self.synths[0].get_source_registers()
        else:
            return []
    
    def get_destination_registers(self):
        if self.synths:
            return self.synths[-1].get_destination_registers()
        else:
            return []

    @staticmethod
    def match(source_registers, destination_registers):
        '''
        Find maximum number of matches from source (previous destinations) to
        destination (current source) registers.
        
        Return list of two-tuples of matches (src_idx, dst_idx)
        '''
        matched_pairs = []
        unmatched_dests = set(destination_registers)
        for dst_idx, dst in enumerate(destination_registers):
            for src_idx, src in enumerate(source_registers):
                if src.llvm_type == dst.llvm_type:
                    matched_pairs.append((src_idx, dst_idx))
                    unmatched_dests.discard(dst)
        
        return matched_pairs, unmatched_dests

    def generate_register_naming(self, dst_reg_names, src_reg_names, used_registers):
        reg_naming_out = []
        last_s = None
        for i, s in enumerate(self.synths):
            if i == 0:
                # first source is passed in from outside
                src_naming = src_reg_names
            else:
                # match with previous destinations
                src_naming = []
                for src in s.get_source_registers():
                    # Find matching destination from previous synths
                    match = False
                    for dst_idx, dst in enumerate(last_s.get_destination_registers()):
                        if dst.llvm_type == src.llvm_type:
                            match = True
                            src_naming.append(dst_naming[dst_idx])
                    if not match:
                        raise ValueError("Unable to find match.")

            if i == len(self.synths) - 1:
                # last destination is passed in from outside
                dst_naming = dst_reg_names
            else:
                dst_naming = [self._get_unused_reg_name(used_registers)
                              for j in s.get_destination_registers()]

            reg_naming_out.append((dst_naming, src_naming))
            last_s = s
        return reg_naming_out, used_registers

    def build_ir(self, dst_reg_names, src_reg_names, used_registers=None):
        if used_registers is None:
            used_registers = set(dst_reg_names + src_reg_names)
        reg_names, used_registers = self.generate_register_naming(
            dst_reg_names, src_reg_names, used_registers)
        code = []
        for s, r in zip(self.synths, reg_names):
            code.append(s.build_ir(*r, used_registers))
        return '\n'.join(code)


class Parallelized(Synthable):
    def __init__(self, synths):
        self.synths = synths
        assert all([isinstance(s, Synthable) for s in synths]), "All elements need to be Sythable"

    def get_source_registers(self):
        sources = []
        for s in self.synths:
            sources += s.get_source_registers()
        return sources
    
    def get_destination_registers(self):
        destinations = []
        for s in self.synths:
            destinations += s.get_destination_registers()
        return destinations

    def generate_register_naming(self, dst_reg_names, src_reg_names, used_registers):
        # Split reg_naming among all synths
        reg_naming_out = []
        for s in self.synths:
            n_dsts = len(s.get_destination_registers())
            n_srcs = len(s.get_source_registers())
            reg_naming_out.append((dst_reg_names[:n_dsts], src_reg_names[:n_srcs]))
            dst_reg_names, src_reg_names = (dst_reg_names[n_dsts:], src_reg_names[n_srcs:])
        return reg_naming_out, used_registers

    def build_ir(self, dst_reg_names, src_reg_names, used_registers=None):
        if used_registers is None:
            used_registers = set(dst_reg_names + src_reg_names)
        reg_names, used_registers = self.generate_register_naming(
            dst_reg_names, src_reg_names, used_registers)
        code = []
        for s, r in zip(self.synths, reg_names):
            code.append(s.build_ir(*r, used_registers))
        return '\n'.join(code)


if __name__ == '__main__':
    i1 = Instruction(
        instruction='add $2, $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r'), Immediate('i64', '1')])
    i2 = Instruction(
        instruction='sub $2, $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r'), Immediate('i64', '1')])
    i3 = Instruction(
        instruction='mul $1, $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r'), Register('i64', 'r')])
    i4 = Instruction(
        instruction='div $2, $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r'), Immediate('i64', '23')])
    i5 = Instruction(
        instruction='mul $2, $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r'), Immediate('i64', '23')])
    i6 = Instruction(
        instruction='inc $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r')])
    s1 = Serialized([i1, i2])
    s2 = Serialized([s1, i3])
    print(s1.build_ir(['%out'], ['%in']), '\n')
    print(s2.build_ir(['%out'], ['%in']), '\n')
    s3 = Serialized([i4, i5])
    p1 = Parallelized([i6, s2, s3])
    print(p1.build_ir(['%out.0', '%out.1', '%out.2'], ['%in.0', '%in.1', '%in.2']), '\n')

    s4 = Serialized([i1, i2, i3, i4, i5, i6])
    print(s4.build_ir(['%out'], ['%in']), '\n')
