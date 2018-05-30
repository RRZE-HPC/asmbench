#!/usr/bin/env python3
import copy
import itertools


# TODO use abc to force implementation of interface requirements

class Operand:
    def __init__(self, llvm_type):
        self.llvm_type = llvm_type
    
    def get_ir_repr(self):
        raise NotImplementedError()
    
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
        
    def get_ir_repr(self):
        return self.value
    
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
        
    def get_ir_repr(self):
        pass # TODO

    def get_constraint_char(self):
        return 'm'


class Register(Operand):
    # Persistent storage of register names
    _REGISTER_NAMES_IN_USE = {}
    
    @classmethod
    def reset(cls):
        cls._REGISTER_NAMES_IN_USE = {}
    
    @staticmethod
    def match(source_registers, destination_registers):
        matches = set()
        unmatched = set()
        for src in source_registers:
            matched = False
            for dst in destination_registers:
                if src.llvm_type == dst.llvm_type:
                    destination_registers.remove(dst)
                    src.join(dst)
                    matches.add(src)
                    matched = True
            if not matched:
                unmatched.add(src)
        return matches, unmatched
    
    def __init__(self, llvm_type, constraint_char='r', name='reg'):
        self.llvm_type = llvm_type
        self.constraint_char = constraint_char
        assert len(name) > 0, "name needs to be at least of length 1."
        if name in self._REGISTER_NAMES_IN_USE:
            i = 0
            name_test = name
            while name_test in self._REGISTER_NAMES_IN_USE:
                name_test = '{}.{}'.format(name, i)
                i += 1
            name = name_test
        self._name = name
        self._REGISTER_NAMES_IN_USE[self._name] = '%"{}"'.format(self._name)
    
    def get_ir_repr(self):
        return self._REGISTER_NAMES_IN_USE[self._name]
    
    def set_to_constant(self, value):
        self._REGISTER_NAMES_IN_USE[self._name] = value
    
    def get_constraint_char(self):
        return self.constraint_char
    
    def join(self, other):
        assert self.llvm_type == other.llvm_type, "LLVM types do not match."
        assert self.constraint_char == other.constraint_char, "Constraint chars do not match."
        if self._name == other._name:
            # nothing to do, already joined or equal
            pass
        else:
            del self._REGISTER_NAMES_IN_USE[self._name]
            self._name = other._name
    
    def __eq__(self, other):
        return (self.llvm_type == other.llvm_type and
                self.constraint_char == other.constraint_char and
                self._name == other._name)
    
    def __hash__(self):
        return hash((self.llvm_type, self.constraint_char, self._name))

class Synthable:
    def __init__(self):
        pass
    
    def build_ir(self):
        raise NotImplementedError()
    
    def get_source_registers(self):
        raise NotImplementedError()
    
    def get_destination_registers(self):
        raise NotImplementeError()

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
            if isinstance(sop, Immediate) or isinstance(sop, Register):
                operands.append('{type} {repr}'.format(type=sop.llvm_type, repr=sop.get_ir_repr()))
            else:
                raise NotImplemente("Only register and immediate operands are supported.")
        args = ', '.join(operands)
        
        # Build instruction from instruction and operands
        return ('{dst_reg} = call {dst_type} asm sideeffect'
                ' "{instruction}", "{constraints}" ({args})').format(
                    dst_reg=self.destination_operand.get_ir_repr(),
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
    
    def get_source_registers(self):
        sources = []
        last_destinations = []
        for s in self.synths:
            for src in s.get_source_registers():
                for dst in last_destinations:
                    if dst.llvm_type == src.llvm_type:
                        last_destinations.remove(dst)
                sources.append(src)
            last_destinations = s.get_destination_registers()
        return sources
    
    def get_destination_registers(self):
        if self.synths:
            return self.synths[-1].get_destination_registers()
        else:
            return []
    
    def build_ir(self):
        code = []
        last = None
        for s in self.synths:
            last_dests = last.get_source_registers() if last else []
            matched, unmatched = Register.match(s.get_source_registers(), last_dests)
            if not matched and last is not None:
                raise ValueError("Could not find a type match to serialize {} to {}.".format(
                    last, self))
            code.append(s.build_ir())
            last = s
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
    
    def build_ir(self):
        code = []
        for s in self.synths:
            code.append(s.build_ir())
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
    s = Serialized([i1, i2])
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
        instruction='add $2, $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r'), Register('i64', 'r')])
    s1 = Serialized([i1, i2])
    s2 = Serialized([s1, i3])
    s2.build_ir()
    s3 = Serialized([i4, i5])
    p1 = Parallelized([i6, s2, s3])
    print(p1.build_ir())
    print('srcs', [r.get_ir_repr() for r in p1.get_source_registers()])
    print('dsts', [r.get_ir_repr() for r in p1.get_destination_registers()])