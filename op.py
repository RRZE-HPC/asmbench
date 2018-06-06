#!/usr/bin/env python3
import copy
import itertools
import collections


# TODO use abc to force implementation of interface requirements

class RegisterMapping:
    def __init__(self, prefix='reg'):
        self.prefix = prefix
        self.mapping = collections.defaultdict(self._unused_name)
        self.source_sink = []
        self.back_edges = []
    
    def source(self, reg):
        print('src ', reg)
        if any([reg in ss for ss in self.source_sink]):
            raise ValueError("Already registered as source or sink.")
        self.source_sink.append((reg, None))
    
    def sink(self, reg):
        print('sink', reg)
        if any([reg in ss for ss in self.source_sink]):
            raise ValueError("Already registered as source or sink.")
        self.source_sink.append((None, reg))
    
    def _unused_name(self):
        # Find free name
        i = 0
        while True:
            name = '%{}.{}'.format(i, self.prefix)
            if name not in self.mapping.values():
                break
            i += 1
        return name
    
    def connect(self, regs):
        name = self._unused_name()

        has_source = None
        sinks = []
        sinks_before_source = []
        for source, sink in self.source_sink:
            if source in regs:
                if has_source is not None:
                    raise ValueError('Connected registers may only have one source. '
                                     'Multiple found.')
                has_source = source
            elif sink in regs:
                sinks.append(sink)
                if has_source is not None:
                    sinks_before_source.append(sink)

        if has_source is None:
            raise ValueError('No sources found in given register list. Connected registers need to '
                             'have one source.')
        if not sinks:
            raise ValueError('No sinks found in given register list. Need at least one sink.')

        for reg in regs:
            if not any([reg in ss for ss in self.source_sink]):
                raise ValueError("Register has not been registered as sink nor source.")

        for r in regs:
            self.mapping[r] = name
            if r in sinks_before_source:
                self.mapping[r] += '_'

        if sinks_before_source:
            self.back_edges.append((has_source, sinks[0]))
        
        return name

    def connected(self, regs):
        """Return True if all registers in regs are connected."""
        # Compile list of names, with '_' stripped from right
        names = [self.mapping[r].rstrip('_') for r in regs]
        # check that all entries are equal
        return not names or names.count(names[0]) == len(names)

    def set_constant(self, reg, value):
        """Set register to constant value."""
        if reg in self.mapping:
            raise ValueError("Register is already connected/set constant elsewhere.")
        self.mapping[reg] = value
    
    def get_ir_repr(self, reg):
        if reg in self.mapping:
            return self.mapping(reg)
        else:
            raise ValueError("Register has not been connected.")


class Operand:
    def __init__(self, llvm_type):
        self.llvm_type = llvm_type
    
    def get_ir_repr(self):
        raise NotImplementedError()
    
    def get_constraint_char(self):
        raise NotImplementedError()

    def __repr__(self):
        return hex(id(self))+'{}({})'.format(
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
        raise NotImplementedError("TODO")

    def get_constraint_char(self):
        return 'm'


class Register(Operand):
    def __init__(self, llvm_type, constraint_char='r'):
        self.llvm_type = llvm_type
        self.constraint_char = constraint_char
    
    def get_ir_repr(self, reg_mapping):
        return reg_mapping.get_ir_repr(self)
    
    def get_constraint_char(self):
        return self.constraint_char


class Synthable:
    def __init__(self):
        pass
    
    def build_ir(self, reg_mapping):
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

    def generate_register_mapping(self, reg_mapping):
        print(self, 'generate_register_mapping')
        for r in self.get_source_registers():
            reg_mapping.sink(r)
        for r in self.get_destination_registers():
            reg_mapping.source(r)

    def build_ir(self, reg_mapping):
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
                operands.append('{type} {repr}'.format(
                    type=sop.llvm_type,
                    repr=sop.get_ir_repr(reg_mapping)))
            else:
                raise NotImplemente("Only register and immediate operands are supported.")
        args = ', '.join(operands)
        
        # Build instruction from instruction and operands
        return ('{dst_reg} = call {dst_type} asm sideeffect'
                ' "{instruction}", "{constraints}" ({args})').format(
                    dst_reg=self.destination_operand.get_ir_repr(reg_mapping),
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

    @staticmethod
    def match(source_registers, destination_registers):
        '''
        Find maximum number of matches from source (previous destinations) to
        destination (current source) registers.
        
        Return list of two-tuples of matches (src, dst) and set of unmachted destination registers.
        '''
        matched_pairs = []
        unmatched_dests = set(destination_registers)
        for dst in destination_registers:
            for src in source_registers:
                if src.llvm_type == dst.llvm_type:
                    matched_pairs.append((src, dst))
                    unmatched_dests.discard(dst)
        
        return matched_pairs, unmatched_dests
    
    def generate_register_mapping(self, reg_mapping=None):
        if reg_mapping is None:
            reg_mapping = RegisterMapping()

        # First register all registers for later mapping
        for s in self.synths:
            s.generate_register_mapping()

        last_s = s
        for s in self.synths:
            matched_pairs, unmatched_dests = self.match(
                last_s.get_destination_registers(), s.get_source_registers())

            for p in matched_pairs:
                #print(reg_mapping.mapping)
                #print(reg_mapping.source_sink)
                #print(p)
                reg_mapping.connect(p)
            
            if not matched_pairs:
                raise ValueError("Could not find a type match to serialize {} to {}.".format(
                    last_s, s))
        return reg_mapping

    def build_ir(self, reg_mapping=None):
        reg_mapping = self.generate_register_mapping(reg_mapping)
        code = []
        for s in self.synths:
            code.append(s.build_ir(reg_mapping))
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
        instruction='inc $0',
        destination_operand=Register('i64', 'r'),
        source_operands=[Register('i64', 'r')])
    s1 = Serialized([i1, i2])
    s2 = Serialized([s1, i3])
    s2.build_ir()
    s3 = Serialized([i4, i5])
    p1 = Parallelized([i6, s2, s3])
    print(p1.build_ir())
    print('srcs', [r.get_ir_repr() for r in p1.get_source_registers()])
    print('dsts', [r.get_ir_repr() for r in p1.get_destination_registers()])