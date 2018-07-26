#!/usr/bin/env python3
import collections
import itertools

from asmjit import op, bench
import jit


def jit_based_benchs():
    modules = collections.OrderedDict()
    modules['lea base LAT'] = jit.AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallel=1,
        serial=5)

    modules['lea base TP'] = jit.AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallel=10,
        serial=1)

    modules['lea base+offset LAT'] = jit.AddressGenerationBenchmark(
        offset=('i', None, '23'),
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallel=1,
        serial=5)

    modules['lea base+offset TP'] = jit.AddressGenerationBenchmark(
        offset=('i', None, '23'),
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallel=10,
        serial=1)

    modules['lea index*width LAT'] = jit.AddressGenerationBenchmark(
        offset=None,
        base=None,
        index=('r', 'i64', '1'),
        width=('i', None, '4'),
        destination='index',
        parallel=1,
        serial=5)

    modules['lea index*width TP'] = jit.AddressGenerationBenchmark(
        offset=None,
        base=None,
        index=('r', 'i64', '1'),
        width=('i', None, '4'),
        destination='index',
        parallel=10,
        serial=1)

    modules['lea offset+index*width LAT'] = jit.AddressGenerationBenchmark(
        offset=('i', 'i64', '-0x8'),
        base=None,
        index=('r', 'i64', '51'),
        width=('i', None, '4'),
        destination='index',
        parallel=1,
        serial=5)

    modules['lea offset+index*width TP'] = jit.AddressGenerationBenchmark(
        offset=('i', 'i64', '-0x8'),
        base=None,
        index=('r', 'i64', '51'),
        width=('i', None, '4'),
        destination='index',
        parallel=10,
        serial=1)

    modules['lea base+index*width LAT'] = jit.AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallel=1,
        serial=5)

    modules['lea base+index*width TP'] = jit.AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallel=10,
        serial=1)

    modules['lea base+offset+index*width LAT'] = jit.AddressGenerationBenchmark(
        offset=('i', None, '42'),
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallel=1,
        serial=5)

    modules['lea base+offset+index*width TP'] = jit.AddressGenerationBenchmark(
        offset=('i', None, '42'),
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallel=10,
        serial=1)

    modules['LD linear LAT'] = jit.LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        structure='linear',
        parallel=1,
        serial=2)

    modules['LD random LAT'] = jit.LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        structure='random',
        parallel=1,
        serial=2)

    modules['LD linear TP'] = jit.LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        structure='linear',
        parallel=4,
        serial=2)

    modules['LD random TP'] = jit.LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        structure='random',
        parallel=4,
        serial=2)

    for name, module in modules.items():
        r = module.build_and_execute(repeat=3)
        cy_per_it = min(r['runtimes']) * r['frequency'] / (
                    r['iterations'] * module.parallel * module.serial)
        print('{key:<30} {cy_per_it:.3f} cy'.format(
            key=name,
            module=module,
            cy_per_it=cy_per_it,
            runtime_sum=sum(r['runtimes'])))


if __name__ == '__main__':
    bench.setup_llvm()
    instructions = [
        (i, op.Instruction.from_string(i))
        for i in [
            'add {src:i32:1}, {srcdst:i32:r}',
            'add {src:i32:1}, {srcdst:i64:r}',
            'inc {srcdst:i64:r}',
            'mov {src:i32:1}, {srcdst:i64:r}',
            'sub {src:i32:1}, {srcdst:i64:r}',
            'vaddpd {src:<4 x double>:x}, {src:<4 x double>:x}, {dst:<4 x double>:x}',
            'vaddsd {src:double:x}, {src:double:x}, {dst:double:x}',
            'vaddss {src:float:x}, {src:float:x}, {dst:float:x}',
            'vcvtsi2ss {src:i64:r}, {src:float:x}, {dst:float:x}',
            'vfmadd213pd {src:<4 x double>:x}, {src:<4 x double>:x}, {srcdst:<4 x double>:x}',
            'vfmadd213pd {src:<2 x double>:x}, {src:<2 x double>:x}, {srcdst:<2 x double>:x}',
            'vfmadd213ps {src:<4 x double>:x}, {src:<4 x double>:x}, {srcdst:<4 x double>:x}',
            'vfmadd213ps {src:<2 x double>:x}, {src:<2 x double>:x}, {srcdst:<2 x double>:x}',
            'vfmadd213sd {src:double:x}, {src:double:x}, {srcdst:double:x}',
            'vfmadd213ss {src:float:x}, {src:float:x}, {srcdst:float:x}',
            'vinsertf128 {src:i8:0}, {src:<2 x double>:x}, {src:<4 x double>:x}, {dst:<4 x double>:x}',
            'vmulpd {src:<4 x double>:x}, {src:<4 x double>:x}, {dst:<4 x double>:x}',
            'vmulsd {src:double:x}, {src:double:x}, {dst:double:x}',
            'vmulss {src:float:x}, {src:float:x}, {dst:float:x}',
            'vsubsd {src:double:x}, {src:double:x}, {dst:double:x}',
            'vsubss {src:float:x}, {src:float:x}, {dst:float:x}'
            ]
    ]
    instructions_measured = collections.OrderedDict()
    for i_str, i in instructions:
        lat, tp = bench.bench_instructions(
            [i], serial_factor=8, throughput_serial_factor=8, parallel_factor=10)
        print('{:<30} LAT {:.3f} cy'.format(i_str, lat))
        print('{:<30} TP  {:.3f} cy'.format(i_str, tp))
        instructions_measured[i_str] = (lat, tp)

    #jit_based_benchs()

    two_combinations_measured = collections.OrderedDict()

    for a, b in itertools.combinations_with_replacement(instructions, 2):
        print(a[0], b[0])
        lat, tp = bench.bench_instructions(
            [a[1], b[1]],
            serial_factor = 8, throughput_serial_factor = 8, parallel_factor = 10)
        same_port_metric = ((
            tp-max(instructions_measured[a[0]][1], instructions_measured[b[0]][1])) /
            min(instructions_measured[a[0]][1], instructions_measured[b[0]][1]))
        print("LAT {:.3f} cy, TP {:.3f} cy, SPM {:.2f}".format(lat, tp, same_port_metric))
        two_combinations_measured[a[0], b[0]] = (lat, tp, same_port_metric)

