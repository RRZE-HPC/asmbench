#!/usr/bin/env python3
import collections
import itertools
import socket

import numpy
import matplotlib.pyplot as plt
import matplotlib as mpl

from asmbench import op, bench
from asmbench import oldjit


def jit_based_benchs():
    modules = collections.OrderedDict()
    modules['lea_b'] = (
        oldjit.AddressGenerationBenchmark(
            offset=None,
            base=('r', 'i64', '666'),
            index=None,
            width=None,
            destination='base',
            parallel=1,
            serial=5),
        oldjit.AddressGenerationBenchmark(
            offset=None,
            base=('r', 'i64', '666'),
            index=None,
            width=None,
            destination='base',
            parallel=10,
            serial=1))

    modules['lea_b+off'] = (
        oldjit.AddressGenerationBenchmark(
            offset=('i', None, '23'),
            base=('r', 'i64', '666'),
            index=None,
            width=None,
            destination='base',
            parallel=1,
            serial=5),
        oldjit.AddressGenerationBenchmark(
            offset=('i', None, '23'),
            base=('r', 'i64', '666'),
            index=None,
            width=None,
            destination='base',
            parallel=10,
            serial=1))

    modules['lea_idx*w'] = (
        oldjit.AddressGenerationBenchmark(
            offset=None,
            base=None,
            index=('r', 'i64', '1'),
            width=('i', None, '4'),
            destination='index',
            parallel=1,
            serial=5),
        oldjit.AddressGenerationBenchmark(
            offset=None,
            base=None,
            index=('r', 'i64', '1'),
            width=('i', None, '4'),
            destination='index',
            parallel=10,
            serial=1))

    modules['lea_off+idx*w'] = (
        oldjit.AddressGenerationBenchmark(
            offset=('i', 'i64', '-0x8'),
            base=None,
            index=('r', 'i64', '51'),
            width=('i', None, '4'),
            destination='index',
            parallel=1,
            serial=5),
        oldjit.AddressGenerationBenchmark(
            offset=('i', 'i64', '-0x8'),
            base=None,
            index=('r', 'i64', '51'),
            width=('i', None, '4'),
            destination='index',
            parallel=10,
            serial=1))

    modules['lea_b+idx*w'] = (
        oldjit.AddressGenerationBenchmark(
            offset=None,
            base=('r', 'i64', '23'),
            index=('r', 'i64', '12'),
            width=('i', None, '4'),
            destination='base',
            parallel=1,
            serial=5),
        oldjit.AddressGenerationBenchmark(
            offset=None,
            base=('r', 'i64', '23'),
            index=('r', 'i64', '12'),
            width=('i', None, '4'),
            destination='base',
            parallel=10,
            serial=1))

    modules['lea_b+off+idx*w'] = (
        oldjit.AddressGenerationBenchmark(
            offset=('i', None, '42'),
            base=('r', 'i64', '23'),
            index=('r', 'i64', '12'),
            width=('i', None, '4'),
            destination='base',
            parallel=1,
            serial=5),
        oldjit.AddressGenerationBenchmark(
            offset=('i', None, '42'),
            base=('r', 'i64', '23'),
            index=('r', 'i64', '12'),
            width=('i', None, '4'),
            destination='base',
            parallel=10,
            serial=1))

    modules['LD_linear'] = (
        oldjit.LoadBenchmark(
            chain_length=2048,  # 2048 * 8B = 16kB
            structure='linear',
            parallel=1,
            serial=2),
        oldjit.LoadBenchmark(
            chain_length=2048,  # 2048 * 8B = 16kB
            structure='linear',
            parallel=4,
            serial=2))

    modules['LD_random'] = (
        oldjit.LoadBenchmark(
            chain_length=2048,  # 2048 * 8B = 16kB
            structure='random',
            parallel=1,
            serial=2),
        oldjit.LoadBenchmark(
            chain_length=2048,  # 2048 * 8B = 16kB
            structure='random',
            parallel=4,
            serial=2))

    for name, mods in modules.items():
        lat_module, tp_module = mods
        r_lat = lat_module.build_and_execute(repeat=3)
        cy_per_it_lat = min(r_lat['runtimes']) * r_lat['frequency'] / (
                    r_lat['iterations'] * lat_module.parallel * lat_module.serial)
        r_tp = tp_module.build_and_execute(repeat=3)
        cy_per_it_tp = min(r_tp['runtimes']) * r_tp['frequency'] / (
                    r_tp['iterations'] * tp_module.parallel * tp_module.serial)
        print('{key:<16} LAT {cy_per_it_lat:.3f} cy  TP {cy_per_it_tp:.3f} cy'.format(
            key=name,
            cy_per_it_lat=cy_per_it_lat,
            cy_per_it_tp=cy_per_it_tp))

def plot_combined(single_measured, combined_measured):
    instructions = list(single_measured.keys())
    d = numpy.ndarray((len(single_measured), len(single_measured)))
    d.fill(float('nan'))
    for k, v in combined_measured.items():
        i1, i2 = [instructions.index(i) for i in [c[0] for c in k]]
        d[i1, i2] = v[2]
    cmap = mpl.cm.get_cmap('plasma', 5)
    cmap.set_bad('w') # default value is 'k'
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    cax = ax1.imshow(d, interpolation="nearest", cmap=cmap, norm=mpl.colors.Normalize(vmin=-.5, vmax=1.5))
    ax1.set_xticks(range(len(instructions)))
    ax1.set_xticklabels(instructions, rotation=90)
    ax1.set_yticks(range(len(instructions)))
    ax1.set_yticklabels(instructions)
    ax1.set_title(socket.gethostname())
    ax1.grid()
    cb = fig.colorbar(cax, shrink=0.65)
    cb.set_ticks([-.5, 0, 1, 1.5])
    cb.set_ticklabels(['< -0.5', '0.0 (complete overlap)', '1.0 (no overlap)', '> 1.5'])
    cb.set_label('inverse parallel overlap')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    bench.setup_llvm()
    instructions = [
        (i[0], i[1], op.Instruction.from_string(i[1]))
        for i in [
            ('ADD32ri', 'add {src:i32:1}, {srcdst:i32:r}'),
            ('ADD64ri32', 'add {src:i32:1}, {srcdst:i64:r}'),
            ('INC64r', 'inc {srcdst:i64:r}'),
            ('SUB32ri', 'sub {src:i32:1}, {srcdst:i64:r}'),
            ('MOV64ri32', 'mov {src:i32:1}, {srcdst:i64:r}'),
            ('VINSERTF128rr', 'vinsertf128 {src:i8:0}, {src:<2 x double>:x}, {src:<4 x double>:x}, {dst:<4 x double>:x}'),
            ('VCVTSI642SSrr', 'vcvtsi2ss {src:i64:r}, {src:float:x}, {dst:float:x}'),
            ('VADDPDYrr', 'vaddpd {src:<4 x double>:x}, {src:<4 x double>:x}, {dst:<4 x double>:x}'),
            ('VADDSDrr', 'vaddsd {src:double:x}, {src:double:x}, {dst:double:x}'),
            ('VADDSSrr', 'vaddss {src:float:x}, {src:float:x}, {dst:float:x}'),
            ('VFMADD213PDYr', 'vfmadd213pd {src:<4 x double>:x}, {src:<4 x double>:x}, {srcdst:<4 x double>:x}'),
            ('VFMADD213PDr', 'vfmadd213pd {src:<2 x double>:x}, {src:<2 x double>:x}, {srcdst:<2 x double>:x}'),
            ('VFMADD213PSYr', 'vfmadd213ps {src:<4 x double>:x}, {src:<4 x double>:x}, {srcdst:<4 x double>:x}'),
            ('VFMADD213PSr', 'vfmadd213ps {src:<2 x double>:x}, {src:<2 x double>:x}, {srcdst:<2 x double>:x}'),
            ('VFMADD213SDr', 'vfmadd213sd {src:double:x}, {src:double:x}, {srcdst:double:x}'),
            ('VFMADD213SSr', 'vfmadd213ss {src:float:x}, {src:float:x}, {srcdst:float:x}'),
            ('VMULPDYrr', 'vmulpd {src:<4 x double>:x}, {src:<4 x double>:x}, {dst:<4 x double>:x}'),
            ('VMULSDrr', 'vmulsd {src:double:x}, {src:double:x}, {dst:double:x}'),
            ('VMULSSrr', 'vmulss {src:float:x}, {src:float:x}, {dst:float:x}'),
            ('VSUBSDrr', 'vsubsd {src:double:x}, {src:double:x}, {dst:double:x}'),
            ('VSUBSSrr', 'vsubss {src:float:x}, {src:float:x}, {dst:float:x}'),
            ('VDIVPDYrr', 'vdivpd {src:<4 x double>:x}, {src:<4 x double>:x}, {dst:<4 x double>:x}'),
            ('VDIVSDrr', 'vdivsd {src:double:x}, {src:double:x}, {dst:double:x}'),
            ('VDIVSSrr', 'vdivss {src:float:x}, {src:float:x}, {dst:float:x}'),
            ]
    ]
    instructions_measured = collections.OrderedDict()
    for llvm_name, i_str, i in instructions:
        lat, tp = bench.bench_instructions(
            [i],
            serial_factor=8, throughput_serial_factor=8, parallel_factor=10,
            verbosity=0, repeat=10, min_elapsed=0.3, max_elapsed=0.5)
        print('{:<16}  LAT {:.3f} cy  TP {:.3f} cy'.format(llvm_name, lat, tp))
        instructions_measured[llvm_name] = (lat, tp)

    jit_based_benchs()

    two_combinations_measured = collections.OrderedDict()

    for a, b in itertools.combinations_with_replacement(instructions, 2):
        lat, tp = bench.bench_instructions(
            [a[2], b[2]],
            serial_factor=8, throughput_serial_factor=8, parallel_factor=10,
            verbosity=0, repeat=10, min_elapsed=0.3, max_elapsed=0.5)
        same_port_metric = ((
            tp-max(instructions_measured[a[0]][1], instructions_measured[b[0]][1])) /
            min(instructions_measured[a[0]][1], instructions_measured[b[0]][1]))
        print('{:<16} {:<16}  LAT {:.3f} cy  TP {:.3f} cy  SPM {:>5.2f}'.format(
            a[0], b[0], lat, tp, same_port_metric))
        two_combinations_measured[(a[0], a[1]), (b[0], b[1])] = (lat, tp, same_port_metric)

    plot_combined(instructions_measured, two_combinations_measured)