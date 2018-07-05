#!/usr/bin/env python3
import argparse

from . import op, bench


def main():
    parser = argparse.ArgumentParser(description='Assembly Instruction Benchmark Toolkit')
    # parser.add_argument('mode', metavar='MODE', type=str, choices=['latency', 'throughput'])
    parser.add_argument('instructions', metavar='INSTR', type=op.Instruction.from_string, nargs='+',
                        help='instruction declaration, e.g., "add {src:i32:r} {srcdst:i32:r}"')
    parser.add_argument('--serialize', action='store_true',
                        help='Serialize instructions.')
    parser.add_argument('--latency-serial', '-l', type=int, default=8,
                         help='length of serial chain for each instruction in latency benchmark')
    parser.add_argument('--parallel', '-p',type=int, default=10,
                        help='number of parallel instances of serial chains in throughput '
                             'benchmark')
    parser.add_argument('--throughput-serial', '-t', type=int, default=8,
                        help='length of serial instances of serial chains in throughput benchmark')
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="increase output verbosity")
    args = parser.parse_args()

    bench.setup_llvm()
    lat, tp = bench.bench_instructions(args.instructions,
                                       serial_factor=args.latency_serial,
                                       parallel_factor=args.parallel,
                                       throughput_serial_factor=args.throughput_serial,
                                       serialize=args.serialize,
                                       verbosity=args.verbose)
    print("Latency: {:.2f} cycle\nThroughput: {:.2f} cycle\n".format(lat, tp))

    #b = bench.IntegerLoopBenchmark(args.instructions[0])
    #b.get_iaca_analysis()


if __name__ == "__main__":
    main()
