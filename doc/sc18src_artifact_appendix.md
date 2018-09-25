# Artifact Description: OoO Instruction Benchmarking Framework on the Back of Dragons

* Julian Hammer, RRZE University of Erlangen-Nürnberg, julian.hammer@fau.de, +49 9131 85 20101
* Georg Hager (advisor), RRZE University of Erlangen-Nürnberg, georg.hager@fau.de
* Gerhard Wellein (advisor), RRZE University of Erlangen-Nürnberg, gerhard.wellein@fau.de

## A.1 Abstract

In order to construct an accurate instruction execution model for modern out-of-order micro architectures, an accurate description of instruction latency and throughput, as well as resource conflicts is indispensable. Already existing resources and vendor provided information is neither complete nor detailed enough and sometimes faulty. We therefore proclaim to deduct this information through runtime instruction benchmarking of single and composite instructions, and present a framework to support such investigations based on LLVM's just-in-time and cross-platform compilation capabilities.

`asmbench` abstracts instructions, registers, immediates, memory operands and dependency chains, to easily construct benchmarks. The synthesized code is interactively compiled and executed using the `llvmlite` library, which in turn is based on the stable LLVM C-API. `asmbench` offers a command line as well as a programming interface.

Unlike other approaches, we do not rely on model specific performance counters and focus on interoperability and automation to support quick modeling of many microarchitectures.

## A.2 Description

### A.2.1 Check-list (artifact meta information)

- Compilation: llvm jit
- Binary: scripts
- Hardware: intel skylake, amd zen
- Publicly available?: yes

### A.2.2 How software can be obtained (if available)

Check out https://github.com/RRZE-HPC/asmbench

### A.2.3 Hardware dependencies

We ran on an AMD EPYC 7451 (Zen architecture) at 1.8 GHz (fixed, turbo disabled) and Intel I7-6700HQ (Skylake SP architecture) at 2.2 GHz (fixed, turbo disabled). The results should be reproducible on any Zen and Skylake SP processors.

### A.2.4 Software dependencies

To obtain our results and plots, we require the following software dependencies to be installed:

* Python >=3.5 with the following libraries:
    * pip (for easier installation)
    * llvmlite>=0.23.2
    * psutil
    * numpy (for plotting)
    * matplotlib (for plotting)
* libllvm 6.0 (required by llvmlite)

### A.2.5 Datasets

None required, all included.

## A.3 Installation

To install `asmbench` in the correct version and all its dependencies into the users home directory, execute: `pip3 install --user asmbench[sc18src]==0.1.2`.

Alternatively clone https://github.com/RRZE-HPC/asmbench with commit hash 515b28cb4e44426239e6161dc3a79d888a9e0e21 and install using included `setup.py`.

## A.4 Experiment workflow

1. Fix frequency, e.g., using likwid: `likwid-setFrequencies -f <FREQ>`. `<FREQ>` should be the base clock for the specific model used.
2. Disable turbo mode, e.g., using likwid: `likwid-setFrequencies -t 0`.
3. Run `asmbench.sc18src` module: `python3 -m asmbench.sc18src`.

## A.5 Evaluation and expected result

Compare results to presented data on poster. If frequency was correctly fixed, difference of single instruction measurements should be less than 10%. Due to accumulating errors in combined measurements, results presented in the resource conflict plots should qualitatively match ours.

## A.6 Experiment customization

None

## A.7 Notes

None
