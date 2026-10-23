# FastUnitfulOperations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lucasvalenzuela.github.io/FastUnitfulOperations.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lucasvalenzuela.github.io/FastUnitfulOperations.jl/dev)
[![Build Status](https://github.com/lucasvalenzuela/FastUnitfulOperations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lucasvalenzuela/FastUnitfulOperations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lucasvalenzuela/FastUnitfulOperations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lucasvalenzuela/FastUnitfulOperations.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

FastUnitfulOperations provides faster matrix multiplication and sorting for arrays of
[Unitful](https://github.com/JuliaPhysics/Unitful.jl) quantities.

Operations on such arrays use generic Julia code instead of the optimized routines available for
plain numbers (e.g., BLAS for matrix multiplication). The fast methods of this package work on
the numeric values to use these routines where possible, and fall back to the regular operations
otherwise.

## Installation

Note that this package usually does not have to be directly installed by the user, instead install any of the
implementing packages.

This package is available through the public registry
[CosmoAnalysisRegistry](https://github.com/lucasvalenzuela/CosmoAnalysisRegistry). Add the registry
once per Julia installation, together with the General registry, which provides the dependencies:
```julia
using Pkg
Pkg.Registry.add("General")
Pkg.Registry.add(url="https://github.com/lucasvalenzuela/CosmoAnalysisRegistry")
```

The package requires Julia 1.12 or newer and can then be installed with:
```julia
Pkg.add("FastUnitfulOperations")
```

## Usage

```julia
using FastUnitfulOperations, Unitful
using SortingAlgorithms

a = rand(1000, 1000)u"m"
b = rand(1000, 1000)u"s"
c = a *ᵘ b                  # same values as a * b, computed with BLAS
uproduct!(c, a, b)          # in-place version

x = rand(10^6)u"m"
usort(x; alg=RadixSort)     # sorting algorithms that require plain numbers
usortperm(x)
```
