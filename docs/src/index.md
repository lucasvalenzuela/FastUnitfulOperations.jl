```@meta
CurrentModule = FastUnitfulOperations
```

# FastUnitfulOperations

[FastUnitfulOperations](https://github.com/lucasvalenzuela/FastUnitfulOperations.jl) provides
faster matrix multiplication and sorting for arrays of
[Unitful](https://github.com/JuliaPhysics/Unitful.jl) quantities.

Operations on such arrays use generic Julia code instead of the optimized routines available for
plain numbers (e.g., BLAS for matrix multiplication). The fast methods of this package work on
the numeric values to use these routines where possible, and fall back to the regular operations
otherwise.

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

## Supported arrays

The fast methods operate on `Array`s and on views, `reinterpret`s, `adjoint`s, and
`transpose`s of them, whose elements share one numeric type and unit (e.g., all `Float64` in
`m`). Other arguments, such as sparse matrices, ranges, or `Quantity[1.0u"m", 2.0u"cm"]`, fall back
to the regular operations from Base. To use the fast methods with other array types,
convert them first, e.g., `uproduct(Array(a), b)`.

## Performance

The fast methods avoid the generic code that Julia uses for arrays of quantities. The
timings below are from an example run, after a first call of each function to compile it, and
depend on the machine.

```jldoctest performance; filter = r"\d+\.\d+(e-\d+)?"
julia> using FastUnitfulOperations, Unitful

julia> a = rand(300, 300)u"m"; b = rand(300, 300)u"s"; x = rand(10^6)u"m";

julia> a * b; a *ᵘ b; sort(x); usort(x);

julia> @elapsed a * b
0.0146

julia> @elapsed a *ᵘ b
0.00069

julia> @elapsed sort(x)
0.0352

julia> @elapsed usort(x)
0.0179
```

## Reference

```@autodocs
Modules = [FastUnitfulOperations]
Filter = f -> f ∉ (FastUnitfulOperations.convertoperands, FastUnitfulOperations.isfastarray)
```

## Developer documentation

Functions for packages that extend the fast methods to further array types.

```@docs
FastUnitfulOperations.convertoperands
FastUnitfulOperations.isfastarray
```
