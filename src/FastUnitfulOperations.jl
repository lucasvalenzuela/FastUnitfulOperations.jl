"""
    FastUnitfulOperations

Fast matrix multiplication and sorting for arrays of Unitful quantities.

Arrays of quantities are handled by generic Julia code, while arrays of plain numbers can use
optimized routines such as BLAS. The fast methods of this package reinterpret a quantity array
as an array of its numeric values without copying it, apply the routine for plain numbers, and
attach the units to the result. Arguments that the fast methods do not support fall back to the
regular operations from Base.
"""
module FastUnitfulOperations

using LinearAlgebra
using Unitful

export *ᵘ, uproduct, uproduct!, usort, usort!, usortperm, usortperm!
public convertoperands, isfastarray

# Array types handled by the fast methods: `Array`s and views, reinterpretations, adjoints, and
# transposes of them. Other array types use the methods from Base, which may be specialized
# for them (e.g., sparse matrices or ranges).
_isfastarraytype(::Array) = true
_isfastarraytype(a::Union{SubArray,Base.ReinterpretArray,Adjoint,Transpose}) = _isfastarraytype(parent(a))
_isfastarraytype(::AbstractArray) = false

"""
    FastUnitfulOperations.isfastarray(x)

Return whether the fast methods of this package can operate on `x` directly.

This is the case if `x` is an `Array`, or a view, `reinterpret`, `adjoint`, or `transpose` of one
(also nested), and its element type is a concrete numeric type with a fixed size in memory
(`isbitstype`). For unitful arrays, this means that all elements share one numeric type and
unit.

Methods of [`FastUnitfulOperations.convertoperands`](@ref) can use this to check whether an
operand they do not convert is supported by the fast methods. To support further array types,
add methods to `convertoperands` instead of `isfastarray`.

# Examples

```jldoctest
julia> using Unitful

julia> FastUnitfulOperations.isfastarray([1.0, 2.0]u"m")
true

julia> FastUnitfulOperations.isfastarray(view([1.0 2.0; 3.0 4.0]u"m", :, 1)')
true

julia> FastUnitfulOperations.isfastarray(Quantity[1.0u"m", 2.0u"cm"])
false

julia> FastUnitfulOperations.isfastarray((1:3)u"m")
false
```
"""
isfastarray(x::AbstractArray{T}) where {T<:Number} = isbitstype(T) && _isfastarraytype(x)
isfastarray(x) = false

# View of the numeric values of a quantity array that shares memory with `a`, so that in-place
# operations on the result modify `a`. `ustrip` cannot be used for this: it copies any input
# that is not an `Array`. Requires `isfastarray(a)`.
_strip(a::AbstractArray{T}) where {T<:Quantity} = reinterpret(Unitful.numtype(T), a)
_strip(a::AbstractArray) = a
# Keeping the wrapper outside lets LinearAlgebra recognize transposed operands, e.g., to use BLAS
# or to return a scalar for the product of an adjoint vector and a vector.
_strip(a::Adjoint{T}) where {T<:Quantity} = adjoint(_strip(parent(a)))
_strip(a::Transpose{T}) where {T<:Quantity} = transpose(_strip(parent(a)))

_unit(::Type{T}) where {T<:Quantity} = unit(T)
_unit(::Type) = NoUnits

_showunit(u) = u == NoUnits ? "no unit" : string(u)

# Element type of a product with numeric type `T` and unit `u`; unitless products are plain
# numbers, as for `*`.
_quantitytype(::Type{T}, u) where {T} = u == NoUnits ? T : Quantity{T,dimension(u),typeof(u)}

# Element type of a matrix product, determined as in LinearAlgebra.
_matprod(x, y) = x * y + x * y
_prodtype(::Type{Ta}, ::Type{Tb}) where {Ta,Tb} = Base.promote_op(_matprod, Ta, Tb)

_isadjvec(x) = x isa Union{Adjoint{<:Any,<:AbstractVector},Transpose{<:Any,<:AbstractVector}}

"""
    FastUnitfulOperations.convertoperands(a, b)

Return the operands that [`uproduct`](@ref) and [`uproduct!`](@ref) pass to their fast methods
in place of `a` and `b`. By default, `a` and `b` are returned unchanged.

Add methods to enable the fast methods for array types they do not support, by converting such
operands to an `Array` (or another array type for which [`FastUnitfulOperations.isfastarray`](@ref)
returns `true`). Only convert operands for which the product of the converted operands has the
same type as `a * b`.
If the fast methods cannot be used even with the converted operands, `uproduct` and `uproduct!`
call `*` and `mul!` with the original operands.

With StaticArraysCore loaded (e.g., through StaticArrays or Rotations), a static matrix
multiplied with a matrix of a supported array type is converted to a `Matrix`.

# Examples

A method for a hypothetical matrix type `MyMatrix`, for which `MyMatrix * Matrix` returns a
`Matrix`:

```julia
FastUnitfulOperations.convertoperands(a::MyMatrix, b::Matrix) = (Matrix(a), b)
```
"""
convertoperands(a, b) = (a, b)

# Product of the numeric arrays `astrip` and `bstrip` with unit `u`, in an array of the type `*`
# returns.
function _numericproduct(astrip, bstrip, u)
    # An adjoint or transposed vector is an `AbstractMatrix`, but `*` returns a scalar or a row
    # vector for it, so its result type is left to LinearAlgebra below.
    if astrip isa AbstractMatrix && bstrip isa AbstractVecOrMat && !_isadjvec(astrip)
        # Write the product directly into the memory of the result
        T = _quantitytype(_prodtype(eltype(astrip), eltype(bstrip)), u)
        c = similar(bstrip, T, (axes(astrip, 1), Base.tail(axes(bstrip))...))
        mul!(_strip(c), astrip, bstrip)
        return c
    end

    cstrip = astrip * bstrip
    cstrip isa Number && return cstrip * u

    c = similar(cstrip, _quantitytype(eltype(cstrip), u))
    copyto!(_strip(c), cstrip)
    return c
end

# Shared implementation of the `uproduct` methods: the fast method if possible, `a * b` otherwise.
function _uproduct(a, b)
    ac, bc = convertoperands(a, b)
    isfastarray(ac) && isfastarray(bc) || return a * b
    return _numericproduct(_strip(ac), _strip(bc), _unit(eltype(a)) * _unit(eltype(b)))
end

"""
    uproduct(a, b)
    a *ᵘ b

Fast matrix multiplication for Unitful arrays.

This method multiplies the numeric values with the routines for plain numbers (e.g., BLAS) and
attaches the units to the result. The result has the same type as `a * b`.
The method falls back to `a * b` if an argument is not a
[supported array](@ref "Supported arrays").

Type `*\\^u` followed by the tab key to input the operator `*ᵘ`.

# Examples

```jldoctest
julia> using Unitful

julia> a = [1.0 2.0; 3.0 4.0]u"m"; b = [1.0, 2.0]u"s";

julia> a *ᵘ b
2-element Vector{Quantity{Float64, 𝐋 𝐓, Unitful.FreeUnits{(m, s), 𝐋 𝐓, nothing}}}:
  5.0 m s
 11.0 m s
```
"""
uproduct(a::AbstractArray{<:Quantity}, b::AbstractArray{<:Quantity}) = _uproduct(a, b)
uproduct(a::AbstractArray{<:Quantity}, b::AbstractArray) = _uproduct(a, b)
uproduct(a::AbstractArray, b::AbstractArray{<:Quantity}) = _uproduct(a, b)
uproduct(a, b) = a * b

const *ᵘ = uproduct

"""
    uproduct!(c, a, b)

Fast in-place matrix multiplication for Unitful arrays, storing `a * b` in `c` and returning `c`.

This method multiplies the numeric values with the routines for plain numbers (e.g., BLAS) and
stores them in `c`. It throws an error if the unit of `c` differs from the product of the units
of `a` and `b` (a unitless `c` requires `a * b` to be unitless).
The method falls back to `mul!(c, a, b)`, which converts units as needed, if an argument is not a
[supported array](@ref "Supported arrays").

# Examples

```jldoctest
julia> using Unitful

julia> a = [1.0 2.0; 3.0 4.0]u"m"; b = [1.0, 2.0]u"s"; c = zeros(2)u"m*s";

julia> uproduct!(c, a, b)
2-element Vector{Quantity{Float64, 𝐋 𝐓, Unitful.FreeUnits{(m, s), 𝐋 𝐓, nothing}}}:
  5.0 m s
 11.0 m s
```
"""
function uproduct!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    ac, bc = convertoperands(a, b)
    isfastarray(c) && isfastarray(ac) && isfastarray(bc) || return mul!(c, a, b)
    uc = _unit(eltype(c))
    uab = _unit(eltype(a)) * _unit(eltype(b))
    uc == uab || throw(
        ArgumentError(
            "unit of c ($(_showunit(uc))) does not match the product of the units of a and b ($(_showunit(uab)))",
        ),
    )
    cstrip, astrip, bstrip = _strip(c), _strip(ac), _strip(bc)
    mul!(cstrip, astrip, bstrip)
    return c
end

uproduct!(c, a, b) = mul!(c, a, b)




# Sorting the numeric values is only equivalent to sorting the quantities when the ordering does
# not depend on a user-supplied function, since such a function expects quantities.
_fastsortable(a, kwargs) =
    isfastarray(a) && !haskey(kwargs, :by) && !haskey(kwargs, :lt) && !haskey(kwargs, :order)

"""
    usort(a; kwargs...)

Return a sorted copy of the unitful array `a`, like
[`sort`](https://docs.julialang.org/en/v1/base/sort/#Base.sort).

The elements are sorted by their numeric values, which is faster than sorting the quantities and
allows sorting algorithms that only accept integers or floats (e.g., RadixSort from
[SortingAlgorithms](https://github.com/JuliaCollections/SortingAlgorithms.jl)).
The result has the same type as `sort(a)`.
The method falls back to `sort(a; kwargs...)` if `a` is not a
[supported array](@ref "Supported arrays") or if any of the keyword arguments `by`, `lt`, or
`order` is given, since these act on quantities.

# Examples

```jldoctest
julia> using Unitful

julia> usort([3.0, 1.0, 2.0]u"m")
3-element Vector{Quantity{Float64, 𝐋, Unitful.FreeUnits{(m,), 𝐋, nothing}}}:
 1.0 m
 2.0 m
 3.0 m
```
"""
function usort(a::AbstractArray{<:Quantity}; kwargs...)
    _fastsortable(a, kwargs) || return sort(a; kwargs...)
    return usort!(copyto!(similar(a), a); kwargs...)
end

usort(a; kwargs...) = sort(a; kwargs...)

"""
    usort!(a; kwargs...)

Sort the unitful array `a` in place and return it, like
[`sort!`](https://docs.julialang.org/en/v1/base/sort/#Base.sort!). This is the in-place version
of [`usort`](@ref), with the same fallbacks.
"""
function usort!(a::AbstractArray{<:Quantity}; kwargs...)
    _fastsortable(a, kwargs) || return sort!(a; kwargs...)
    sort!(_strip(a); kwargs...)
    return a
end

usort!(a; kwargs...) = sort!(a; kwargs...)

"""
    usortperm(a; kwargs...)

Return the permutation that sorts the unitful array `a`, like
[`sortperm`](https://docs.julialang.org/en/v1/base/sort/#Base.sortperm). The permutation is
computed from the numeric values as in [`usort`](@ref), with the same fallbacks.

# Examples

```jldoctest
julia> using Unitful

julia> usortperm([3.0, 1.0, 2.0]u"m")
3-element Vector{Int64}:
 2
 3
 1
```
"""
function usortperm(a::AbstractArray{<:Quantity}; kwargs...)
    _fastsortable(a, kwargs) || return sortperm(a; kwargs...)
    return sortperm(_strip(a); kwargs...)
end

usortperm(a; kwargs...) = sortperm(a; kwargs...)

"""
    usortperm!(ix, a; kwargs...)

Store the permutation that sorts the unitful array `a` in `ix` and return `ix`, like
[`sortperm!`](https://docs.julialang.org/en/v1/base/sort/#Base.sortperm!). This is the in-place
version of [`usortperm`](@ref), with the same fallbacks.
"""
function usortperm!(ix, a::AbstractArray{<:Quantity}; kwargs...)
    _fastsortable(a, kwargs) || return sortperm!(ix, a; kwargs...)
    return sortperm!(ix, _strip(a); kwargs...)
end

usortperm!(ix, a; kwargs...) = sortperm!(ix, a; kwargs...)

end
