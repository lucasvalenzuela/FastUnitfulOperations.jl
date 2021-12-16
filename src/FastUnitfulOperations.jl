module FastUnitfulOperations

using Unitful

export *ᵘ, uproduct, usort, usort!, usortperm, usortperm!

"""
    uproduct(a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1 and/or T2 <:Quantity}
    *ᵘ(a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1 and/or T2 <:Quantity}

Fast matrix multiplication for Unitful arrays.

This method reinterprets the unitful arrays as arrays with the number's type to make use of
optimized matrix multiplication for integers or floats.
The method will throw an error if the array contains Unitful elements that do not have a fixed
value or unit type.

Type `*\\^u` followed by the tab key to input the operator `*ᵘ`.
"""
function uproduct(a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1<:Quantity,T2<:Quantity}
    c = ustrip(a) * ustrip(b)
    u = unit(T1) * unit(T2)
    return reinterpret(Quantity{eltype(c),dimension(u),typeof(u)}, c)
end

function uproduct(a::AbstractArray{T1}, b::AbstractArray) where {T1<:Quantity}
    c = ustrip(a) * b
    return reinterpret(T1, c)
end

function uproduct(a::AbstractArray, b::AbstractArray{T2}) where {T2<:Quantity}
    c = a * ustrip(b)
    return reinterpret(T2, c)
end

"""
    uproduct(a, b)
    *ᵘ(a, b)

Fallback to regular multiplication: `uproduct(a, b) = a * b`
"""
uproduct(a, b) = a * b

*ᵘ = uproduct


"""
    usort(a::AbstractArray{T}; kwargs...) where {T<:Quantity}
    usort(a; kwargs...)

Sort a unitful array as a reinterpreted array according to the number's type.

Calls [`Base.sort`](https://docs.julialang.org/en/v1/base/sort/#Base.sort) on the reinterpreted
array. This enables the use of sorting algorithms that only work for integers or floats
(e.g. RadixSort from
[SortingAlgorithms](https://github.com/JuliaCollections/SortingAlgorithms.jl).

The method falls back `Base.sort` for other array types.
"""
function usort(a::AbstractArray{T}; kwargs...) where {T<:Quantity}
    a_sorted = sort(ustrip(a); kwargs...)
    return reinterpret(T, a_sorted)
end


"""
    usort!(a::AbstractArray{T}; kwargs...) where {T<:Quantity}
    usort!(a; kwargs...)

Sort a unitful array in-place as a reinterpreted array according to the number's type.

Calls [`Base.sort!`](https://docs.julialang.org/en/v1/base/sort/#Base.sort!) on the reinterpreted
array. This enables the use of sorting algorithms that only work for integers or floats
(e.g. RadixSort from
[SortingAlgorithms](https://github.com/JuliaCollections/SortingAlgorithms.jl).

The method falls back `Base.sort!` for other array types.
"""
function usort!(a::AbstractArray{T}; kwargs...) where {T<:Quantity}
    sort!(ustrip(a); kwargs...)
    return a
end

"""
    usortperm(a::AbstractArray{T}; kwargs...) where {T<:Quantity}
    usortperm(a; kwargs...)

Return the permutation vector to sort a unitful array as a reinterpreted array according to the
number's type.

Calls [`Base.sortperm`](https://docs.julialang.org/en/v1/base/sort/#Base.sortperm) on the reinterpreted
array. This enables the use of sorting algorithms that only work for integers or floats
(e.g. RadixSort from
[SortingAlgorithms](https://github.com/JuliaCollections/SortingAlgorithms.jl).

The method falls back `Base.sortperm` for other array types.
"""
function usortperm(a::AbstractArray{T}; kwargs...) where {T<:Quantity}
    return sortperm(ustrip(a); kwargs...)
end


"""
    usortperm!(ix, a::AbstractArray{T}; kwargs...) where {T<:Quantity}
    usortperm!(ix, a; kwargs...)

Return the permutation vector in the preallocated vector `ix` to sort a unitful array as a
reinterpreted array according to the number's type.

Calls [`Base.sortperm!`](https://docs.julialang.org/en/v1/base/sort/#Base.sortperm!) on the reinterpreted
array. This enables the use of sorting algorithms that only work for integers or floats
(e.g. RadixSort from
[SortingAlgorithms](https://github.com/JuliaCollections/SortingAlgorithms.jl).

The method falls back `Base.sortperm!` for other array types.
"""
function usortperm!(ix, a::AbstractArray{T}; kwargs...) where {T<:Quantity}
    return sortperm!(ix, ustrip(a); kwargs...)
end

usort(a; kwargs...) = sort(a; kwargs...)
usort!(a; kwargs...) = sort!(a; kwargs...)
usortperm(a; kwargs...) = sortperm(a; kwargs...)
usortperm!(ix, a; kwargs...) = sortperm!(ix, a; kwargs...)

end
