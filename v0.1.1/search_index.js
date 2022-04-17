var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = FastUnitfulOperations","category":"page"},{"location":"#FastUnitfulOperations","page":"Home","title":"FastUnitfulOperations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for FastUnitfulOperations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [FastUnitfulOperations]","category":"page"},{"location":"#FastUnitfulOperations.uproduct!-Tuple{Any, Any, Any}","page":"Home","title":"FastUnitfulOperations.uproduct!","text":"uproduct!(c, a, b)\n\nFallback to regular multiplication: uproduct!(c, a, b) = mul!(c, a, b)\n\n\n\n\n\n","category":"method"},{"location":"#FastUnitfulOperations.uproduct!-Union{Tuple{T3}, Tuple{T2}, Tuple{T1}, Tuple{AbstractArray{T1}, AbstractArray{T2}, AbstractArray{T3}}} where {T1<:Unitful.Quantity, T2<:Unitful.Quantity, T3<:Unitful.Quantity}","page":"Home","title":"FastUnitfulOperations.uproduct!","text":"uproduct!(c::AbstractArray, a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1 and/or T2 <:Quantity}\n\nFast in-place matrix multiplication for Unitful arrays.\n\nThis method reinterprets the unitful arrays as arrays with the number's type to make use of optimized matrix multiplication for integers or floats. The method will throw an error if the array contains Unitful elements that do not have a fixed value or unit type.\n\n\n\n\n\n","category":"method"},{"location":"#FastUnitfulOperations.uproduct-Tuple{Any, Any}","page":"Home","title":"FastUnitfulOperations.uproduct","text":"uproduct(a, b)\n*ᵘ(a, b)\n\nFallback to regular multiplication: uproduct(a, b) = a * b\n\n\n\n\n\n","category":"method"},{"location":"#FastUnitfulOperations.uproduct-Union{Tuple{T2}, Tuple{T1}, Tuple{AbstractArray{T1}, AbstractArray{T2}}} where {T1<:Unitful.Quantity, T2<:Unitful.Quantity}","page":"Home","title":"FastUnitfulOperations.uproduct","text":"uproduct(a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1 and/or T2 <:Quantity}\n*ᵘ(a::AbstractArray{T1}, b::AbstractArray{T2}) where {T1 and/or T2 <:Quantity}\n\nFast matrix multiplication for Unitful arrays.\n\nThis method reinterprets the unitful arrays as arrays with the number's type to make use of optimized matrix multiplication for integers or floats. The method will throw an error if the array contains Unitful elements that do not have a fixed value or unit type.\n\nType *\\^u followed by the tab key to input the operator *ᵘ.\n\n\n\n\n\n","category":"method"},{"location":"#FastUnitfulOperations.usort!-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T<:Unitful.Quantity","page":"Home","title":"FastUnitfulOperations.usort!","text":"usort!(a::AbstractArray{T}; kwargs...) where {T<:Quantity}\nusort!(a; kwargs...)\n\nSort a unitful array in-place as a reinterpreted array according to the number's type.\n\nCalls Base.sort! on the reinterpreted array. This enables the use of sorting algorithms that only work for integers or floats (e.g. RadixSort from SortingAlgorithms.\n\nThe method falls back Base.sort! for other array types.\n\n\n\n\n\n","category":"method"},{"location":"#FastUnitfulOperations.usort-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T<:Unitful.Quantity","page":"Home","title":"FastUnitfulOperations.usort","text":"usort(a::AbstractArray{T}; kwargs...) where {T<:Quantity}\nusort(a; kwargs...)\n\nSort a unitful array as a reinterpreted array according to the number's type.\n\nCalls Base.sort on the reinterpreted array. This enables the use of sorting algorithms that only work for integers or floats (e.g. RadixSort from SortingAlgorithms.\n\nThe method falls back Base.sort for other array types.\n\n\n\n\n\n","category":"method"},{"location":"#FastUnitfulOperations.usortperm!-Union{Tuple{T}, Tuple{Any, AbstractArray{T}}} where T<:Unitful.Quantity","page":"Home","title":"FastUnitfulOperations.usortperm!","text":"usortperm!(ix, a::AbstractArray{T}; kwargs...) where {T<:Quantity}\nusortperm!(ix, a; kwargs...)\n\nReturn the permutation vector in the preallocated vector ix to sort a unitful array as a reinterpreted array according to the number's type.\n\nCalls Base.sortperm! on the reinterpreted array. This enables the use of sorting algorithms that only work for integers or floats (e.g. RadixSort from SortingAlgorithms.\n\nThe method falls back Base.sortperm! for other array types.\n\n\n\n\n\n","category":"method"},{"location":"#FastUnitfulOperations.usortperm-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T<:Unitful.Quantity","page":"Home","title":"FastUnitfulOperations.usortperm","text":"usortperm(a::AbstractArray{T}; kwargs...) where {T<:Quantity}\nusortperm(a; kwargs...)\n\nReturn the permutation vector to sort a unitful array as a reinterpreted array according to the number's type.\n\nCalls Base.sortperm on the reinterpreted array. This enables the use of sorting algorithms that only work for integers or floats (e.g. RadixSort from SortingAlgorithms.\n\nThe method falls back Base.sortperm for other array types.\n\n\n\n\n\n","category":"method"}]
}
