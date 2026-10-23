module FastUnitfulOperationsStaticArraysCoreExt

using FastUnitfulOperations: FastUnitfulOperations, isfastarray, _isadjvec
using StaticArraysCore: StaticMatrix

# A static matrix multiplied with a matrix supported by the fast methods is converted to a
# `Matrix`, which lets BLAS compute the product. `*` returns a `Matrix` for these operands as
# well. With a vector or another static array, `*` returns a static array, so the operands are
# not converted.
_isfastmatrix(x) = x isa AbstractMatrix && !_isadjvec(x) && isfastarray(x)

FastUnitfulOperations.convertoperands(a::StaticMatrix, b::AbstractArray) =
    _isfastmatrix(b) ? (Matrix(a), b) : (a, b)
FastUnitfulOperations.convertoperands(a::AbstractArray, b::StaticMatrix) =
    _isfastmatrix(a) ? (a, Matrix(b)) : (a, b)
FastUnitfulOperations.convertoperands(a::StaticMatrix, b::StaticMatrix) = (a, b)

end
