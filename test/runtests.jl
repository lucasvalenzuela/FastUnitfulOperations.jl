using FastUnitfulOperations
using Test

using SortingAlgorithms
using Unitful

@testset "FastUnitfulOperations.jl" begin
    @testset "Product" begin
        a = rand(200, 3)
        b = rand(3, 24)
        au = a * u"m"
        bu = b * u"s"

        @test uproduct(au, bu) == a * b * u"m*s"
        @test uproduct(a, bu) == a * b * u"s"
        @test uproduct(au, b) == a * b * u"m"
        @test uproduct(a, b) == a * b

        @test au *ᵘ bu == a * b * u"m*s"
    end

    @testset "Sorting" begin
        a = rand(1000)
        au = a * u"m"

        @test usort(au) |> issorted
        @test usort(au; alg=RadixSort) |> issorted
        @test usort(a; alg=QuickSort) |> issorted

        auc = copy(au)
        usort!(auc)
        @test auc |> issorted

        auc = copy(au)
        usort!(auc; alg=RadixSort)
        @test auc |> issorted

        ac = copy(a)
        usort!(ac; alg=QuickSort)
        @test ac |> issorted

        @test au[usortperm(au)] |> issorted
        @test au[usortperm(au; alg=RadixSort)] |> issorted
        @test a[usortperm(a; alg=QuickSort)] |> issorted

        ix = Vector{Int}(undef, length(au))
        usortperm!(ix, au)
        @test au[ix] |> issorted

        ix = collect(1:length(au))
        usortperm!(ix, au; alg=RadixSort, initialized=true)
        @test au[ix] |> issorted

        ix = collect(1:length(a))
        usortperm!(ix, a; alg=QuickSort, initialized=true)
        @test a[ix] |> issorted
    end
end
