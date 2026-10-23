using FastUnitfulOperations
using Test
using Aqua
using ExplicitImports
using JET

using OffsetArrays
using Rotations
using SortingAlgorithms
using SparseArrays
using StaticArrays
using Unitful

@testset "FastUnitfulOperations.jl" begin
    @testset "Code quality" begin
        Aqua.test_all(FastUnitfulOperations)
        @test check_no_stale_explicit_imports(FastUnitfulOperations) === nothing
        @test check_no_self_qualified_accesses(FastUnitfulOperations) === nothing
        if JET.JET_AVAILABLE
            JET.test_package(FastUnitfulOperations; target_modules=(FastUnitfulOperations,))
        end
    end

    @testset "uproduct" begin
        a = rand(200, 3)
        b = rand(3, 24)
        au = a * u"m"
        bu = b * u"s"

        @testset "Values" begin
            @test uproduct(au, bu) == a * b * u"m*s"
            @test uproduct(a, bu) == a * b * u"s"
            @test uproduct(au, b) == a * b * u"m"
            @test uproduct(a, b) == a * b
            @test au*ᵘbu == a * b * u"m*s"
            @test uproduct(2, 3) == 6
            @test uproduct(2u"m", 3u"s") == 6u"m*s"
        end

        @testset "Result has the type of `*`" begin
            v = rand(3)
            vu = v * u"kg"
            wu = rand(200) * u"kg"
            ai = [1 2; 3 4] * u"m"
            bf = [0.5 0.0; 0.0 0.5]
            a32 = Float32[1 2; 3 4] * u"m"
            b32 = Float32[1 0; 0 2] * u"s"
            for (x, y) in (
                # Shapes
                (au, bu),
                (a, bu),
                (au, b),
                (au, vu),
                (au, v),
                (a, vu),
                (vu, vu'),
                (vu, transpose(v)),
                (vu', vu),
                (transpose(v), vu),
                (vu', bu),
                (wu', au),
                (au', au),
                (transpose(au), a),
                (a', au),
                (view(au, 1:10, :), view(bu, :, 1:4)),
                # Unitless products
                (au, bu * u"m^-1*s^-1"),
                (vu', v * u"kg^-1"),
                # Differing numeric types
                (ai, bf),
                (bf, ai),
                (a32, bf),
                (bf, a32),
                (ai, b32),
            )
                @test typeof(uproduct(x, y)) == typeof(x * y)
                @test uproduct(x, y) ≈ x * y
            end
        end

        @testset "Fallbacks" begin
            sp = sprand(3, 5, 0.5)
            spu = sp * u"s"
            anyb = Any[1.0 2.0; 3.0 4.0; 5.0 6.0]
            mixed = Quantity[3.0u"m", 2.0u"cm", 1.0u"m"]
            row = [1.0 2.0]
            rowu = row * u"s"
            for (x, y) in (
                # Unsupported array types
                (au, sp),
                (au, spu),
                (permutedims(sp), permutedims(au)),
                (sparse(au), spu),
                # Non-numeric element type
                (au, anyb),
                # Mixed units
                (mixed, row),
                (row', permutedims(mixed)),
                (mixed, rowu),
                (rowu', permutedims(mixed)),
            )
                @test typeof(uproduct(x, y)) == typeof(x * y)
                @test uproduct(x, y) == x * y
            end
        end
    end

    @testset "uproduct!" begin
        a = rand(4, 3)
        b = rand(3, 5)
        au = a * u"m"
        bu = b * u"s"

        @testset "Values" begin
            for (x, y, u) in ((au, bu, u"m*s"), (au, b, u"m"), (a, bu, u"s"), (a, b, NoUnits))
                c = zeros(4, 5) * u
                @test uproduct!(c, x, y) === c
                @test c ≈ x * y
            end

            # Product of reciprocal units is unitless
            c = zeros(4, 5)
            uproduct!(c, au, b * u"m^-1")
            @test c ≈ a * b

            # Differing numeric types of c, a, and b
            ai = [1 2; 3 4] * u"m"
            b32 = Float32[1 0; 0 2] * u"s"
            c = zeros(2, 2) * u"m*s"
            uproduct!(c, ai, b32)
            @test c == ai * b32
        end

        @testset "Wrapped arguments and destinations" begin
            atu = permutedims(au)
            btu = permutedims(bu)
            for (x, y) in (
                (atu', bu),
                (au, btu'),
                (atu', btu'),
                (transpose(atu), transpose(btu)),
                (reinterpret(eltype(au), a), bu),
                (au, reinterpret(eltype(bu), b)),
                (view(au, [4, 2, 1, 3], :), bu),
                (view(atu, :, :)', bu),
            )
                c = zeros(4, 5) * u"m*s"
                @test uproduct!(c, x, y) === c
                @test c ≈ x * y
            end

            cbig = zeros(6, 7) * u"m*s"
            uproduct!(view(cbig, 2:5, 3:7), au, bu)
            @test cbig[2:5, 3:7] ≈ au * bu
            @test all(iszero, cbig[1, :]) && all(iszero, cbig[6, :]) && all(iszero, cbig[:, 1:2])

            cr = reinterpret(eltype(au * bu), zeros(4, 5))
            @test uproduct!(cr, au, bu) === cr
            @test cr ≈ au * bu

            for wrap in (adjoint, transpose)
                cparent = zeros(5, 4) * u"m*s"
                uproduct!(wrap(cparent), au, bu)
                @test cparent ≈ permutedims(au * bu)
            end
        end

        @testset "Unit mismatch" begin
            @test_throws "unit of c (kg) does not match the product of the units of a and b (m s)" uproduct!(
                zeros(4, 5) * u"kg",
                au,
                bu,
            )
            @test_throws "unit of c (no unit) does not match the product of the units of a and b (m s)" uproduct!(
                zeros(4, 5),
                au,
                bu,
            )
            @test_throws "unit of c (m s) does not match the product of the units of a and b (m)" uproduct!(
                zeros(4, 5) * u"m*s",
                au,
                b,
            )
        end

        @testset "Fallbacks" begin
            # Unsupported array type
            spu = sprand(3, 5, 0.5) * u"s"
            c = zeros(4, 5) * u"m*s"
            @test uproduct!(c, au, spu) === c
            @test c ≈ au * spu

            # Mixed units
            mixed = Quantity[3.0u"m", 2.0u"cm", 1.0u"m"]
            for (y, u) in (([1.0 2.0], u"m"), ([1.0 2.0] * u"s", u"m*s"))
                c = zeros(3, 2) * u
                @test uproduct!(c, mixed, y) === c
                @test c == mixed * y
            end

            # Non-array arguments
            c = zeros(4, 3) * u"m"
            @test uproduct!(c, au, 2.0) === c
            @test c == 2au
            c = zeros(4, 3) * u"m*s"
            @test uproduct!(c, 2.0u"s", au) === c
            @test c == 2.0u"s" * au
        end
    end

    @testset "Sorting" begin
        a = rand(1000)
        au = a * u"m"
        M = rand(4, 3) * u"m"

        @testset "Values and result types" begin
            for (x, kw) in (
                (au, (;)),
                (au, (; rev=true)),
                (au, (; alg=RadixSort)),
                (au, (; alg=QuickSort)),
                (rand(-100:100, 1000) * u"m", (; alg=RadixSort)),
                (view(au, 1:10), (;)),
                (view(au, [5, 1, 9, 3]), (;)),
                (reinterpret(eltype(au), a), (;)),
                (M, (; dims=1)),
                (M, (; dims=2)),
                (M', (; dims=1)),
                (transpose(M), (; dims=2)),
            )
                @test typeof(usort(x; kw...)) == typeof(sort(x; kw...))
                @test usort(x; kw...) == sort(x; kw...)
                @test usortperm(x; kw...) == sortperm(x; kw...)
            end

            # Plain arrays use Base
            @test usort(a; alg=QuickSort) == sort(a)
            @test usortperm(a; alg=QuickSort) == sortperm(a)

            # Out-of-place sorting leaves the input unchanged
            auc = copy(au)
            usort(auc)
            usortperm(auc)
            @test auc == au
        end

        @testset "In place" begin
            for x in (au, a), kw in ((;), (; alg=RadixSort), (; alg=QuickSort))
                xc = copy(x)
                @test usort!(xc; kw...) === xc
                @test xc == sort(x)

                ix = Vector{Int}(undef, length(x))
                @test usortperm!(ix, x; kw...) === ix
                @test ix == sortperm(x)
                ix = collect(eachindex(x))
                usortperm!(ix, x; kw..., initialized=true)
                @test x[ix] == sort(x)
            end

            # Wrapped arrays: sorting modifies the parent
            ac = copy(au)
            usort!(view(ac, 6:15))
            @test ac[6:15] == sort(au[6:15])
            @test ac[1:5] == au[1:5] && ac[16:end] == au[16:end]
            ac = copy(au)
            usort!(view(ac, [5, 1, 9, 3]); alg=RadixSort)
            @test ac[[5, 1, 9, 3]] == sort(au[[5, 1, 9, 3]])
            x = copy(a)
            @test usort!(reinterpret(eltype(au), x)) == sort(au)
            @test x == sort(a)
            Mc = copy(M)
            usort!(Mc'; dims=2)
            @test Mc' == sort(M'; dims=2)
            ix = Vector{Int}(undef, 10)
            @test usortperm!(ix, reinterpret(eltype(au), a[1:10])) == sortperm(au[1:10])
        end

        @testset "by, lt, or order" begin
            # These orderings only work on quantities, and sort in descending order
            by = q -> -ustrip(u"cm", q)
            lt = (p, q) -> ustrip(u"cm", p) > ustrip(u"cm", q)
            order = Base.Order.By(by)
            for kw in ((; by), (; lt), (; order))
                @test usort(au; kw...) == sort(au; rev=true)
                ac = copy(au)
                @test usort!(ac; kw...) === ac
                @test ac == sort(au; rev=true)
                @test usortperm(au; kw...) == sortperm(au; rev=true)
                ix = Vector{Int}(undef, length(au))
                @test usortperm!(ix, au; kw...) == sortperm(au; rev=true)
            end
        end

        @testset "Fallbacks" begin
            # Unsupported array types: ranges and OffsetArrays
            for x in ((5:-1:1) * u"m", OffsetArray(au[1:20], -10:9))
                @test typeof(usort(x)) == typeof(sort(x))
                @test usort(x) == sort(x)
                @test usortperm(x) == sortperm(x)
                ix = similar(sortperm(x))
                @test usortperm!(ix, x) == sortperm(x)
            end
            ao = OffsetArray(au[1:20], -10:9)
            usort!(ao)
            @test axes(ao) == (-10:9,)
            @test parent(ao) == sort(au[1:20])

            # Mixed units
            mixed = Quantity[3.0u"m", 2.0u"cm", 1.0u"m"]
            @test usort(mixed) == sort(mixed)
            @test usortperm(mixed) == sortperm(mixed)
            ix = Vector{Int}(undef, length(mixed))
            @test usortperm!(ix, mixed) == sortperm(mixed)
            mc = copy(mixed)
            @test usort!(mc) === mc
            @test mc == sort(mixed)
        end
    end

    @testset "Developer functions" begin
        A = rand(4, 3) * u"m"
        v = rand(4) * u"m"
        for x in (
            A,
            v,
            ustrip(A),
            rand(1:9, 3),
            view(A, 1:2, :),
            view(v, [3, 1]),
            A',
            transpose(v),
            reinterpret(eltype(A), ustrip(A)),
            view(A, :, 1)',
            rand(ComplexF64, 2) * u"m",
        )
            @test FastUnitfulOperations.isfastarray(x)
        end
        for x in (
            Quantity[1.0u"m", 2.0u"cm"],
            Any[1.0, 2.0],
            BigFloat[1, 2],
            ["a", "b"],
            (1:3) * u"m",
            sprand(3, 3, 0.5),
            OffsetArray(v, -1:2),
            SMatrix{2,2}(1.0, 2.0, 3.0, 4.0),
            1.0u"m",
            (1.0, 2.0),
        )
            @test !FastUnitfulOperations.isfastarray(x)
        end

        @test FastUnitfulOperations.convertoperands(A, v) === (A, v)
    end

    @testset "StaticArraysCore extension" begin
        @test Base.get_extension(FastUnitfulOperations, :FastUnitfulOperationsStaticArraysCoreExt) !== nothing

        # Rotation types are static matrices that compute their elements from their parameters
        R = RotZYX(0.3, 0.2, 0.1)
        S = @SMatrix rand(3, 3)
        Su = S * u"s"
        Mm = MMatrix{3,3}(rand(3, 3))
        P = rand(3, 50) * u"m"
        Pt = rand(50, 3) * u"m"
        v = rand(3) * u"m"

        # A static matrix with a matrix of a supported type is converted to a `Matrix`
        for (x, y) in (
            (R, P),
            (RotMatrix(R), P),
            (QuatRotation(R), P),
            (AngleAxis(R), P),
            (R', P),
            (S, P),
            (Su, P),
            (Mm, P),
            (R, view(P, :, 1:10)),
            (Pt, R),
            (P', R),
            (Pt, Su),
        )
            xc, yc = FastUnitfulOperations.convertoperands(x, y)
            @test (x isa StaticMatrix ? xc : yc) isa Matrix
            @test (x isa StaticMatrix ? yc : xc) === (x isa StaticMatrix ? y : x)
            @test typeof(uproduct(x, y)) == typeof(x * y)
            @test uproduct(x, y) ≈ x * y
            c = zeros(eltype(x * y), size(x * y))
            @test uproduct!(c, x, y) === c
            @test c ≈ x * y
        end

        # With vectors and other static arrays, `*` returns a static array
        # A matrix that the fast methods do not support does not lead to a conversion either
        mixed = Quantity[1.0u"m" 2.0u"cm"; 3.0u"m" 4.0u"cm"; 5.0u"m" 6.0u"cm"]
        for (x, y) in ((R, v), (R, SVector{3}(v)), (Su, S), (S, Su), (v', R), (R, mixed))
            @test FastUnitfulOperations.convertoperands(x, y) === (x, y)
            @test typeof(uproduct(x, y)) == typeof(x * y)
            @test uproduct(x, y) == x * y
        end
    end
end
