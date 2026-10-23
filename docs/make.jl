using FastUnitfulOperations
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(FastUnitfulOperations, :DocTestSetup, :(using FastUnitfulOperations); recursive=true)

links = InterLinks("Julia" => "https://docs.julialang.org/en/v1/")

makedocs(;
    modules=[FastUnitfulOperations],
    plugins=[links],
    authors="Lucas Valenzuela",
    sitename="FastUnitfulOperations.jl",
    format=Documenter.HTML(;
        canonical="https://lucasvalenzuela.github.io/FastUnitfulOperations.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/lucasvalenzuela/FastUnitfulOperations.jl", devbranch="main")
