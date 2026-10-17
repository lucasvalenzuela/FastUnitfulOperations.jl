using FastUnitfulOperations
using Documenter

DocMeta.setdocmeta!(FastUnitfulOperations, :DocTestSetup, :(using FastUnitfulOperations); recursive=true)

makedocs(;
    modules=[FastUnitfulOperations],
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
