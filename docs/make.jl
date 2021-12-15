using FastUnitfulOperations
using Documenter

DocMeta.setdocmeta!(FastUnitfulOperations, :DocTestSetup, :(using FastUnitfulOperations); recursive=true)

makedocs(;
    modules=[FastUnitfulOperations],
    authors="Lucas Valenzuela",
    repo="https://github.com/lucasvalenzuela/FastUnitfulOperations.jl/blob/{commit}{path}#{line}",
    sitename="FastUnitfulOperations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lucasvalenzuela.github.io/FastUnitfulOperations.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lucasvalenzuela/FastUnitfulOperations.jl",
    devbranch="main",
)
