---
title:  "Installing Julia in Windows and Running in Jupyter Notebook"
date:   2022-02-13 10:29:17 +0545
# last_modified_at: 2022-01-23 12:29:17 +0545
categories:
    - Julia
tags:
    - data analysis
    - jupyter
header:
  teaser: assets/install_julia/tmp.gif
---

## Installation of Julia
* First download the installer from [official site](https://julialang.org/downloads/) then run the installer. 
* Set the environvent path variable.
* Test it by opening the Julia Command Line or REPL (Read Eval Print Loop).

```julia
println("hello world")
```

  ![png]({{site.url}}/assets/julia_install/cmd.png) 

## Adding Julia into Jupyter
* First add package `IJulia` by typing `use Pkg` and then enter.
* Then `Pkg.add("IJulia")`. 
* Wait for the completion.

![png]({{site.url}}/assets/julia_install/ijulia.png)

## Test it
* Hit `jupyter notebook` from anaconda prompt or terminal.
* Selct Julia from dropdown.

![]({{site.url}}/assets/julia_install/jupyter_julia.png)

* Create a new notebook and run a code.

![]({{site.url}}/assets/julia_install/first_cell.png)

## Plot in Julia
Lets try to plot a Lorenz Attractor visualization in Julia. For code reference check [here](https://docs.juliaplots.org/stable/).

But we need to install package `Plots`. So lets do it from Julia Command Line by:

```shell
using Pkg
Pkg.add("Plots")
```

Please have patience, it will take some time.

Once done something like below will be seen.

![]({{site.url}}/assets/julia_install/plots_install.png)

```julia
using Plots
# define the Lorenz attractor
Base.@kwdef mutable struct Lorenz
    dt::Float64 = 0.02
    σ::Float64 = 10
    ρ::Float64 = 28
    β::Float64 = 8/3
    x::Float64 = 1
    y::Float64 = 1
    z::Float64 = 1
end

function step!(l::Lorenz)
    dx = l.σ * (l.y - l.x)
    dy = l.x * (l.ρ - l.z) - l.y
    dz = l.x * l.y - l.β * l.z
    l.x += l.dt * dx
    l.y += l.dt * dy
    l.z += l.dt * dz
end

attractor = Lorenz()


# initialize a 3D plot with 1 empty series
plt = plot3d(
    1,
    xlim = (-30, 30),
    ylim = (-30, 30),
    zlim = (0, 60),
    title = "Lorenz Attractor",
    marker = 2,
)

# build an animated gif by pushing new points to the plot, saving every 10th frame
@gif for i=1:1500
    step!(attractor)
    push!(plt, attractor.x, attractor.y, attractor.z)
end every 10
```

Once done, a `tmp.gif` file will be stored on the working directory.

<img src = "{{site.url}}/assets/install_julia/tmp.gif">

## Reading CSV
### Reading Local CSV
* Add packages CSV and DataFrames.
```
Pkg.add("CSV)
Pkg.add("DataFrames")
```

* Now using it.

```julia
using CSV, DataFrames

CSV.read("country_info_lat.csv",DataFrame, header=1, delim=",")
```

![]({{site.url}}/assets/julia_install/df.png)
