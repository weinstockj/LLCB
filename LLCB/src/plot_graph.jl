function figure_output_dir()
    dir = output_dir()
    return joinpath(dir, "figures")
end

function plot_scatter(posterior_quantile::Matrix{Float64}, data::DataFrame, x_name::String, y_name::String, beta_x::Float64, intercept::Float64, beta_pip::Float64, beta_x_interact_i::Float64)
    out_dir = figure_output_dir()
    data_copy = deepcopy(data)

    cols = setdiff(ko_targets(), ko_controls())
    z_names = setdiff(cols, [x_name, y_name])
    # z_names = setdiff(z_names, ["intervention"])

    @assert length(z_names) > 10

    # z = Matrix(@view data_copy[:, z_names])
    # lambda = 1.0
    # proj_z = I - z * inv(z' * z + lambda * I) * z'
    # data_copy.y_resid = proj_z * data_copy[!, y_name]
    # data_copy.x_resid = proj_z * data_copy[!, x_name]

    intervention = Vector{String}(undef, nrow(data_copy))
    
    for i in 1:nrow(data_copy)
        if data_copy.intervention[i] == x_name
            intervention[i] = x_name
        elseif data_copy.intervention[i] == y_name
            intervention[i] = y_name
        else
            intervention[i] = "Other KO or control"
        end
    end

    rsq = cor(data_copy[!, y_name], posterior_quantile[:, 3]) ^ 2

    # theme = theme_minimal()
    # set_theme!()
    data_copy.setting = intervention

    fig = Figure(resolution = (800, 600), fontsize = 20)
    ax = fig[1, 1] = Axis(
        fig, 
        xlabel = x_name, 
        ylabel = y_name, 
        title = "Rsq = $(sprintf1("%0.2f", rsq)), β1 = $(sprintf1("%0.2f", beta_x)), PIP = $(sprintf1("%0.2f", beta_pip)), β2 = $(sprintf1("%0.2f", beta_x_interact_i))"
    )
    for (i, g) in enumerate(unique(data_copy.setting))
        
        ind = data_copy.setting .== g
        sub = @view data_copy[ind, :]

        @assert nrow(sub) >= 3

        rangebars!(ax, sub[!, x_name], posterior_quantile[ind, 1], posterior_quantile[ind, 5], whiskerwidth = 5, alpha = 0.2)
        scatter!(ax, sub[!, x_name], sub[!, y_name], alpha = 0.5, label = g)
    end

    abline!(ax, intercept, beta_x, color = :gray, linestyle = :dash, alpha = 0.5)

    # axislegend()
    leg = Legend(fig, ax)
    fig[1, 2] = leg


    # marginal_p = plot(
    #     layer(
    #         data_copy,
    #         x = Symbol(x_name),
    #         y = Symbol(y_name),
    #         color = :setting,
    #         Geom.point
    #     ),
    #     layer(
    #         x = data_copy[!, x_name],
    #         y = posterior_quantile[:, 3],
    #         ymin = posterior_quantile[:, 1],
    #         ymax = posterior_quantile[:, 5],
    #         Geom.errorbar
    #     ),
    #     Guide.xlabel(x_name),
    #     Guide.ylabel(y_name),
    #     Theme(
    #         background_color = color("white"),
    #         major_label_font_size = 17pt,
    #         minor_label_font_size = 13pt,
    #         panel_stroke=color("black"),
    #         default_color = color("black"),
    #         highlight_width = 1pt
    #     )
    # )


    save(joinpath(out_dir, "posterior_scatter_$(x_name)_$(y_name).png"), fig)
    # draw(PNG(joinpath(out_dir, "posterior_scatter_$(x_name)_$(y_name).png"), 6inch, 4inch), marginal_p)
end

function plot_scatter(beta_x::Float64, intercept::Float64, beta_x_interact_i::Float64, data::DataFrame, x_name::String, y_name::String, cols = setdiff(ko_targets(), ko_controls()))
    out_dir = figure_output_dir()
    data_copy = deepcopy(data)

    # cols = setdiff(ko_targets(), ko_controls())
    z_names = setdiff(cols, [x_name, y_name])
    # z_names = setdiff(z_names, ["intervention"])

    @assert length(z_names) > 10

    # println("z_names: $z_names")
    # println("z_name length: $(length(z_names))")
    z = Matrix(@view data_copy[:, z_names])
    # Q, R = qr(z)
    # proj_z = I - Q * Q'
    lambda = 1.0
    proj_z = I - z * inv(z' * z + lambda * I) * z'
    data_copy.y_resid = proj_z * data_copy[!, y_name]
    data_copy.x_resid = proj_z * data_copy[!, x_name]

    intervention = Vector{String}(undef, nrow(data_copy))
    
    for i in 1:nrow(data_copy)
        if data_copy.intervention[i] == x_name
            intervention[i] = x_name
        elseif data_copy.intervention[i] == y_name
            intervention[i] = y_name
        else
            intervention[i] = "Other KO or control"
        end
    end

    data_copy.setting = intervention

    y_resid_var = var(data_copy[data_copy.setting .!= y_name, "y_resid"])
    y_observed_var = var(data_copy[data_copy.setting .!= y_name, y_name])

    if y_resid_var > y_observed_var
        println("y resid var $y_resid_var, y observed var $y_observed_var")
        println(data_copy[:, ["setting", "y_resid", y_name]])
    end

    @assert y_resid_var < 10 * y_observed_var

    marginal_p = plot(
        data_copy,
        x = Symbol(x_name),
        y = Symbol(y_name),
        color = :setting,
        Geom.point,
        intercept = [intercept],
        slope = [beta_x],
        Geom.abline,
        Guide.xlabel(x_name),
        Guide.ylabel(y_name),
        Theme(
            background_color = color("white"),
            major_label_font_size = 17pt,
            minor_label_font_size = 13pt,
            panel_stroke=color("black"),
            default_color = color("black"),
            highlight_width = 1pt
        )
    )

    adj_p = plot(
        data_copy,
        x = :x_resid,
        y = :y_resid,
        color = :setting,
        Geom.point,
        intercept = [intercept],
        slope = [beta_x],
        Geom.abline,
        Guide.xlabel("$(x_name) residuals"),
        Guide.ylabel("$(y_name) residuals"),
        Theme(
            background_color = color("white"),
            major_label_font_size = 17pt,
            minor_label_font_size = 13pt,
            panel_stroke=color("black"),
            default_color = color("black"),
            highlight_width = 1pt
        )
    )

    m_p = hstack([marginal_p, adj_p])

    # draw(PDF(joinpath(out_dir, "scatter_$(x_name)_$(y_name).pdf"), 6inch, 4inch), marginal_p)
    # draw(PNG(joinpath(out_dir, "scatter_$(x_name)_$(y_name).png"), 6inch, 4inch), marginal_p)
    # draw(SVG(joinpath(out_dir, "scatter_$(x_name)_$(y_name).svg"), 6inch, 4inch), marginal_p)
    draw(SVG(joinpath(out_dir, "merged_scatter_$(x_name)_$(y_name).svg"), 8inch, 4inch), m_p)
end

function plot_scatter(parsed_skeleton::DataFrame, data::DataFrame)
    
    Threads.@threads for i in 1:nrow(parsed_skeleton)
    # for i in 1:nrow(parsed_skeleton)
        x_name = parsed_skeleton.x[i]
        y_name = parsed_skeleton.y[i]
        @info " $(now()) Now plotting x = $(x_name) and y = $(y_name)"
        # data_subset = data[:, [x_name, y_name]]
        data_subset = data
        plot_scatter(
            parsed_skeleton.beta_x[i],
            parsed_skeleton.intercept[i],
            parsed_skeleton.beta_x_interact_i[i],
            data_subset,
            parsed_skeleton.x[i],
            parsed_skeleton.y[i]
        )
    end
end

function plot_scatter(parsed_skeleton::DataFrame, data::DataFrame, posterior_quantile::Vector{Matrix{Float64}})
    
    # Threads.@threads for i in 1:nrow(parsed_skeleton)
    for i in 1:nrow(parsed_skeleton)
        x_name = parsed_skeleton.x[i]
        y_name = parsed_skeleton.y[i]
        @info " $(now()) Now plotting x = $(x_name) and y = $(y_name)"
        plot_scatter(
            posterior_quantile[i],
            data,
            parsed_skeleton.x[i],
            parsed_skeleton.y[i],
            parsed_skeleton.beta_x[i],
            parsed_skeleton.intercept[i],
            # parsed_skeleton.beta_pip[i],
            parsed_skeleton.beta_pip_crude[i],
            parsed_skeleton.beta_x_interact_i[i]
        )
    end
end

function plot_posterior_mean(graph::MetaDiGraph)
    out_dir = figure_output_dir()

    nlabels = Vector{String}(undef, nv(graph))
    elabels = Vector{String}(undef, ne(graph))
    edge_color = Vector{Float64}(undef, ne(graph))
    for i in 1:nv(graph)
        nlabels[i] = graph.vprops[i][:name]
    end
    #TODO: don't hard code these!
    min_weight = -0.9
    max_weight = 0.9 
    # max_weight = maximum(graph.skeleton.beta_x)
    # max_weight = maximum(graph.skeleton.beta_x)

    for (i, val) in enumerate(values(graph.eprops))
        edge_color[i] = val[:weight]
        # elabels[i] = string(elabels_color[i])
    end
    # set_theme!(resolution = (900, 900))
    fig = Figure(resolution=(1100,750))
    # fig[1,1] = title = Label(fig, "Skeleton", textsize=20)
    fig[1,1] = title = Label(fig, "", textsize=1)
    title.tellwidth = false

    println("debug 2")
    fig[2,1] = ax = Axis(fig)
    p = graphplot!(ax, graph;
        layout = SFDP(Ptype=Float32, tol=0.001, C=3.0, K=70.0, iterations = 900),
        # layout = Spectral(dim = 2),
        # layout = Spring(Ptype = Float32),
        nlabels = nlabels, 
        # elabels = elabels
        # arrows_show = true,
        # arrows_shift = 0.9,
        edge_color = edge_color,
        edge_width = 5.0,
        edge_attr=(colormap=Reverse(:RdBu_5), colorrange = (min_weight, max_weight))
    )
    # offsets = 0.15 * (p[:node_pos][] .- p[:node_pos][][1])
    # p.nlabels_offset[] = offsets
    hidedecorations!(ax); hidespines!(ax)
    ax.aspect = DataAspect()
    # autolimits!(ax)
    #
    # println("p[:node_pos] = $(p[:node_pos])")
    #
    node_x = Vector{Float32}(undef, nv(graph))
    node_y = Vector{Float32}(undef, nv(graph))
    # println("p[:node_pos][] = $(p[:node_pos][])")
    # println("p[:node_pos][:, 1] = $(p[:node_pos][:, 1])")

    idx = 0
    for xy in p[:node_pos][]
        idx = idx + 1
        node_x[idx] = xy[1]
        node_y[idx] = xy[2]
    end

    buffer_x = 45
    buffer_y = 40
    xlims!(ax, minimum(node_x) - buffer_x, maximum(node_x) + buffer_x)
    ylims!(ax, minimum(node_y) - buffer_y, maximum(node_y) + buffer_y)


    fig[2,2] = cb = Colorbar(fig, p.plots[1], label = "", vertical=true)

    save(joinpath(out_dir, "directed_skeleton_plot.png"), fig)
    save(joinpath(out_dir, "directed_skeleton_plot.pdf"), fig)
    set_theme!()
end

function plot_posterior_mean(graph::interventionGraph)
    out_dir = figure_output_dir()

    nlabels = Vector{String}(undef, nv(graph.g))
    elabels = Vector{String}(undef, ne(graph.g))
    edge_color = Vector{Float64}(undef, ne(graph.g))
    for i in 1:nv(graph.g)
        nlabels[i] = graph.g.vprops[i][:name]
    end
    #TODO: don't hard code these!
    min_weight = -0.9
    max_weight = 0.9 
    # max_weight = maximum(graph.skeleton.beta_x)
    # max_weight = maximum(graph.skeleton.beta_x)

    for (i, val) in enumerate(values(graph.g.eprops))
        edge_color[i] = val[:weight]
        # elabels[i] = string(elabels_color[i])
    end
    # set_theme!(resolution = (900, 900))
    fig = Figure(resolution=(1100,750))
    # fig[1,1] = title = Label(fig, "Skeleton", textsize=20)
    fig[1,1] = title = Label(fig, "", textsize=1)
    title.tellwidth = false

    fig[2,1] = ax = Axis(fig)
    p = graphplot!(ax, graph.g;
        layout = SFDP(Ptype=Float32, tol=0.001, C=3.0, K=95.0, iterations = 900),
        # layout = Spectral(dim = 2),
        # layout = Spring(Ptype = Float32),
        nlabels = nlabels, 
        # elabels = elabels
        arrows_show = true,
        arrows_shift = 0.9,
        edge_color = edge_color,
        edge_width = 5.0,
        edge_attr=(colormap=Reverse(:RdBu_5), colorrange = (min_weight, max_weight))
    )
    # offsets = 0.15 * (p[:node_pos][] .- p[:node_pos][][1])
    # p.nlabels_offset[] = offsets
    hidedecorations!(ax); hidespines!(ax)
    ax.aspect = DataAspect()
    # autolimits!(ax)
    #
    # println("p[:node_pos] = $(p[:node_pos])")
    #
    node_x = Vector{Float32}(undef, nv(graph.g))
    node_y = Vector{Float32}(undef, nv(graph.g))
    # println("p[:node_pos][] = $(p[:node_pos][])")
    # println("p[:node_pos][:, 1] = $(p[:node_pos][:, 1])")

    idx = 0
    for xy in p[:node_pos][]
        idx = idx + 1
        node_x[idx] = xy[1]
        node_y[idx] = xy[2]
    end

    buffer_x = 45
    buffer_y = 40
    xlims!(ax, minimum(node_x) - buffer_x, maximum(node_x) + buffer_x)
    ylims!(ax, minimum(node_y) - buffer_y, maximum(node_y) + buffer_y)


    fig[2,2] = cb = Colorbar(fig, p.plots[1], label = "", vertical=true)

    save(joinpath(out_dir, "directed_skeleton_plot.png"), fig)
    save(joinpath(out_dir, "directed_skeleton_plot.pdf"), fig)
    set_theme!()
end

function plot_skeleton_as_matrix(s::DataFrame, suffix::String)

    out_dir = figure_output_dir()

    genes = ["CBFB", "TNFAIP3", "KLF2", "FOXK1", "ATXN7L3", "ZNF217", "HIVEP2", "IRF2", "MYB", "IRF1", 
             "MED12", "YY1", "MBD2", "RELA", "ETS1", "PTEN", "FOXP1", "JAK3", "KMT2A", "IRF4", "GATA3", 
             "STAT5A", "STAT5B", "IL2RA"]

    X = genes
    Y = reverse(genes)
    N_genes = length(X)

    values = zeros(N_genes, N_genes)

    for i in 1:N_genes
        for j in 1:N_genes
            val = s[(s.row .== X[j]) .& (s.col .== Y[i]), "estimate"]
            @assert length(val) <= 1
            if length(val) == 1
                values[i, j] = val[1]
                # values[j, i] = val[1]
            end
        end
    end

    fig = Figure(resolution=(800, 700))

    fig[1,1] = ax = Axis(fig)

    cmap = colormap("RdBu", 5; mid = 0)

    println("values = $values")

    println("debug me 3")
    max_value = maximum(abs.(values))
    CairoMakie.heatmap!(
        ax,
        1:N_genes,
        1:N_genes,
        Float32.(values),
        # colormap = Reverse(:RdBu_9),
        colormap = Reverse(cmap),
        # colormap = colormap(:vik, 5; mid = 0),
        # colorange = (-1, 1)
        colorange = (-max_value, max_value)
    )

    # fig[1,2] = cb = Colorbar(fig, colormap = Reverse(:RdBu_9), limits = (-max_value, max_value), label = "edge", vertical=true)
    # fig[1,2] = cb = Colorbar(fig, colormap = Reverse(cmap), limits = (-max_value, max_value), label = "edge", vertical=true)
    fig[1,2] = cb = Colorbar(fig, colormap = cmap, limits = (-max_value, max_value), label = "edge", vertical=true)

    ax.xticks = (1:N_genes, Y)
    ax.xticklabelrotation = 45.0
    ax.yticks = (1:N_genes, X)

    ax.aspect = DataAspect()

    save(joinpath(out_dir, "directed_skeleton_matrix_$(suffix).png"), fig)
    save(joinpath(out_dir, "directed_skeleton_matrix_$(suffix).pdf"), fig)
    set_theme!()

end
