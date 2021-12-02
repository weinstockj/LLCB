function figure_output_dir()
    dir = output_dir()
    return dir
end

function plot_scatter(posterior_quantile::Matrix{Float64}, data::DataFrame, x_name::String, y_name::String)
    out_dir = figure_output_dir()
    data_copy = deepcopy(data)

    cols = setdiff(ko_targets(), ko_controls())
    z_names = setdiff(cols, [x_name, y_name])
    # z_names = setdiff(z_names, ["intervention"])

    @assert length(z_names) > 10

    z = Matrix(@view data_copy[:, z_names])
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

    marginal_p = plot(
        layer(
            data_copy,
            x = Symbol(x_name),
            y = Symbol(y_name),
            color = :setting,
            Geom.point
        ),
        layer(
            x = data_copy[!, x_name],
            y = posterior_quantile[:, 3],
            ymin = posterior_quantile[:, 1],
            ymax = posterior_quantile[:, 5],
            Geom.errorbar
        ),
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


    draw(SVG(joinpath(out_dir, "posterior_scatter_$(x_name)_$(y_name).svg"), 6inch, 4inch), marginal_p)
end

function plot_scatter(beta_x::Float64, intercept::Float64, beta_x_interact_i::Float64, data::DataFrame, x_name::String, y_name::String)
    out_dir = figure_output_dir()
    data_copy = deepcopy(data)

    cols = setdiff(ko_targets(), ko_controls())
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
    
    Threads.@threads for i in 1:nrow(parsed_skeleton)
    # for i in 1:nrow(parsed_skeleton)
        x_name = parsed_skeleton.x[i]
        y_name = parsed_skeleton.y[i]
        @info " $(now()) Now plotting x = $(x_name) and y = $(y_name)"
        # data_subset = data[:, [x_name, y_name]]
        data_subset = data
        plot_scatter(
            posterior_quantile[i],
            data_subset,
            parsed_skeleton.x[i],
            parsed_skeleton.y[i]
        )
    end
end
