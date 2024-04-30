using Gen
using Plots
include("utils.jl")

@gen function line_model_fancy(xs::Vector{Float64})
    slope = ({:slope} ~ normal(0, 1))
    intercept = ({:intercept} ~ normal(0, 2))
    
    function y(x)
        return slope * x + intercept
    end
    
    noise = ({:noise} ~ gamma(1, 1))
    for (i, x) in enumerate(xs)
        {(:y, i)} ~ normal(slope * x + intercept, noise)
    end
    return y
end;

@gen function sine_model_fancy(xs::Vector{Float64})

    # Sampling a phase, period, and amplitude >
    period = ({:period} ~ Gen.gamma(1, 1))
    amplitude = ({:amplitude} ~ Gen.gamma(1, 1))
    phase = ({:phase} ~ Gen.uniform(0, 2*pi))
    noise = ({:noise} ~ gamma(1, 1))

    function y(x)
        return amplitude*sin(2*pi*(1/period)*x+phase)
    end
    
    for (i, x) in enumerate(xs)
        {(:y, i)} ~ normal(y(x), noise)
    end
    
    return y # We return the y function so it can be used for plotting, below. 
end;

@gen function combined_model(xs::Vector{Float64})
    if ({:is_line} ~ bernoulli(0.5))
        # Call line_model_fancy on xs, and import
        # its random choices directly into our trace.
        return ({*} ~ line_model_fancy(xs))
    else
        # Call sine_model_fancy on xs, and import
        # its random choices directly into our trace
        return ({*} ~ sine_model_fancy(xs))
    end
end;

xs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.];
ys = [6.75003, 6.1568, 4.26414, 1.84894, 3.09686, 1.94026, 1.36411, -0.83959, -0.976, -1.93363, -2.91303];
ys_sine = [2.89, 2.22, -0.612, -0.522, -2.65, -0.133, 2.70, 2.77, 0.425, -2.11, -2.76];

traces = [Gen.simulate(combined_model, (xs,)) for _=1:12];
fig = grid(render_trace, traces)
savefig(fig, string(@__DIR__)*"/res/combine.png")

traces = [do_inference(combined_model, xs, ys, 10000) for _=1:10];
linear_dataset_plot = overlay(render_trace, traces)
traces = [do_inference(combined_model, xs, ys_sine, 10000) for _=1:10];
sine_dataset_plot = overlay(render_trace, traces)
fig = Plots.plot(linear_dataset_plot, sine_dataset_plot)
savefig(fig, string(@__DIR__)*"/res/combine_overlay.png")