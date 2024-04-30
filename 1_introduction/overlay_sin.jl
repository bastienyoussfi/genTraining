using Gen
using Plots
import("utils.jl")

@gen function sine_model(xs::Vector{Float64})
    
    # Sampling a phase, period, and amplitude >
    period = ({:period} ~ Gen.gamma(1, 1))
    amplitude = ({:amplitude} ~ Gen.gamma(1, 1))
    phase = ({:phase} ~ Gen.uniform(0, 2*pi))

    function y(x)
        return amplitude*sin(2*pi*(1/period)*x+phase)
    end
    
    for (i, x) in enumerate(xs)
        {(:y, i)} ~ normal(y(x), 0.1)
    end
    
    return y # We return the y function so it can be used for plotting, below. 
end;

xs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.];
ys = [2.89, 2.22, -0.612, -0.522, -2.65, -0.133, 2.70, 2.77, 0.425, -2.11, -2.76];

traces = [do_inference(sine_model, xs, ys, 1000) for _=1:10];
fig = overlay(render_trace, traces)
savefig(fig, string(@__DIR__)*"/overlay_sin.png")