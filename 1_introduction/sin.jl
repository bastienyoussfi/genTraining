using Gen
using Plots

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

function render_trace(trace; show_data=true)
    
    # Pull out xs from the trace
    xs, = get_args(trace)
    
    xmin = minimum(xs)
    xmax = maximum(xs)

    # Pull out the return value, useful for plotting
    y = get_retval(trace)
    
    # Draw the line
    test_xs = collect(range(-5, stop=5, length=1000))
    fig = plot(test_xs, map(y, test_xs), color="black", alpha=0.5, label=nothing,
                xlim=(xmin, xmax), ylim=(xmin, xmax))

    if show_data
        ys = [trace[(:y, i)] for i=1:length(xs)]
        
        # Plot the data set
        scatter!(xs, ys, c="black", label=nothing)
    end
    
    return fig
end;

function grid(renderer::Function, traces)
     return Plots.plot(map(renderer, traces)...)
end;

xs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.];

traces = [Gen.simulate(sine_model, (xs,)) for _=1:12];
fig = grid(render_trace, traces)
savefig(fig, string(@__DIR__)*"/grid_sine.png")
