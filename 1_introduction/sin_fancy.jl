using Gen
using Plots
include("utils.jl")

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


xs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.];
new_xs = collect(range(-5, stop=10, length=100));
ys_sine = [2.89, 2.22, -0.612, -0.522, -2.65, -0.133, 2.70, 2.77, 0.425, -2.11, -2.76];
ys_noisy = [5.092, 4.781, 2.46815, 1.23047, 0.903318, 1.11819, 2.10808, 1.09198, 0.0203789, -2.05068, 2.66031];


# Modify the line below to experiment with the amount_of_computation parameter
pred_ys = infer_and_predict(sine_model, xs, ys_sine, new_xs, [], 20, 1)
fixed_noise_plot = plot_predictions(xs, ys_sine, new_xs, pred_ys; title="Fixed noise level")

# Modify the line below to experiment with the amount_of_computation parameter
pred_ys = infer_and_predict(sine_model_fancy, xs, ys_sine, new_xs, [], 20, 1)
inferred_noise_plot = plot_predictions(xs, ys_sine, new_xs, pred_ys; title="Inferred noise level")

fig = Plots.plot(fixed_noise_plot, inferred_noise_plot)
savefig(fig, string(@__DIR__)*"/res/comparison_sines.png")

# Modify the line below to experiment with the amount_of_computation parameter
pred_ys = infer_and_predict(sine_model, xs, ys_noisy, new_xs, [], 20, 1)
fixed_noise_plot = plot_predictions(xs, ys_noisy, new_xs, pred_ys; title="Fixed noise level")

# Modify the line below to experiment with the amount_of_computation parameter
pred_ys = infer_and_predict(sine_model_fancy, xs, ys_noisy, new_xs, [], 20, 1)
inferred_noise_plot = plot_predictions(xs, ys_noisy, new_xs, pred_ys; title="Inferred noise level")

fig = Plots.plot(fixed_noise_plot, inferred_noise_plot)
savefig(fig, string(@__DIR__)*("/res/comparison_y"))