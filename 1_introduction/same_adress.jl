using Gen

@gen function line_model(xs::Vector{Float64})

    slope = ({:slope} ~ normal(0, 1))
    intercept = ({:intercept} ~ normal(0, 2))
    
    # We define a function to compute y for a given x
    function y(x)
        return slope * x + intercept
    end

    for (i, x) in enumerate(xs)
        ({(:y, 1)} ~ normal(y(x), 0.1))
    end
end

xs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.];
trace = Gen.simulate(line_model, (xs,));

println(Gen.get_choices(trace))

# Error: LoadError: Attempted to visit address (:y, 1), but it was already visited