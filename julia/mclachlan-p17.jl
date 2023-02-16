using DataFrames, Distributions, DataFramesMeta
using Pipe: @pipe

function gen_data(n = 500000, mu = [0, 2], sigma = 1, pi = 0.8)
    data = DataFrame(
        Obs_ID = 1:n,
        Z = (rand(Uniform(0,1), n) .> pi) .+ 1,
        )

    @transform!(data, :Y = rand(Normal(0, sigma), n) + mu[:Z])    
    @select!(data, :Y, :Obs_ID)
    @transform!(data, :p_Y_given_Z_1 = pdf.(Normal(mu[1], sigma), :Y))
    @transform!(data, :p_Y_given_Z_2 = pdf.(Normal(mu[2], sigma), :Y))

    return data
  end

function fit_model(
    data; 
    pi_init = 0.5, 
    tolerance = 0.001,
    max_iterations = 1000)

    progress = [(pi_hat=pi_init, loglik=0.0)]
    for i in 1:max_iterations
        Q = E_step(data, pi_init)
        mu, sigma, pi = M_step(Y, Q)
        log_likelihood = sum(logpdf.(MixtureModel(pi, [Normal(mu[1], sigma), Normal(mu[2], sigma)]), Y))
        push!(progress, (pi_hat=pi, log_likelihood=log_likelihood))
        if abs(log_likelihood - progress[end-1].log_likelihood) < tolerance
            break
        end
    end
    return (mu, sigma, pi, progress)
en

function E_step(Y, mu, sigma, pi)
    Q = Array{Float64}(undef, length(Y))
    for i in 1:length(Y)
        Q[i] = pi * pdf(Normal(mu[1], sigma), Y[i]) /
               (pi * pdf(Normal(mu[1], sigma), Y[i]) + (1-pi) * pdf(Normal(mu[2], sigma), Y[i]))
    end
    return Q
end
rs
function M_step(Y, Q)
    N1 = sum(Q)
    N2 = length(Q) - N1
    mu = [sum(Q .* Y) / N1, sum((1-Q) .* Y) / N2]
    sigma = sqrt((sum(Q .* (Y - mu[1]).^2) / N1 + sum((1-Q) .* (Y - mu[2]).^2) / N2) / length(Y))
    pi = N1 / length(Y)
    return (mu, sigma, pi)
end

Y = gen_data(1000, [5, 10], 2, 0.5)
@time fit_model(Y, [0, 20], 2, 0.5)
