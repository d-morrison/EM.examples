using DataFrames, Distributions, DataFramesMeta

function gen_data(;n = 500000, mu = [0, 2], sigma = 1, pZ1 = 0.8)
    data = DataFrame(
        Obs_ID = 1:n,
        Z = (rand(Uniform(0,1), n) .> pZ1) .+ 1,
        )

    @transform!(data, :Y = rand(Normal(0, sigma), n) + mu[:Z])    
    @select!(data, :Obs_ID, :Y)
    @transform!(data, :p_Y_given_Z_1 = pdf.(Normal(mu[1], sigma), :Y))
    @transform!(data, :p_Y_given_Z_2 = pdf.(Normal(mu[2], sigma), :Y))

    return data
  end

  function fit_model!(
    data; 
    pi_hat_0 = 0.5, 
    tolerance = 0.0001,
    max_iterations = 1000,
    progress = DataFrame(
        iter = 1:(max_iterations+1), 
        pi_hat = Vector{Float64}(undef, max_iterations+1), 
        ll = Vector{Float64}(undef, max_iterations+1), 
        ll_diff = Vector{Float64}(undef, max_iterations+1)
        )
    )

    pi_hat = pi_hat_0
    E_step!(data, pi_hat)
    ll = loglik!(data)
    # progress = [(iter = 0, pi_hat, ll, ll_diff = NaN)]
    progress[1,:] = (0, pi_hat, ll, NaN)
    
    last_iter = 0
    for i in 1:max_iterations
        pi_hat = M_step(data)
        E_step!(data, pi_hat)
        
        ll_old = ll
        ll = loglik!(data)
        ll_diff = ll - ll_old
        progress[i+1,:] = (i, pi_hat, ll, ll_diff)

        if ll_diff < tolerance
            last_iter = i
            break
        end
    end
    return progress[1:last_iter, :]
end

function E_step!(data, pi_hat)
    @transform!(data, :pY_Z1 = :p_Y_given_Z_1 .* pi_hat)
    @transform!(data, :pY_Z2 = :p_Y_given_Z_2 .* (1- pi_hat))
    @transform!(data, :pY = :pY_Z1 + :pY_Z2)
    @transform!(data, :pZ1_given_Y = :pY_Z1 ./ :pY)
end

function M_step(data)
    mean(data[!, :pZ1_given_Y])
end

function loglik!(data)
    sum(log.(data[!, :pY]))
end

data = gen_data(n = 1000, pZ1 = 0.8)
@time fit_model!(data)
