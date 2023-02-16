using DataFrames

function gen_data(n = 500000, mu = [0, 2], sigma = 1, pi = 0.8)
  complete_data = DataFrame(Obs_ID = 1:n,
                            Z = (rand(Uniform(0,1), n) .> pi) .+ 1,
                            Y = rand(Normal(mu[Z], sigma), n))

  observed_data = complete_data[:, [:Obs_ID, :Y]]
  p_Y_given_Z = observed_data
  p_Y_given_Z[:p_Y_given_Z_1] = pdf(Normal(mu[1], sigma), p_Y_given_Z[:Y])
  p_Y_given_Z[:p_Y_given_Z_2] = pdf(Normal(mu[2], sigma), p_Y_given_Z[:Y])

  return p_Y_given_Z
end

srand(1)
p_Y_given_Z = gen_data()

function fit_model(p_Y_given_Z)
  function E_step(p_Y_given_Z, p_Z_1)
    p_Y_Z_1 = p_Y_given_Z[:p_Y_given_Z_1] * p_Z_1
    p_Y_Z_2 = p_Y_given_Z[:p_Y_given_Z_2] * (1 - p_Z_1)
    p_Y = p_Y_Z_1 + p_Y_Z_2
    p_Z_1_given_Y = p_Y_Z_1 ./ p_Y

    p_Y_given_Z[:p_Y_Z_1] = p_Y_Z_1
    p_Y_given_Z[:p_Y_Z_2] = p_Y_Z_2
    p_Y_given_Z[:p_Y] = p_Y
    p_Y_given_Z[:p_Z_1_given_Y] = p_Z_1_given_Y

    return p_Y_given_Z
  end

  function likelihood(p_Z_1, p_Y_given_Z)
    p_Y_given_Z = E_step(p_Y_given_Z, p_Z_1)
    return prod(p_Y_given_Z[:p_Y])
  end

  function loglik(p_Z_1, p_Y_given_Z)
    return log(likelihood(p_Z_1, p_Y_given_Z))
  end

  p_Z_1 = 0.5
  diff = Inf
  tolerance = 0.00001
  progress = DataFrame(Iteration = 0, p_Z_1 = p_Z_1, loglik = loglik(p_Z_1, p_Y_given_Z))
  max_iterations = 1000

#   for i in 1:max_iterations
#     p_Y_given_Z = E_step(p_Y_given_Z, p_Z_1)
#     pi_hat_prev = p_Z_1
#     p_
end