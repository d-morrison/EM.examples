bin_mix_pdf = function(x, pi, `p(X=1|Z=0)` = 1/2, `p(X=1|Z=1)` = 1/4)
{
  `p(X=x|Z=0)` = dbinom(size = 1, prob = `p(X=1|Z=0)`, x = x)
  `p(X=x|Z=1)` = dbinom(size = 1, prob = `p(X=1|Z=1)`, x = x)
  `p(X=x)` = `p(X=x|Z=0)`*pi + `p(X=x|Z=1)`*(1-pi)
  return(prod(`p(X=x)`))
}

#' Title
#'
#' @param x
#' @param `p(X=1|Z=0)`
#' @param `p(X=1|Z=1)`
#' @param pi
#'
#' @return
#' @export
#'
likelihood_bin_mix = Vectorize(FUN = bin_mix_pdf, "pi")

