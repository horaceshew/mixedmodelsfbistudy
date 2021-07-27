data {
  int N;                               // Number of people
  int J;                               // Number of items
  int Y[N,J];                          // Binary Target
}

transformed data{
  
}

parameters {
  vector[J] difficulty;                // Item difficulty
  real<lower = 0> discrim;             // Item discrimination (constant)
  vector[N] Z;                         // Latent person ability
}

model {
  matrix[N, J] lmat;
  
  // priors
  Z ~ normal(0, 1);
  
  discrim    ~ student_t(3, 0, 5);
  difficulty ~ student_t(3, 0, 5);

  for (j in 1:J){
    lmat[,j] = discrim * (Z - difficulty[j]);
  }
  
  // likelihood
  for (j in 1:J)  Y[,j] ~ bernoulli_logit(lmat[,j]);
  
}
