data {
  int<lower=1> N;
  int<lower=1> T;

  array[N] int home_team;
  array[N] int away_team;
  array[N] int home_goals;
  array[N] int away_goals;
}

parameters {
  real home_adv;
  real mu;

  vector[T] attack_raw;
  vector[T] defense_raw;

  real<lower=0> sigma_attack;
  real<lower=0> sigma_defense;
}

transformed parameters {
  vector[T] attack = attack_raw * sigma_attack;
  vector[T] defense = defense_raw * sigma_defense;
}

model {
  // Priors
  home_adv ~ normal(0, 0.5);
  mu ~ normal(0, 1);

  attack_raw ~ normal(0, 1);
  defense_raw ~ normal(0, 1);

  sigma_attack ~ exponential(1);
  sigma_defense ~ exponential(1);

  // Likelihood
  for (n in 1:N) {
    home_goals[n] ~ poisson_log(
      mu + home_adv + attack[home_team[n]] - defense[away_team[n]]
    );

    away_goals[n] ~ poisson_log(
      mu + attack[away_team[n]] - defense[home_team[n]]
    );
  }
}
