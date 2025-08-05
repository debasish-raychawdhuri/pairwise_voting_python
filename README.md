# Pairwise Ranking

A Python library for computing rankings from pairwise comparison data using maximum likelihood estimation and Bayesian inference.

## Overview

This library implements algorithms to:
- Convert pairwise winner-loser comparisons into numerical scores
- Compute rankings from these scores
- Estimate uncertainty in rankings using Bayesian methods
- Calculate probabilities for ranking scenarios

## Features

- **Maximum Likelihood Estimation**: Optimizes scores to best explain observed pairwise outcomes
- **Bayesian Uncertainty**: Computes Hessian-based confidence intervals for rankings
- **GPU Support**: Automatically uses CUDA when available for faster computation
- **Probability Estimation**: Calculate likelihood of top-N items appearing in top-M positions

## Installation

Requires Python with PyTorch and tqdm:

```bash
pip install torch tqdm
```

## Usage

```python
import torch
from ranking_from_pairwise import maximise_log_likelihood_from_list_of_winner_loser_pairs

# Define pairwise comparisons as (winner_index, loser_index) tuples
winner_loser_pairs = [(0, 1), (2, 1), (0, 2)]  # Item 0 beats 1, Item 2 beats 1, etc.

# Initialize scores (will be optimized)
scores = torch.zeros(3, dtype=torch.float64)

# Optimize scores using maximum likelihood
optimized_scores = maximise_log_likelihood_from_list_of_winner_loser_pairs(
    winner_loser_pairs, scores
)

# Get ranking (indices sorted by score, highest first)
ranking = torch.argsort(optimized_scores, descending=True)
print("Ranking:", ranking.tolist())
```

## API Reference

### Core Functions

- `maximise_log_likelihood_from_list_of_winner_loser_pairs(pairs, scores)`: Optimizes scores using gradient ascent
- `compute_log_likelihood_from_list_of_winner_loser_pairs(pairs, scores)`: Computes likelihood of scores given data
- `compute_ranking_from_scores(scores)`: Converts scores to ranking indices
- `probability_of_first_n_in_first_m_calculated_from_scores_and_winner_loser_pairs(scores, pairs, n, m)`: Bayesian probability estimation

### Advanced Functions

- `compute_hessian_from_winner_loser_pairs_and_scores(pairs, scores)`: Computes Hessian matrix for uncertainty estimation
- `generate_normal_points_from_mean_and_covariance(mean, cov, num_points)`: Samples from multivariate normal distribution

## Algorithm Details

The library uses the Bradley-Terry model for pairwise comparisons:
- P(i beats j) = σ(score_i - score_j) where σ is the sigmoid function
- Scores are optimized to maximize the log-likelihood of observed outcomes
- Uncertainty is estimated using the inverse Hessian as the covariance matrix

## Example Output

```
Target Scores: tensor([0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000, 5.0000])
Ranking: tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
Probability of first 3 in first 5: 0.89
```

## Testing

Run the built-in test function:

```bash
python ranking_from_pairwise.py
```

This generates synthetic pairwise data and demonstrates the recovery of true rankings.