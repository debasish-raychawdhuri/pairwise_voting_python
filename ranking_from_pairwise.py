import torch
import tqdm
def compute_log_likelihood_from_list_of_winner_loser_pairs(
    winner_loser_pairs: list[tuple[int, int]],
    scores: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the log likelihood of the scores given the winner-loser pairs.

    Args:
        winner_loser_pairs (list[tuple[int, int]]): List of tuples where each tuple contains a winner and a loser index.
        scores (torch.Tensor): Tensor containing the scores for each item.

    Returns:
        torch.Tensor: Log likelihood of the scores given the winner-loser pairs.
    """
    log_likelihood = 0.0
    for winner, loser in winner_loser_pairs:
        log_likelihood += torch.log(torch.sigmoid(scores[winner] - scores[loser]))
    return log_likelihood

def maximise_log_likelihood_from_list_of_winner_loser_pairs(
    winner_loser_pairs: list[tuple[int, int]],
    scores: torch.Tensor,
) -> torch.Tensor:
    """
    Maximise the log likelihood of the scores given the winner-loser pairs.

    Args:
        winner_loser_pairs (list[tuple[int, int]]): List of tuples where each tuple contains a winner and a loser index.
        scores (torch.Tensor): Tensor containing the scores for each item.

    Returns:
        torch.Tensor: Updated scores after maximising the log likelihood.
    """
    # Try to use cuda if available
    if torch.cuda.is_available():
        scores = scores.cuda()
    else:
        scores = scores.cpu()
    num_iterations = 1000
    #torch.optim.LBFGS(scores, lr=0.1)  # Using LBFGS optimizer for better convergence
    with tqdm.tqdm(total=num_iterations, desc="Maximising Log Likelihood") as pbar:

        for _ in range(num_iterations):  # Number of iterations for convergence
            scores.requires_grad = True
            log_likelihood = compute_log_likelihood_from_list_of_winner_loser_pairs(
                winner_loser_pairs, scores
            )
            pbar.update(1)
            log_likelihood.backward()
            with torch.no_grad():
                # Update scores using the gradient
                scores += 0.01 * scores.grad  # Update scores with a small learning rate
                pbar.set_postfix({"Gradiant": scores.grad.norm().item(), "Log Likelihood": log_likelihood.item()})
    
            scores.grad.zero_()  # Reset gradients for the next iteration 
    return scores

def compute_hessian_from_winner_loser_pairs_and_scores(winner_loser_pairs: list[tuple[int, int]], scores: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hessian matrix from the winner-loser pairs and scores.

    Args:
        winner_loser_pairs (list[tuple[int, int]]): List of tuples where each tuple contains a winner and a loser index.
        scores (torch.Tensor): Tensor containing the scores for each item.

    Returns:
        torch.Tensor: Hessian matrix computed from the winner-loser pairs and scores.
    """
    
    hessian = torch.zeros((len(scores), len(scores)), dtype=scores.dtype, device=scores.device)
    for winner, loser in winner_loser_pairs:
        diff = scores[winner] - scores[loser]
        prob = torch.sigmoid(diff)
        dd = prob * (1 - prob) * (1 - 2*prob) # Second derivative of the sigmoid function
        hessian[winner, winner] += dd
        hessian[loser, loser] += dd
        hessian[winner, loser] -= dd
        hessian[loser, winner] -= dd
    # drop the first row and first column of the hessian matrix
    hessian = hessian[1:, 1:]
    return hessian


def compute_ranking_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Compute the ranking from the scores.

    Args:
        scores (torch.Tensor): Tensor containing the scores for each item.

    Returns:
        torch.Tensor: Indices of the items sorted by their scores in descending order.
    """
    return torch.argsort(scores, descending=True)

def generate_normal_points_from_mean_and_covariance(mean: torch.Tensor, covariance: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Generate points from a multivariate normal distribution given mean and covariance.

    Args:
        mean (torch.Tensor): Mean of the distribution.
        covariance (torch.Tensor): Covariance matrix of the distribution.
        num_points (int): Number of points to generate.

    Returns:
        torch.Tensor: Generated points from the multivariate normal distribution.
    """
    return torch.distributions.MultivariateNormal(mean, covariance).sample((num_points,))
def probability_of_first_n_in_first_m_calculated_from_scores_and_winner_loser_pairs( scores: torch.Tensor, winner_loser_pairs: list[tuple[int,int]], n: int, m: int) -> float:
    hessian = compute_hessian_from_winner_loser_pairs_and_scores(winner_loser_pairs, scores)
    covariance = -torch.linalg.inv(hessian)
    mean = scores[1:]  # Exclude the first score, which is assumed to be a constant or baseline score
    points = generate_normal_points_from_mean_and_covariance(mean, covariance, 100)
    #We had removed the first element from the scores, so we need to add it back to the points
    print("Generated Points Shape:", points.shape)
    original_ranking = compute_ranking_from_scores(scores)
    # Calculate the probabilities of the first n points being in the first m points

    matching = 0
    for point in points:
        point = torch.cat((torch.tensor(scores[0].unsqueeze(0), dtype=torch.float64), point))  # Add the first score back
        point_ranking = compute_ranking_from_scores(point)
        first_n_indices = point_ranking[:n]
        first_m_indices = original_ranking[:m]
        if all(idx in first_m_indices for idx in first_n_indices):
            matching += 1
    return matching / len(points)

def test():
    # Example usage
    # generate some random scores
    test_scores = torch.tensor([1.0,2.0,3.0, 4.0, 5.0,6,7,8,9,10], dtype=torch.float64)
    test_scores = test_scores * 0.5  # Random scores around 50
    print("Target Scores:", test_scores)
    winner_loser_pairs = []
    #generate winning and losing pairs from scores
    for i in range(200):
        p1 = torch.randint(0, len(test_scores), (1,)).item()
        p2 = torch.randint(0, len(test_scores), (1,)).item()
        if p1 == p2:
            continue
        prob = torch.sigmoid(test_scores[p1] - test_scores[p2])
        winner = p1 if torch.rand(1).item() < prob else p2
        loser = p2 if winner == p1 else p1
        winner_loser_pairs.append((winner, loser))

    print("Winner-Loser Pairs:", winner_loser_pairs[:10])  # Show first 10 pairs for brevity
    scores = torch.zeros(10, dtype=torch.float64)

    updated_scores = maximise_log_likelihood_from_list_of_winner_loser_pairs(
        winner_loser_pairs, scores
    )
    print("Updated Scores - Test Scores (Should be nearly constant):", updated_scores-test_scores)
    log_likelihood = compute_log_likelihood_from_list_of_winner_loser_pairs(
        winner_loser_pairs, scores
    )
    print("Log Likelihood:", log_likelihood.item())



    ranking = compute_ranking_from_scores(updated_scores)
    print("Ranking:", ranking)
    
    # Probability of first 3 in first 5 calculated from scores and winner-loser pairs
    probability = probability_of_first_n_in_first_m_calculated_from_scores_and_winner_loser_pairs(
        updated_scores, winner_loser_pairs, 3, 5
    )
    print("Probability of first 3 in first 5:", probability)

if __name__ == "__main__":
    test()
# This code is a self-contained module that computes rankings from pairwise comparisons.
