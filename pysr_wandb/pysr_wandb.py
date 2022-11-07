"""Main module."""
import wandb
from pysr import PySRRegressor
from .problems import *
from copy import deepcopy

PROCS = 8

UNCHANGED_PARAMS = dict(
    niterations=10000000,
    model_selection="best",
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "square", "cube"],
    max_evals=None,
    maxdepth=None,
    maxsize=30,
    constraints=None,
    nested_constraints=None,
    complexity_of_operators=None,
    complexity_of_constants=1,
    complexity_of_variables=1,
    loss="L2DistLoss()",
    use_frequency=True,
    use_frequency_in_tournament=True,
    early_stop_condition="early_stop(loss, complexity) = loss < 1e-15 && complexity < 20",
    skip_mutation_failures=True,
    migration=True,
    hof_migration=True,
    optimizer_algorithm="BFGS",
    should_optimize_constants=True,
    multithreading=True,
    update=False,
    progress=False,
    verbosity=0,
    warm_start=False,
    turbo=True,
    precision=32,
    fast_cycle=False,
    random_state=None,
    deterministic=False,
)

timeout_minutes = 1.0

DEFAULT_CONFIG = dict(
    timeout_in_seconds=timeout_minutes * 60,
    populations=15,
    population_size=33,
    warmup_maxsize_by=0.0,
    parsimony=0.0032,
    adaptive_parsimony_scaling=20.0,
    alpha=0.1,
    annealing=False,
    ncyclesperiteration=550,
    fraction_replaced=0.000364,
    fraction_replaced_hof=0.035,
    weight_add_node=0.79,
    weight_insert_node=5.1,
    weight_delete_node=1.7,
    weight_do_nothing=0.21,
    weight_mutate_constant=0.048,
    weight_mutate_operator=0.47,
    weight_randomize=0.00023,
    weight_simplify=0.0020,
    weight_optimize=0.0,
    crossover_probability=0.066,
    topn=12,
    optimizer_nrestarts=2,
    optimize_probability=0.14,
    optimizer_iterations=8,
    perturbation_factor=0.076,
    tournament_selection_n=10,
    tournament_selection_p=0.86,
    batching=False,
    batch_size=50,
)


def init_wandb():
    wandb.init(
        project="pysr_wandb",
        entity="mcranmer",
        config=DEFAULT_CONFIG,
    )
    return wandb


def run(wandb, problem, seed):
    X, y = problem(seed)
    model = PySRRegressor(**UNCHANGED_PARAMS)
    config = deepcopy(wandb.config)
    config["annealing"] = bool(config["annealing"])
    config["batching"] = bool(config["batching"])
    config["tournament_selection_n"] = int(config["tournament_selection_n"])
    config["optimizer_nrestarts"] = int(config["optimizer_nrestarts"])
    config["optimizer_iterations"] = int(config["optimizer_iterations"])
    config["topn"] = int(config["topn"])
    config["batch_size"] = int(config["batch_size"])
    config["ncyclesperiteration"] = int(config["ncyclesperiteration"])
    config["population_size"] = int(config["population_size"])
    config["populations"] = int(config["populations"])
    model.set_params(**wandb.config)
    model.set_params(procs=PROCS)
    model.fit(X, y)
    best_loss = model.get_best().loss
    best_complexity = model.get_best().complexity
    combined = best_loss**0.1

    problem_name = problem.__name__
    wandb.log(
        {
            f"loss_{problem_name}": best_loss,
            f"complexity_{problem_name}": best_complexity,
            f"combined_{problem_name}": combined,
        },
        step=seed,
    )
    wandb_table = wandb.Table(dataframe=model.equations_[["equation", "score", "loss", "complexity"]])
    wandb.log(
        {"equations_{problem_name}": wandb_table},
    )
    return combined


def runall(wandb):
    all_results = []
    for problem in PROBLEMS:
        combined = []
        for seed in range(5):
            combined.append(run(wandb, problem, seed))
        med = np.median(combined)
        wandb.log(
            {
                f"combined_{problem.__name__}": med,
            }
        )
        all_results.append(med)
    wandb.log(
        {
            "overall": np.median(all_results),
        }
    )


def main():
    wandb = init_wandb()
    runall(wandb)


if __name__ == "__main__":
    main()
