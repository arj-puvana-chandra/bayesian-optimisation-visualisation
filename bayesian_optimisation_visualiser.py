# Bayesian Optimisation Visualiser
# Author: Arj Puvanachandra using Claude Sonnet 5.5
# Date: 24/11/2026

"""
This is code to visualise the process of BO. It was written by Claude Code and edited by myself
to fit the functions I'd like.

basic process:
- random 3 points are sampled for initialisation
- the GP is fitted to the data and the UCB acquisition function is computed
- UCB is maximised to select the next point to sample
- iterations (here a fixed number, but usually a termination condition is used)

- things you can change (placed just below this introduction block):
    A. user_n_steps
        - the number of BO iterations to run
        - longer runs shows more of the optimisation path
    B. user_lambda
        - the exploration vs exploitation weighting in the UCB acquisition function
        - higher lambda = more exploration, lower lambda = more exploitation
    C. user_n_init
        - the number of random initial observations to sample
        - higher n_init = more initial data, lower n_init = less initial data
        - these give an initialisation to the GP and therefore affect the optimisation path
    D. user_seed
        - the random seed for reproducible results
        - change to get different initial points and therefore different optimisation paths
    E. user_function
        - the fitness landscape to optimise (1-4) (getting more rugged)
        - 1 = sin(2.5x) + 0.6sin(6x) + 0.15x
        - 2 = sin(2.5x) + 0.6sin(6x) + 0.3cos(11x) + 0.15x
        - 3 = sin(2.5x) + 0.6sin(6x) + 0.3cos(11x) + 0.1sin(20x) + 0.15x
        - 4 = sin(2.5x) + 0.6sin(6x) + 0.3cos(11x) + 0.1sin(20x) + 0.05cos(30x) + 0.15x
"""

# Change these to see how the optimisation path changes
user_n_steps = 10 # number of BO iterations to run
user_lambda = 3 # explortaion vs exploitation weighting
user_n_init = 3 # number of random initial observations
user_seed = 42   # random seed for reproducible results
user_function = 2 # fitness landscape to optimise (1-4)



# main script from Claude Code
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


# Objective: different fitness landscape (the "unknown" function being optimised)
def true_function(x):
    if user_function == 1:
        return (
            np.sin(2.5 * x)
            + 0.6 * np.sin(6 * x)
            + 0.15 * x
        )
    elif user_function == 2:
        return (
            np.sin(2.5 * x)
            + 0.6 * np.sin(6 * x)
            + 0.3 * np.cos(11 * x)
            + 0.15 * x
        )
    elif user_function == 3:
        return (
            np.sin(2.5 * x)
            + 0.6 * np.sin(6 * x)
            + 0.3 * np.cos(11 * x)
            + 0.1 * np.sin(20 * x)
            + 0.15 * x
        )
    elif user_function == 4:
        return (
            np.sin(2.5 * x)
            + 0.6 * np.sin(6 * x)
            + 0.3 * np.cos(11 * x)
            + 0.1 * np.sin(20 * x)
            + 0.05 * np.cos(30 * x)
            + 0.15 * x
        )
    else:
        raise ValueError("Invalid user_function value: CHOOSE 1,2,3, or 4")


# Gaussian Process surrogate
#   ConstantKernel -> prior variance = 1.0
#   RBF            -> nearby points have similar values (smoothness);
#                     length_scale = distance over which points are strongly
#                     correlated (larger = smoother)
# Hyperparameters are fixed.

def make_gp():
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=0.4, length_scale_bounds="fixed")
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,       # tiny jitter: "noiseless" but numerically stable
        optimizer=None,    # keep hyperparameters fixed
        normalize_y=False, # retains the zero prior mean assumption
    )


# Bayesian optimisation loop (run up front; the UI just replays the history)
def run_bayes_opt(X, n_steps, lamb, n_init, seed):
    gp = make_gp()

    # Random initial design: n_init distinct grid points, reproducible via seed
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(X), size=n_init, replace=False)
    X_train = X[init_idx].copy()
    y_train = true_function(X_train.ravel())

    # Grid indices that have already been evaluated (excluded from selection)
    sampled_idx = {int(i) for i in init_idx}

    history = []

    for _ in range(n_steps):
        gp.fit(X_train, y_train)
        mu, sigma = gp.predict(X, return_std=True)

        # Upper Confidence Bound acquisition function
        ucb = mu + lamb * sigma
        ucb[list(sampled_idx)] = -np.inf  # don't re-select observed points

        max_idx = int(np.argmax(ucb))
        x_next = X[max_idx, 0]

        # Save this iteration BEFORE sampling the new point
        history.append({
            "X_train": X_train.copy(),
            "y_train": y_train.copy(),
            "mu": mu.copy(),
            "sigma": sigma.copy(),
            "ucb": ucb.copy(),
            "x_next": x_next,
            "ucb_max": ucb[max_idx],
        })

        # Evaluate the true fitness at the chosen point and update the data
        y_next = true_function(x_next)
        X_train = np.vstack([X_train, [[x_next]]])
        y_train = np.append(y_train, y_next)
        sampled_idx.add(max_idx)

    return history


def main():
    # Settings
    n_steps = user_n_steps # number of BO iterations to run
    lamb = user_lambda  # exploration weight; matches the ±2σ band shown on the plot
    n_init = user_n_init  # number of random initial observations
    seed = user_seed    # change for a different random starting design
    X = np.linspace(0, 6, 500).reshape(-1, 1)

    history = run_bayes_opt(X, n_steps, lamb, n_init, seed)

    # Fixed axis limits so the plots don't jump when moving the slider
    y_true = true_function(X.ravel())
    x_opt = X.ravel()[np.argmax(y_true)]
    y_pad = 0.3 * (y_true.max() - y_true.min())
    fitness_ylim = (y_true.min() - y_pad, y_true.max() + y_pad)

    all_ucb = np.concatenate([h["ucb"][np.isfinite(h["ucb"])] for h in history])
    ucb_pad = 0.05 * (all_ucb.max() - all_ucb.min())
    ucb_ylim = (all_ucb.min() - ucb_pad, all_ucb.max() + ucb_pad)

    # Interactive figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(bottom=0.36)
    fig.suptitle(
        "Bayesian Optimisation Visualisation Tool - by Arj Puvanachandra"
    )

    # NOTE: keep references to the widgets (slider/buttons) for the lifetime of
    # the figure, otherwise matplotlib may garbage collect them and they stop
    # responding.
    slider_ax = plt.axes([0.25, 0.09, 0.50, 0.03])
    step_slider = Slider(slider_ax, "Iteration", 1, n_steps, valinit=1, valstep=1)

    prev_ax = plt.axes([0.25, 0.025, 0.12, 0.05])
    next_ax = plt.axes([0.63, 0.025, 0.12, 0.05])
    prev_button = Button(prev_ax, "Previous")
    next_button = Button(next_ax, "Next")

    def update_plot(step):
        data = history[int(step) - 1]
        mu, sigma, ucb = data["mu"], data["sigma"], data["ucb"]
        x_next = data["x_next"]
        X_train, y_train = data["X_train"], data["y_train"]

        for ax in axes:
            ax.clear()

        # Left: GP posterior
        ax = axes[0]
        ax.plot(X.ravel(), y_true, "k--", alpha=0.5, label="True fitness")
        ax.axvline(x_opt, color="gold", lw=1.5, alpha=0.8, label="True optimum")
        ax.plot(X.ravel(), mu, label="GP mean")
        ax.fill_between(
            X.ravel(), mu - 2 * sigma, mu + 2 * sigma, alpha=0.2, label="±2σ"
        )
        ax.scatter(
            X_train.ravel(), y_train, color="red", zorder=5, label="Observed points"
        )
        best = int(np.argmax(y_train))
        ax.scatter(
            X_train[best], y_train[best], marker="*", s=220,
            color="orange", edgecolor="k", zorder=6, label="Best so far",
        )
        ax.axvline(
            x_next, color="green", linestyle="--",
            label="Next sample (not yet evaluated)",
        )
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(*fitness_ylim)
        ax.set_title(f"GP Posterior - {len(X_train)} observations")
        ax.set_xlabel("x")
        ax.set_ylabel("Fitness")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8
        )

        # Right: UCB acquisition function
        ax = axes[1]
        ax.plot(X.ravel(), np.where(np.isfinite(ucb), ucb, np.nan), label="UCB")
        ax.scatter(
            x_next, data["ucb_max"], color="red", s=60, zorder=5,
            label="Maximum UCB",
        )
        ax.axvline(x_next, color="green", linestyle="--")
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(*ucb_ylim)
        ax.set_title(f"UCB Acquisition (lambda={lamb})")
        ax.set_xlabel("x")
        ax.set_ylabel("a(x)")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8
        )

        fig.canvas.draw_idle()

    # Connect controls
    step_slider.on_changed(update_plot)

    def previous(event):
        step_slider.set_val(max(1, step_slider.val - 1))

    def next_step(event):
        step_slider.set_val(min(n_steps, step_slider.val + 1))

    prev_button.on_clicked(previous)
    next_button.on_clicked(next_step)

    # Initial display (slider's valinit does not fire on_changed)
    update_plot(1)
    plt.show()


if __name__ == "__main__":
    main()