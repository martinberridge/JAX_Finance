---
layout: post
title: "What SABR Is For, and Why Automatic Differentiation Suits It"
date: 2026-11-26
categories: [quant-finance, autodiff]
series: autodiff-for-greeks
part: 5
---


This is the first of two posts on SABR in this series on automatic differentiation for Greeks. Here I cover the ideas: what SABR is for, how the model and Hagan's approximation work, and why automatic differentiation suits them. In the second post, I'll walk through my JAX implementation on an FX example.

# What SABR Is For, and Why Automatic Differentiation Suits It

The market only quotes implied vols at a handful of strikes and expiries, but the trades in a book rarely sit right on those points. To value and hedge them, we need a vol at any strike and expiry—and we need to know how those vols move when the market moves. 

SABR is the standard answer in rates: a stochastic volatility model with intuitive parameters and a fast closed-form approximation, Hagan's formula, for implied vol. In practice, using SABR means calibrating it: choosing its parameters so that the vols from Hagan's approximation match the market's quoted vols as closely as possible. Below, I explain both the underlying model and the approximation, look at why SABR became a market standard, and show why its smooth closed-form approximation is a natural fit for automatic differentiation.




## Motivation: why we need a model for the smile

I'll use FX options as the running example, where the gap between what's quoted and what's needed is especially stark. The FX market quotes each expiry's smile at just five points, and in delta rather than strike: ATM, plus $25\Delta$ and $10\Delta$ risk reversals and butterflies. Unquoted vols come up in several ways:

- **Strikes rather than deltas.** Trades are struck at a specific level, such as a corporate hedging at its budget rate, but the market quotes vols by delta. Converting a strike to a delta itself depends on the vol, so we need a smile model that gives a Black implied vol $\sigma_B$ at any strike, consistent with the delta quotes.
- **Seasoned trades.** A trade that was ATM or $25\Delta$ when booked drifts away from that delta as spot moves.
- **Broken dates.** Trades expire on specific dates that fall between quoted pillars (the standard expiries, such as 1M, 3M or 1Y, where the market quotes vols directly). A smile model like SABR works one expiry at a time, so this is handled by interpolating between the pillar smiles, typically by interpolating the model's parameters.
- **The wings.** Quotes stop at $10\Delta$, yet we still need vols for deeper out-of-the-money strikes, for barrier-style risk and for stress scenarios.
- **Risk and scenarios.** When we bump spot to compute delta, or shift it in a stress scenario, a trade's strike stays fixed but its delta changes. Its vol now has to be read at a point on the smile that wasn't quoted.
- **Unreliable quotes.** Risk reversals and butterflies in less liquid pairs or long expiries can be stale or wide, and are better fitted through than used as given.

The same needs arise in rates, where swaption smiles are quoted at strike offsets from ATM across a grid of expiries and tenors. That's where SABR became the market standard.

The vols we fill in also need to be smooth and free of arbitrage, because prices and Greeks depend on the shape of the smile, not just its level:

- **Consistent marks.** A kinked curve can mark nearby strikes at noticeably different vols for no economic reason.
- **Stable Greeks.** A smile-adjusted delta is the Black delta plus a correction for the implied vol at the trade's strike changing as the forward moves: vega multiplied by $\partial \sigma_B / \partial F$. Because that correction depends on the shape of the smile, kinks in the smile make hedges unstable.
- **No arbitrage.** The risk-neutral density (the probability distribution of the underlying at expiry implied by option prices) is $\partial^2 C / \partial K^2$, and a non-smooth smile can make it negative. A negative density means a butterfly spread, whose payoff can never be negative, would have a negative price: you'd be paid to take a position that can't lose, which is an arbitrage.
- **Clean P&L.** A jagged surface turns small changes in quotes into erratic marks and noisy P&L.

Pure interpolation can deliver smoothness, but it says nothing about how the smile moves when the forward moves, and that dynamic determines your hedges. That's the gap a model like SABR fills.

## What SABR is

SABR (stochastic alpha, beta, rho) models the forward $F$ and its volatility $\sigma$ as two correlated processes:

$$
dF_t = \sigma_t F_t^{\beta}\, dW_t, \qquad d\sigma_t = \nu\, \sigma_t\, dZ_t, \qquad dW_t\, dZ_t = \rho\, dt, \qquad \sigma_0 = \alpha
$$

Dropping the $t$ subscript in the prose below for readability, each piece has a simple intuition:

- **The forward process, $dF = \sigma F^{\beta}\, dW$.** The forward has no drift: any expected movement, like the interest rate differential in FX, is already priced into today's forward, so on average it's expected to stay where it is. A "move" here is the change in the forward, $dF$, over a short interval of time $dt$. It's random, with a typical size (its standard deviation) of $\sigma F^{\beta} \sqrt{dt}$, so $\sigma F^{\beta}$ sets how big the forward's moves are. With $\beta = 1$, moves scale with the level of the forward, as with percentage moves in an exchange rate. With $\beta = 0$, moves are a fixed size whatever the level, as with basis-point moves in a rate.
- **The vol process, $d\sigma = \nu\, \sigma\, dZ$.** Volatility isn't constant. It moves around randomly too. Here a move is the change in vol, $d\sigma$, over a short interval $dt$, with a typical size of $\nu \sigma \sqrt{dt}$. Because that's proportional to vol's own level, vol stays positive and quiet markets stay relatively quiet. $\nu$ sets how restless it is. With $\nu = 0$, vol never changes and we're back to a simple local-vol (CEV) model. As $\nu$ increases, the forward goes through calm spells and turbulent spells, which produces the fat tails that lift the wings of the smile.
- **The correlation, $dW\, dZ = \rho\, dt$.** This links the two processes. Negative $\rho$ means vol tends to rise when the forward falls, so downside moves happen in a more volatile market, making downside strikes more expensive. Positive $\rho$ does the opposite. That asymmetry is the skew.

Each of the four parameters has a clear job:

- **$\alpha$ (alpha), the vol level.** The initial volatility, which mainly sets the level of the smile, and so the ATM vol.
- **$\beta$ (beta), the backbone.** In the Hagan paper [1], the backbone is the path the ATM vol traces as the forward moves. Ignoring small correction terms, $\sigma_{\text{ATM}}$ (the implied vol at the money, where $K = F$) $\approx \alpha / F^{1-\beta}$, so the way $\beta$ scales the forward's moves feeds directly into how ATM vol moves: flat for $\beta = 1$, falling as the forward rises for $\beta < 1$. $\beta$ is usually fixed rather than fitted, because a single day's smile can be fitted almost equally well with different values of $\beta$, with $\rho$ adjusting to compensate. Each $\beta$ still implies different smile dynamics, and so a different delta, which is why the choice matters for risk. In FX, $\beta = 1$ is a common choice.
- **$\rho$ (rho), the skew.** Tilts the smile, and shows up in the risk reversals.
- **$\nu$ (nu), the curvature.** Lifts the wings, and shows up in the butterflies.

This mapping to the FX quotes ($\alpha$ to ATM, $\rho$ to risk reversals, $\nu$ to butterflies) is a useful rule of thumb rather than an exact one-to-one correspondence: each parameter moves all three quotes to some degree.

The key to SABR's practicality is Hagan's approximation [1], an asymptotic expansion that gives the Black implied vol $\sigma_B(K, F, T; \alpha, \beta, \rho, \nu)$ as an explicit formula. To price a vanilla, we compute $\sigma_B$ at the trade's strike and plug it into Black's formula. Two caveats are worth keeping in mind:

- **It's an approximation.** Accuracy degrades for long expiries, high vol of vol and low strikes, where it can even imply a negative density in the wings. Refinements and variants such as shifted SABR exist for these cases.
- **It works one expiry at a time.** Parameters are calibrated separately at each pillar, and broken dates are handled by interpolating them between pillars. SABR describes the smile at an expiry, not the whole surface.

## Why it became the standard

SABR was introduced by Hagan, Kumar, Lesniewski and Woodward in their 2002 paper *Managing Smile Risk* [1]. Several things made it the market standard in rates:

- **It gets the smile dynamics right.** The paper's central point was that local volatility models predict the smile moving in the opposite direction to the forward, which is the wrong way compared with what the market shows, and so they give poor hedges. SABR predicts the smile moving with the forward, giving more realistic deltas.
- **It has a closed-form approximation.** Hagan's formula gives implied vol directly from the parameters, with no PDE or Monte Carlo needed. That makes calibration and pricing fast enough to run across a whole book.
- **Its parameters are intuitive.** Each one has a clear role, so traders can read and reason about a fitted smile, and even mark it by hand.
- **It fits the way rates are quoted.** SABR is fitted one expiry at a time, which maps naturally onto the swaption cube, organised by expiry, tenor and strike.
- **It became a shared language.** Once desks and brokers widely adopted it, SABR parameters became a common way to describe and communicate smiles.

Its use isn't limited to rates. In FX, SABR is a popular choice for fitting each expiry's smile to the delta quotes, because its parameters line up naturally with them: $\alpha$ sets the ATM level, $\rho$ drives the risk reversal (skew) and $\nu$ drives the butterfly (curvature). It isn't the only approach, though. Vanna-volga, which builds the smile from the cost of hedging vega, vanna and volga with the three quoted instruments, is also widely used, especially for pricing first-generation exotics. SABR also appears in inflation and commodity options, although equity desks tend to prefer other approaches for full surfaces.

## The risk problem

Fitting the smile is only half the job. A risk system needs sensitivities, and with SABR they come in several layers. Throughout, $V$ is the value of the option (or of the position): the Black price evaluated at the SABR implied vol, $V = V_{\text{Black}}(F, K, T, \sigma_B(K, F, T; \alpha, \beta, \rho, \nu))$.

**Model-consistent delta.** In a smile model, the vol at a fixed strike changes when the forward moves. A naive Black delta, which holds $\sigma_B$ fixed, misses this. The SABR delta includes it:

$$
\Delta_{\text{SABR}} = \frac{\partial V}{\partial F} + \frac{\partial V}{\partial \sigma_B}\,\frac{\partial \sigma_B}{\partial F}
$$

This is the backbone effect from the 2002 paper [1] in action: getting the smile dynamics right matters precisely because it changes the hedge. Bartlett's refinement [2] goes further, adding the move in $\alpha$ expected through the correlation $\rho$:

$$
\Delta_{\text{Bartlett}} = \frac{\partial V}{\partial F} + \frac{\partial V}{\partial \sigma_B}\left(\frac{\partial \sigma_B}{\partial F} + \frac{\rho\,\nu}{F^{\beta}}\,\frac{\partial \sigma_B}{\partial \alpha}\right)
$$

**Parameter sensitivities.** Traders also want to know how a position responds to each parameter: $\partial V / \partial \alpha$ is close to a vega, while $\partial V / \partial \rho$ and $\partial V / \partial \nu$ are the SABR analogues of vanna and volga, the skew and curvature risks. In FX they map onto the quoted instruments: $\rho$ risk is naturally hedged with risk reversals, and $\nu$ risk with butterflies.

**Sensitivities to market quotes.** What a trader actually hedges with are the quoted instruments: ATM, risk reversals and butterflies. Getting there means differentiating through the calibration itself. With $\beta$ held fixed, if $\theta = (\alpha, \rho, \nu)$ are the calibrated parameters and $q$ the quotes, then:

$$
\frac{dV}{dq} = \frac{\partial V}{\partial \theta}\,\frac{d\theta}{dq}
$$

The awkward part is $d\theta / dq$: how the fitted parameters move when a quote moves.

**Second-order risk.** Gamma, vanna and volga need second derivatives, through the formula and often through the calibration too.

The traditional approach to all of this is bumping: shift an input, recalibrate, reprice, and take a finite difference. It works, but it has real costs:

- **It's expensive.** Every bump of a market quote needs a full recalibration, for every expiry and every risk factor.
- **It's noisy.** The result depends on the bump size. Too large and you pick up curvature; too small and you pick up rounding error and optimiser tolerance.
- **Second order is worse.** Cross-gammas and second-order Greeks multiply the number of bumps, and the noise.

## Why AD fits

Automatic differentiation computes derivatives by applying the chain rule to every elementary operation in the code. It isn't a numerical approximation like bumping, nor symbolic manipulation of formulas: it gives derivatives exact to machine precision, at a cost that's a small multiple of evaluating the function itself. Hagan's formula is close to an ideal case for it.

- **It's a smooth closed form.** Hagan's approximation is built from powers, logs and square roots, with no PDE grid or Monte Carlo paths to differentiate through. AD gives exact derivatives of it, with no bump size to choose and no noise.
- **It gives all the sensitivities at once.** Reverse-mode AD returns the gradient with respect to every input ($F$, $K$, $T$, $\alpha$, $\beta$, $\rho$, $\nu$) in one backward pass. The SABR and Bartlett deltas and every parameter sensitivity come out together.
- **Second order comes cheaply.** In a framework like JAX, gamma, vanna and volga are just derivatives of derivatives, using `jax.grad` twice or `jax.hessian`, with no extra bumping schemes.
- **It speeds up calibration itself.** Optimisers such as Levenberg–Marquardt need the Jacobian of the fit errors with respect to the parameters. AD gives that exactly, which typically means faster and more reliable convergence than finite-difference Jacobians.
- **It differentiates through calibration.** Rather than unrolling the optimiser, we can use the implicit function theorem at the solution. If the calibration solves $r(\theta, q) = 0$, then $\frac{d\theta}{dq} = -\left(\frac{\partial r}{\partial \theta}\right)^{-1}\frac{\partial r}{\partial q}$, and AD supplies both Jacobians. That turns quote sensitivities, the hardest layer of SABR risk, into a single linear solve rather than one recalibration per bump.

There are a few practical edge cases to watch out for. AD gives exact derivatives of Hagan's approximation—not of the true SABR process—so the caveats about Hagan's formula carry over to its Greeks, and $\beta$ is typically held fixed so that $\theta = (\alpha, \rho, \nu)$ remains the calibrated parameter vector. The formula also has a removable singularity at the money ($K = F$), where a naive implementation evaluates $0/0$. The implied vol is well defined there, but AD will propagate `NaN`s unless the ATM case is handled carefully.

> **Aside: is calibration just training a neural network?**
>
> Readers from machine learning may notice that calibrating SABR looks a lot like training a multilayer perceptron (MLP). The analogy holds up to a point.
>
> What's similar:
>
> - **Both minimise a loss.** Typically a sum of squared errors between model outputs and targets.
> - **Both use gradients from AD.** JAX treats a SABR loss and an MLP loss in exactly the same way.
> - **Both are non-convex.** The optimiser can end up in a local minimum depending on where it starts.
>
> What's different:
>
> - **Scale.** With $\beta$ fixed, SABR has three free parameters ($\alpha$, $\rho$, $\nu$) fitted to around five quotes per expiry. An MLP can have thousands to billions of parameters fitted to large datasets.
> - **Optimiser.** A small least-squares problem suits Levenberg–Marquardt or Gauss–Newton, which use the full Jacobian and converge in a handful of iterations. MLPs use stochastic gradient methods like Adam on mini-batches, because the full problem is too large.
> - **Constraints.** SABR parameters have hard bounds ($\alpha > 0$, $\nu \ge 0$, $-1 < \rho < 1$), usually handled by reparameterising or by a bounded optimiser. MLP weights are unconstrained.
> - **Interpretability.** Each SABR parameter has a financial meaning, and the fit is close to interpolating the quotes. MLP weights have no individual meaning, and the concern is how well the network generalises beyond its training data.
> - **Identifiability.** With $\beta$ fixed, the quotes usually pin down SABR's parameters well. MLPs are heavily overparameterised, so many different weight settings fit equally well.
>
> In short, both are gradient-based fits, but SABR calibration is a small, constrained, interpretable problem, closer to curve fitting than to training a network.

In the next post, I'll move from theory to code and show how this looks in JAX on an FX example: writing Hagan's formula as a pure function with safe ATM handling, calibrating $(\alpha, \rho, \nu)$ to delta-quoted smiles, and computing both direct Greeks and implicit-differentiation quote sensitivities with automatic differentiation.

## References

1. Hagan, P. S., Kumar, D., Lesniewski, A. S. and Woodward, D. E. (2002). Managing Smile Risk. *Wilmott Magazine*, September 2002, pp. 84–108.
2. Bartlett, B. (2006). Hedging under SABR Model. *Wilmott Magazine*, pp. 2–4.