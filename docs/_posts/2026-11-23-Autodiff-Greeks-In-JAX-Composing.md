---
layout: post
title: "Autodiff Greeks in JAX: composing transformations, and where it goes wrong"
date: 2026-11-23
categories: [quant-finance, autodiff]
series: autodiff-for-greeks
part: 4
---
[Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) printed the program that `jax.grad` writes for Black-76, checked its answers against [Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %})'s closed forms, and showed what the XLA compiler does to it. This post builds on that program. Composing JAX's transformations gives the second-order Greeks and whole strike ladders from the same few lines. Then we cover the gotchas that matter when pricing with JAX.

The running example is unchanged: the `black_price` function from [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}), for a three-month EUR/USD option with $F = 1.0850$, $K = 1.0800$, $T = 0.25$, $r_d = 5\%$ and $\sigma = 10\%$, in float64.

## Transformations are functions of functions

Every JAX transformation takes a function and returns a function. `grad(f)`, `jit(f)`, `vmap(f)` and `hessian(f)` are ordinary Python functions that JAX can trace again, so the transformations nest in any order. This post uses five compositions: `grad` of `grad` for gamma, forward mode over reverse mode for the whole second-order block, `jacfwd` over `vmap` for one rate's effect on a book of options, `vmap` of an already-differentiated function for a strike ladder, and `jit` around all of it.

## Second-order Greeks

### The maths

The first-order Greeks are the gradient of the price. The second-order Greeks are its **Hessian**, the matrix of second derivatives. For the two inputs an FX options book cares about most, the forward and the vol, it's a 2×2 matrix.

The Hessian isn't a new kind of object. [Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %}) defined the Jacobian as the matrix of partial derivatives of a function's outputs with respect to its inputs. For a pricing function, which has one output, the Jacobian is a single row: the gradient, the first-order Greeks. Now treat that gradient as a function in its own right. It takes the same inputs as the price, but it returns a vector, one Greek per input. Its Jacobian is a square matrix: row $i$ says how the $i$th Greek moves when each input moves. That matrix is the Hessian:

$$H = J(\nabla c), \qquad H_{ij} = \frac{\partial}{\partial x_j}\left(\frac{\partial c}{\partial x_i}\right)$$

So the Hessian is the Jacobian of the gradient. Row 1 of the 2×2 block below is the gradient of delta: how delta moves with the forward (gamma) and with the vol (vanna). Row 2 does the same for vega. This is also how JAX builds it: `jax.hessian` is literally a Jacobian (`jacfwd`) of a gradient (`jacrev`), as the next section shows.

Here it is for the forward and the vol:

$$
H = \begin{pmatrix}
\dfrac{\partial^2 c}{\partial F^2} & \dfrac{\partial^2 c}{\partial F\,\partial\sigma} \\
\dfrac{\partial^2 c}{\partial \sigma\,\partial F} & \dfrac{\partial^2 c}{\partial \sigma^2}
\end{pmatrix}
= \begin{pmatrix} \Gamma & \text{vanna} \\ \text{vanna} & \text{volga} \end{pmatrix}
$$

The matrix is symmetric, because mixed partial derivatives don't depend on the order you take them in. Its three entries are:

- **Gamma**, $\Gamma = \partial^2 c/\partial F^2 = e^{-rT}n(d_1)/(F\sigma\sqrt{T})$: how fast delta changes as the forward moves. It decides how much a delta hedge has to be adjusted, and whether rebalancing makes or loses money.
- **Vanna**, $\partial^2 c/\partial F\,\partial\sigma = -e^{-rT}n(d_1)\,d_2/\sigma$: how delta changes when vol moves, or equivalently how vega changes when the forward moves.
- **Volga** (or vomma), $\partial^2 c/\partial\sigma^2 = \text{vega}\cdot d_1 d_2/\sigma$: how vega changes when vol moves.

Vanna and volga matter more in FX than almost anywhere else, because the market's standard smile adjustment, the vanna-volga method, prices an option from the cost of hedging exactly these two exposures.

### Differentiating a derivative: gamma

```python
gamma = grad(grad(black_price, argnums=0), argnums=0)(F, K, T, r_d, sigma, is_call)
```

`grad(black_price)` is an ordinary JAX function. It happens to contain a forward sweep and a backward sweep, and JAX can trace it and differentiate *that*, just like any other function. No new machinery is involved: the same linearise-and-transpose steps from [Part 2]({{ site.baseurl }}{% post_url 2026-11-16-Inside-The-Autodiff-Engine %}) are applied to a program that already contains a backward sweep. The result, 7.21257, matches the closed form, as the check below shows.

### The whole block: `jax.hessian`

Two refinements are worth knowing about.

First, `grad(grad(...))` is **reverse-over-reverse**. For a Hessian, the usually preferred pattern is **forward-over-reverse**: take the gradient in reverse mode, then differentiate the gradient function in forward mode. For $n$ inputs that's $n$ forward passes through the gradient program, one per column of the Hessian, and none of them needs to store the residuals of a backward sweep in order to run another backward sweep over it. `jax.hessian` is defined exactly this way, as `jacfwd(jacrev(f))`.

Second, you rarely want gamma alone. Differentiate with respect to both $F$ and $\sigma$ and you get the whole second-order risk block in one call:

```python
H = jax.hessian(lambda x: black_price(x[0], K, T, r_d, x[1]))(jnp.array([F, sigma]))
# [[ 7.2126, -0.2636],     [[ gamma, vanna ],
#  [-0.2636,  0.0168]]  =   [ vanna, volga ]]
```

Gamma, vanna and volga all come from one line.

### Checking the second-order Greeks

[Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) checked the first-order Greeks. The same check for the second order evaluates the closed forms next to the Hessian:

```python
sqrt_T = jnp.sqrt(T)
d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
d2 = d1 - sigma * sqrt_T
df = jnp.exp(-r_d * T)
n1 = norm.pdf(d1)
vega = df * F * sqrt_T * n1

gamma_cf = df * n1 / (F * sigma * sqrt_T)
vanna_cf = -df * n1 * d2 / sigma
volga_cf = vega * d1 * d2 / sigma
```

| Quantity | JAX autodiff | Closed form |
|---|---|---|
| Gamma | 7.2125664738 | 7.2125664738 |
| Vanna | −0.2636413934 | −0.2636413934 |
| Volga | 0.0167881731 | 0.0167881731 |

Agreement is again to every printed digit.

## Forward or reverse: the shape rule as an API

JAX offers both `jax.jacfwd` and `jax.jacrev` to compute a full Jacobian. `jacfwd` runs one JVP per input (column by column), and `jacrev` runs one VJP per output (row by row). This is [Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %})'s shape rule turned into an API.

One price from five inputs is a 1×5 Jacobian: one reverse pass versus five forward passes, so use `jacrev` (or `grad`). The opposite case is one input moving many prices. Here a single rate moves a book of five options at different strikes:

```python
strikes = jnp.linspace(1.06, 1.10, 5)
book = lambda r: vmap(lambda k: black_price(F, k, T, r, sigma))(strikes)

jax.jacfwd(book)(r_d)
# [-0.00893117 -0.00736196 -0.00597054 -0.00476058 -0.00372964]
```

That's a 5×1 Jacobian: each option's rho, equal to $-T\cdot c$ at its strike, from a single forward pass. `jacrev` would need five reverse passes, one per option, and for a book of 1,000 options it would need 1,000.

## The strike ladder: `vmap`

A **batch** is a set of independent calculations of the same kind, stacked into arrays so they can be done together: here, one option per strike, with each input held as an array that has one entry per option. The **batch axis** is the array dimension that indexes those options. For a ladder of five strikes, `strikes` has shape `(5,)` and its only axis, axis 0, is the batch axis: `strikes[i]` belongs to the `i`th option. Every result comes back with the same axis, so `delta[i]` is that option's delta.

`jax.vmap(fun, in_axes=0)` takes a function written for a single example, such as one option at one strike, and returns a function that runs it over a batch, such as the whole ladder.¹ It doesn't loop over the options. It traces the function once and rewrites each primitive to act along the batch axis: the `log` that took one $F/K$ now takes all five, one per entry of axis 0. The result is a single program of array operations. `in_axes` says, for each positional argument, which axis is its batch axis. For `strikes` that's axis 0, where `strikes[i]` belongs to option `i`. `None` marks an argument with no batch axis, a single value shared by every option, like the forward in a strike ladder. The outputs get the batch axis too, so `delta[i]` lines up with `strikes[i]`. If the function returns a dict, every value in it gets the batch axis, so a dict of scalars comes back as a dict of arrays.

`vmap` composes with the other transformations, and the accompanying code uses the most striking composition. `batch_price_and_greeks` doesn't vectorize `black_price` and then differentiate. It vectorizes the *already-differentiated* function:

```python
batch_fn = vmap(lambda f, k, t, r, s: compute_greeks_single(f, k, t, r, s, is_call))
```

`vmap` takes the jaxpr of `compute_greeks_single`, which contains the forward sweep, the backward sweep and the nested gamma calculation, and rewrites every primitive in it to operate on a batch dimension. A strike ladder is a classic use: pricing and risking the same option at a row of strikes, with all the other market data held fixed. You write the single-option Greeks once, and the whole ladder comes out.

[`batch_price_and_greeks`](https://github.com/martinberridge/JAX_Finance/blob/main/jax_finance/black.py) maps over all five inputs, so the market data that's shared by every strike is broadcast to the ladder's length.

To **broadcast** a value is to stretch it to the shape of a batch, so that every element of the batch sees the same value. Here the forward, expiry, rate and vol are single numbers, but `batch_price_and_greeks` expects one entry per option on every input. `jnp.full(n, F)` builds an array of length `n` with $F$ in every slot, so option `i` gets `F[i] = 1.0850`, the same as every other option, alongside its own `strikes[i]`. (NumPy and JAX also broadcast automatically in arithmetic: `F * strikes` treats the scalar `F` as if it were repeated five times without ever making the copies. `in_axes=None`, below, gets the same effect for `vmap`.)

```python
from jax_finance.black import batch_price_and_greeks

F, T, r_d, sigma = 1.0850, 0.25, 0.05, 0.10
strikes = jnp.linspace(1.06, 1.10, 5)        # 1.06, 1.07, ..., 1.10
n = strikes.shape[0]

ladder = batch_price_and_greeks(
    jnp.full(n, F), strikes, jnp.full(n, T), jnp.full(n, r_d), jnp.full(n, sigma), "call"
)
# ladder is a dict of arrays: ladder["delta"][i] is the delta at strikes[i]
```

To vary only the strike, skip the broadcasting and use `in_axes`, with `None` for the shared inputs:

```python
ladder = vmap(compute_greeks_single, in_axes=(None, 0, None, None, None, None))(
    F, strikes, T, r_d, sigma, True
)
```

Both give the same ladder:

| Strike | Price | Delta | Gamma | Vega (1%) | Theta (1 day) |
|---|---|---|---|---|---|
| 1.0600 | 0.035725 | 0.679814 | 6.437012 | 0.001894 | −0.000099 |
| 1.0700 | 0.029448 | 0.611526 | 6.935685 | 0.002041 | −0.000108 |
| 1.0800 | 0.023882 | 0.539929 | 7.212566 | 0.002123 | −0.000113 |
| 1.0900 | 0.019042 | 0.467430 | 7.246165 | 0.002133 | −0.000114 |
| 1.1000 | 0.014919 | 0.396460 | 7.039680 | 0.002072 | −0.000111 |

Delta falls through 0.5 near the money, and gamma and vega peak just above the forward, as they should. Wrap that in `jit`, as the code does, and the whole ladder becomes one fused kernel of the kind [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) described.

¹ `vmap` looks like `np.vectorize`, but they're very different underneath. `np.vectorize` is a convenience wrapper around a Python `for` loop (NumPy's docs say it's "provided primarily for convenience, not for performance"), so it calls your function once per element. If you like its interface, `jnp.vectorize` has the same signature and is built on `vmap`.

## Static versus traced arguments

`batch_price_and_greeks` is decorated with

```python
@jit(static_argnames=['option_type'])
def batch_price_and_greeks(F, K, T, r_d, sigma, option_type='call'):
    is_call = (option_type.lower() == 'call')
    ...
```

[Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) explained that `jit` traces a function with tracers that carry only a shape and a dtype. A string has neither, so it can't be traced. Without the static marking, the call fails with `TypeError: Error interpreting argument ... as an abstract array`. And even if a string could be traced, `.lower()` and `==` are plain Python operations, so they need the actual string.

A static argument isn't traced. JAX passes its real value in while tracing, so `is_call` comes out as an ordinary `True` or `False`. The static value also becomes part of the compilation cache key from [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}). The first call with `'call'` compiles one program, the first call with `'put'` compiles a second, and later calls reuse whichever one matches. There are two costs. Every distinct value triggers its own compilation, and `'Call'` counts as different from `'call'`. And because the type is fixed for each program, a single batch can't mix calls and puts.

Making the argument static doesn't, on its own, remove the put from a call's program. `black_price` still passes `is_call` to `jnp.where`, and `jnp.where` records both branches whether its condition is traced or not. That's the `select_n True` in [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %})'s jaxpr, and it's why a call's Greeks cost about three prices rather than two. To fix the branch at compile time, branch in Python on the static value:

```python
def black_price_static(F, K, T, r_d, sigma, is_call=True):
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * (sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T)
    if is_call:     # a Python bool, decided while tracing
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
```

Now only one branch is ever traced, and the compiled gradient of a call contains two `erf` calls instead of four. Rerunning [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %})'s benchmark of 100,000 options with this version added, the price plus a full set of Greeks drops from 3.3 prices to 1.9, against 5.7 for bumping. The Greeks agree with the `jnp.where` version to within $10^{-13}$, and the compiled gradient now contains exactly as many exponentials as the compiled price, a sign the put has gone. The cost is the one [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) warned about: this version works only while `is_call` is a plain Python value. Pass it a traced `is_call` and it fails with a `TracerBoolConversionError`.

The alternative goes the other way: pass `is_call` as a boolean array and keep `jnp.where`. One compiled program then covers both types, and you can `vmap` over `is_call` to price a mixed book of calls and puts in one call. The cost is that both branches are always computed and one is selected. For Black-76 that's only a cost in time, but it's exactly the situation the branch gotcha below warns about.

## Practical gotchas for quant use

### Precision: turn on float64

JAX defaults to **float32**. That's the right default for neural networks and the wrong one for pricing. Running `black_price` in a fresh Python process, once as it is and once with float64 turned on, gives:

| | float32 (default) | float64 |
|---|---|---|
| Delta | 0.5399290 | 0.5399286 |
| Gamma | 7.212564 | 7.212566 |
| Parity residual | 2.6 × 10⁻⁸ | 5.7 × 10⁻¹⁷ |

Delta is already wrong in the sixth digit. That's fine for a demo, but not for a risk system that aggregates thousands of positions or differences Greeks across scenarios. One line at the top of the program fixes it:

```python
jax.config.update("jax_enable_x64", True)
```

It's a process-wide setting. Importing `jax_finance` turns it on, which is why the float32 column needs a fresh process that runs the function on its own.

### What "rho" actually means

The autodiff rho from [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) is −0.00597: a small negative number, equal to $-T\cdot c$. An FX trader might expect something different. That's because `black_price` takes $F$ as an *input*, so the derivative with respect to `r_d` holds the forward fixed and picks up only the discounting effect.

Autodiff answers exactly the question the graph asks. To get the rho a trader means, re-parameterize from spot, so the rates also move the forward through $F = S\,e^{(r_d - r_f)T}$:

```python
def price_from_spot(S, K, T, r_d, r_f, sigma):
    F = S * jnp.exp((r_d - r_f) * T)
    return black_price(F, K, T, r_d, sigma)

jax.grad(price_from_spot, argnums=(0, 3, 4))(1.0800, 1.0800, 0.25, 0.05, 0.03, 0.10)
# spot delta 0.5456, domestic rho +0.1413, foreign rho -0.1473
```

No new derivative formulas were written. `r_d` now fans out to two places, the discount factor and the forward, and the backward sweep sums both contributions. That turns domestic rho from −0.006 into +0.141. Foreign rho appears for free, because `r_f` is now in the graph. This is the fan-out accumulation from [Part 2]({{ site.baseurl }}{% post_url 2026-11-16-Inside-The-Autodiff-Engine %})'s reverse-mode section doing real work, and it's the general lesson for AD in production: model your graph from the true market inputs, and the correct risk comes out automatically.

### Non-differentiable points

At expiry, $\sigma\sqrt{T} = 0$ and `d1` becomes a division by zero. JAX evaluates the price correctly: at $T = 0$ it returns 0.005, the intrinsic value $F - K$. But delta, theta and vega come back `NaN`, because the derivative rules for `div` and `sqrt` meet an infinity along the way. (Rho survives, as zero, because its only route is through $e^{-rT}$.) Payoffs with kinks and jumps, such as digitals, barriers and the expiry payoff itself, have the same problem in a more fundamental form. The derivative either doesn't exist or is a Dirac delta that no sampling will ever hit.

The remedies are smoothing the payoff, using a `jax.custom_jvp` to supply a hand-written derivative rule for a delicate piece of the graph, or restructuring the calculation (the pathwise and likelihood-ratio methods in Monte Carlo). `custom_jvp` is also how you would replace the `erf`/`erfc` construction inside `norm.cdf` with the textbook $n(x)$ if you wanted to control its behaviour deep in the tails. For expiry itself there's a simpler fix, and it's the subject of the next gotcha.

### Branches

`jnp.where` computes both branches and selects one. That's why it works under tracing, but it has a sting in its tail. If the unselected branch produces a `NaN` or `inf`, its gradient can still poison the result, because zero times `NaN` is `NaN` on the backward sweep.

Expiry shows it. The obvious fix for the `NaN` Greeks is to return the intrinsic value when $T = 0$:

```python
price = jnp.where(T > 0, black_76_formula, jnp.maximum(F - K, 0.0))
```

The price is now right, but delta, theta and vega are still `NaN`. The Black-76 branch is still computed at $T = 0$, its backward sweep still produces `NaN`, and the `where` multiplies that by zero instead of discarding it.

The standard fix is the **double where**: a first `where` makes the dangerous input safe before it reaches the formula, and a second chooses the result.

```python
def black_price_safe(F, K, T, r_d, sigma, is_call=True):
    live = T > 0
    T_ = jnp.where(live, T, 1.0)          # where #1: no T = 0 ever reaches sqrt or div
    sqrt_T = jnp.sqrt(T_)
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T_) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T_)
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    intrinsic = jnp.where(is_call, jnp.maximum(F - K, 0.0), jnp.maximum(K - F, 0.0))
    return jnp.where(live, jnp.where(is_call, call, put), intrinsic)   # where #2
```

At $T = 0$ the Black-76 branch now runs on a harmless $T = 1$ and produces finite numbers, so zero times them is zero. Delta comes back as 1.0, the delta of an in-the-money call at expiry, and theta, rho and vega come back as 0. At $T = 0.25$ the function returns the same price and Greeks as before. The SABR implementation uses the same trick to keep its at-the-money limit finite, and a later post looks at it in detail.

There are two other ways out. `jax.lax.cond` is traced as a real conditional and runs only the branch it takes. A Python `if` on a static value, as in `black_price_static` above, traces only one branch in the first place, at the cost of one compiled program per value.

## Takeaway

Because every JAX transformation returns an ordinary function, the Greeks compose. Add `grad` to the gradient and you get gamma. Add `hessian` and you get the whole second-order block, gamma, vanna and volga, checked against their closed forms to every printed digit. Add `vmap` and the single-option code prices and risks a whole strike ladder, compiled into one kernel. Getting numbers you can trust still takes care: float64, a graph built from the true market inputs, and attention to kinks and branches, where the double where turns `NaN` Greeks back into the right answer.

Black-76 is the easy case, though, because it has a closed form and we already know the answers. The next post takes the machinery somewhere hand differentiation really struggles: Monte Carlo pricing with adjoint Greeks, where the graph has millions of nodes, memory becomes the binding constraint and the payoff isn't smooth.
