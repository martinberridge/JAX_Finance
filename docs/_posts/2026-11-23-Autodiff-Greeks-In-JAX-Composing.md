---
layout: post
title: "Autodiff Greeks in JAX: composing transformations, and where it goes wrong"
date: 2026-11-23
categories: [quant-finance, autodiff]
series: autodiff-for-greeks
part: 4
---
[Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) printed the program that `jax.grad` writes for Black-76, checked its answers against [Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %})'s closed forms, and showed what the XLA compiler does to it. This post builds on that program. Composing JAX's transformations gives second-order Greeks and whole strike ladders from the same few lines. Then we cover the gotchas that matter when pricing with JAX.

The running example is unchanged: the `black_price` function from [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) for a three-month EUR/USD option with $F = 1.0850$, $K = 1.0800$, $T = 0.25$, $r_d = 5\%$ and $\sigma = 10\%$, in float64.

## Transformations are functions of functions

Every JAX transformation takes a function and returns a function. `grad(f)`, `jit(f)`, `vmap(f)` and `hessian(f)` are ordinary Python functions that JAX can trace again, so they nest in any order. This post uses five compositions: `grad` of `grad` for gamma, forward mode over reverse mode for the whole second-order block, `jacfwd` over `vmap` for one rate's effect on a book of options, `vmap` of an already-differentiated function for a strike ladder, and `jit` around all of it.

## Second-order Greeks

### The maths

The first-order Greeks are the gradient of the price; the second-order Greeks are its **Hessian**, the matrix of second derivatives. For the two inputs an FX options book cares about most—the forward $F$ and the volatility $\sigma$—the Hessian is a 2×2 matrix.

The Hessian isn't a new kind of object. [Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %}) defined the Jacobian as the matrix of partial derivatives of a function's outputs with respect to its inputs. For a pricing function with one output, the Jacobian is a single row: the gradient of first-order Greeks. Treat that gradient as a function in its own right—taking the same inputs as the price and returning a vector of Greeks—and its Jacobian is a square matrix where row $i$ shows how the $i$th Greek moves when each input moves:

$$H = J(\nabla c), \qquad H_{ij} = \frac{\partial}{\partial x_j}\left(\frac{\partial c}{\partial x_i}\right)$$

For $(F, \sigma)$, row 1 is the gradient of delta (how delta moves with forward and vol), and row 2 is the gradient of vega:

$$
H = \begin{pmatrix}
\dfrac{\partial^2 c}{\partial F^2} & \dfrac{\partial^2 c}{\partial F\,\partial\sigma} \\
\dfrac{\partial^2 c}{\partial \sigma\,\partial F} & \dfrac{\partial^2 c}{\partial \sigma^2}
\end{pmatrix}
= \begin{pmatrix} \Gamma & \text{vanna} \\ \text{vanna} & \text{volga} \end{pmatrix}
$$

The matrix is symmetric because mixed partial derivatives don't depend on order. Its three entries are:

- **Gamma**, $\Gamma = \partial^2 c/\partial F^2 = e^{-rT}n(d_1)/(F\sigma\sqrt{T})$: how fast delta changes as the forward moves. It governs how much a delta hedge must be adjusted and whether rebalancing makes or loses money.
- **Vanna**, $\partial^2 c/\partial F\,\partial\sigma = -e^{-rT}n(d_1)\,d_2/\sigma$: how delta changes when vol moves, or equivalently how vega changes when the forward moves.
- **Volga** (or vomma), $\partial^2 c/\partial\sigma^2 = \text{vega}\cdot d_1 d_2/\sigma$ (where $\text{vega} = \partial c/\partial\sigma$ is unscaled): how vega changes when vol moves.

Vanna and volga matter more in FX than almost anywhere else because the market's standard smile adjustment, the vanna-volga method, prices an option from the cost of hedging these two exposures.

### Differentiating a derivative: gamma

```python
gamma = grad(grad(black_price, argnums=0), argnums=0)(F, K, T, r_d, sigma)
```

`grad(black_price)` is an ordinary JAX function. It contains a forward sweep and a backward sweep, and JAX can trace and differentiate *that* just like any other function—applying [Part 2]({{ site.baseurl }}{% post_url 2026-11-16-Inside-The-Autodiff-Engine %})'s linearise-and-transpose steps to a program that already includes a backward sweep. The result, 7.21257, matches the closed form below.

### The whole block: `jax.hessian`

Two refinements improve on `grad(grad(...))`.

First, `grad(grad(...))` is **reverse-over-reverse**. For a Hessian, the standard pattern is **forward-over-reverse**: take the gradient in reverse mode, then differentiate it in forward mode. For $n$ inputs, that takes $n$ forward passes through the gradient program (one per column of the Hessian) without storing a backward sweep's residuals just to run a second backward sweep over them. `jax.hessian` is defined directly as `jacfwd(jacrev(f))`.

Second, you rarely want gamma alone. Pack $F$ and $\sigma$ into `x = jnp.array([F, sigma])` (`x[0]` is $F$, `x[1]` is $\sigma$) to get the entire second-order risk block in one call:

```python
H = jax.hessian(lambda x: black_price(x[0], K, T, r_d, x[1]))(jnp.array([F, sigma]))
# [[ 7.2126, -0.2636],     [[ gamma, vanna ],
#  [-0.2636,  0.0168]]  =   [ vanna, volga ]]
```

### Checking the second-order Greeks

Evaluating the closed forms next to the Hessian confirms every digit:

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

## Forward or reverse: the shape rule as an API

JAX exposes both `jax.jacfwd` (one JVP per input, column by column) and `jax.jacrev` (one VJP per output, row by row)—[Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %})'s shape rule turned into an API.

One price from five inputs is a 1×5 Jacobian (one reverse pass versus five forward passes), so use `jacrev` or `grad`. Conversely, when one rate moves a book of five options at different strikes:

```python
strikes = jnp.linspace(1.06, 1.10, 5)
book = lambda r: vmap(lambda k: black_price(F, k, T, r, sigma))(strikes)

jax.jacfwd(book)(r_d)
# [-0.00893117 -0.00736196 -0.00597054 -0.00476058 -0.00372964]
```

That's a 5×1 Jacobian: each option's rho ($-T\cdot c$ at its strike) from a single forward pass. `jacrev` would need five reverse passes—or 1,000 for a 1,000-option book.

## The strike ladder: `vmap`

A **batch** stacks independent calculations of the same kind into arrays along a **batch axis**. For a five-strike ladder, `strikes` has shape `(5,)` and axis 0 indexes the options: `strikes[i]` belongs to option `i`, and every output comes back aligned along that same axis (`delta[i]` is option `i`'s delta).

`jax.vmap(fun, in_axes=0)` lifts a single-option function to run across a batch.¹ Rather than looping in Python, it traces the function once and rewrites each primitive to act along the batch axis:

- `in_axes` specifies which axis is the batch axis for each positional argument (`0` for array arguments, or `None` for scalar arguments shared across every option, such as $F$).
- Outputs inherit the batch axis; if `fun` returns a dict of scalars, `vmap(fun)` returns a dict of arrays.

The accompanying code vectorizes the *already-differentiated* function:

```python
batch_fn = vmap(lambda f, k, t, r, s: compute_greeks_single(f, k, t, r, s, is_call))
```

`vmap` takes the jaxpr of `compute_greeks_single`—forward sweep, backward sweep, and nested gamma—and rewrites every primitive for a batch dimension.

[`batch_price_and_greeks`](https://github.com/martinberridge/JAX_Finance/blob/main/jax_finance/black.py) maps over all five market inputs, so shared parameters are broadcast to the ladder's length with `jnp.full`. To vary only the strike without broadcasting, call `vmap` directly with `in_axes=None` on the shared inputs:

```python
from jax_finance.black import batch_price_and_greeks

F, T, r_d, sigma = 1.0850, 0.25, 0.05, 0.10
strikes = jnp.linspace(1.06, 1.10, 5)        # 1.06, 1.07, ..., 1.10
n = strikes.shape[0]

# Option 1: broadcast shared inputs for batch_price_and_greeks
ladder = batch_price_and_greeks(
    jnp.full(n, F), strikes, jnp.full(n, T), jnp.full(n, r_d), jnp.full(n, sigma), option_type="call"
)

# Option 2: map only over strikes using in_axes
ladder = vmap(compute_greeks_single, in_axes=(None, 0, None, None, None, None))(
    F, strikes, T, r_d, sigma, True
)
```

Both produce the same ladder:

| Strike | Price | Delta | Gamma | Vega (1%) | Theta (1 day) |
|---|---|---|---|---|---|
| 1.0600 | 0.035725 | 0.679814 | 6.437012 | 0.001894 | −0.000099 |
| 1.0700 | 0.029448 | 0.611526 | 6.935685 | 0.002041 | −0.000108 |
| 1.0800 | 0.023882 | 0.539929 | 7.212566 | 0.002123 | −0.000113 |
| 1.0900 | 0.019042 | 0.467430 | 7.246165 | 0.002133 | −0.000114 |
| 1.1000 | 0.014919 | 0.396460 | 7.039680 | 0.002072 | −0.000111 |

Delta crosses 0.5 near the money, and gamma and vega peak just above the forward. Wrapped in `jit`, the entire ladder compiles into a single fused kernel ([Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %})).

¹ Unlike `np.vectorize`, which wraps a Python `for` loop, `vmap` vectorizes at the primitive level. If you prefer NumPy's signature, `jnp.vectorize` has the same signature and is built on `vmap`.

## Static versus traced arguments

`batch_price_and_greeks` marks `option_type` as static:

```python
@jit(static_argnames=['option_type'])
def batch_price_and_greeks(F, K, T, r_d, sigma, option_type='call'):
    is_call = (option_type.lower() == 'call')
    ...
```

As [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}) noted, `jit` traces inputs using abstract tracers that carry only a shape and dtype. A string has neither—and Python methods like `.lower()` require a concrete string—so without `static_argnames`, JAX raises a `TypeError`.

Marking an argument static passes its concrete value at trace time and includes it in the compilation cache key (`'call'` and `'put'` compile separate executables, and `'Call'` triggers a third). However, making `is_call` a Python `bool` doesn't remove the put branch if `black_price` still passes it to `jnp.where`, because `jnp.where` traces both branches unconditionally—which is why a call's Greeks cost ~3.3 prices in [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %}).

To eliminate the unused branch at compile time, branch with a Python `if` on the static boolean:

```python
def black_price_static(F, K, T, r_d, sigma, is_call=True):
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * (sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T)
    if is_call:     # Python bool, resolved at trace time
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
```

Now only one branch is traced, cutting the compiled gradient's `erf` calls from four to two. On [Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %})'s 100,000-option benchmark, price plus full Greeks drops from 3.3× a price to 1.9× (versus 5.7× for bumping), matching the `jnp.where` Greeks to within $10^{-13}$. The tradeoff is that `is_call` must be a static Python `bool`; passing a traced array raises `TracerBoolConversionError`.

| Approach | Compilations | Mixed call/put batch? | Unused branch cost & `NaN` risk |
|---|---|---|---|
| Python `if` + static `is_call` | One per option type | No | Zero — pruned at trace time |
| `jnp.where` + traced `is_call` array | One for both types | Yes (`vmap` over `is_call`) | Evaluates both branches; `NaN`/`inf` in unselected branch can leak |

## Practical gotchas for quant use

### Precision: turn on float64

JAX defaults to **float32**—ideal for neural networks, too coarse for pricing. Running `black_price` in a fresh Python process with and without float64 shows the gap:

| | float32 (default) | float64 |
|---|---|---|
| Delta | 0.5399290 | 0.5399286 |
| Gamma | 7.212564 | 7.212566 |
| Parity residual | 2.6 × 10⁻⁸ | 5.7 × 10⁻¹⁷ |

Delta is already off in the sixth digit. Enable 64-bit mode at startup (importing `jax_finance` does this automatically):

```python
jax.config.update("jax_enable_x64", True)
```

### What "rho" actually means

[Part 3]({{ site.baseurl }}{% post_url 2026-11-19-Autodiff-Greeks-In-JAX %})'s autodiff rho is −0.00597 ($-T\cdot c$) because `black_price` takes $F$ as an independent input, differentiating with respect to `r_d` while holding $F$ fixed.

Autodiff answers the exact question the graph encodes. To get spot-based FX rho, re-parameterize from spot $S$ via $F = S\,e^{(r_d - r_f)T}$:

```python
def price_from_spot(S, K, T, r_d, r_f, sigma):
    F = S * jnp.exp((r_d - r_f) * T)
    return black_price(F, K, T, r_d, sigma)

jax.grad(price_from_spot, argnums=(0, 3, 4))(1.0800, 1.0800, 0.25, 0.05, 0.03, 0.10)
# spot delta 0.5456, domestic rho +0.1413, foreign rho -0.1473
```

`r_d` now fans out to both the forward and the discount factor, and the backward sweep accumulates both paths ([Part 2]({{ site.baseurl }}{% post_url 2026-11-16-Inside-The-Autodiff-Engine %})), flipping domestic rho from −0.006 to +0.141 and giving foreign rho for free. Model your graph from the true market inputs, and the right risk follows automatically.

### Non-differentiable points

At expiry ($T = 0$), $\sigma\sqrt{T} = 0$ causes a division by zero in `d1`. JAX still evaluates the forward price cleanly (returning the intrinsic value $0.005 = F - K$), but delta, theta and vega become `NaN` because the backward rules for `div` and `sqrt` hit infinities. (Rho survives as zero since its only path is through $e^{-rT}$.) Payoffs with kinks or jumps—such as digitals, barriers, and expiry payoffs—have the same issue whenever evaluated at a non-differentiable point.

General remedies include payoff smoothing, `jax.custom_jvp` for hand-written derivative rules (also useful for replacing `erf`/`erfc` in `norm.cdf` with $n(x)$ in deep tails), or likelihood-ratio methods in Monte Carlo. For expiry itself, there is a simpler fix.

### Branches

Because `jnp.where` evaluates both branches, a `NaN` or `inf` in the *unselected* branch can still poison the gradient on the backward sweep (`0 * NaN = NaN` and `0 * inf = NaN`). Guarding expiry with a single `where`:

```python
price = jnp.where(T > 0, black_76_formula, jnp.maximum(F - K, 0.0))
```

fixes the price at $T = 0$, but delta, theta and vega remain `NaN` because `black_76_formula` is still differentiated at $T = 0$.

The standard idiom is the **double where**: sanitize the dangerous input before the formula (`where #1`), then select the output (`where #2`):

```python
def black_price_safe(F, K, T, r_d, sigma, is_call=True):
    live = T > 0
    T_ = jnp.where(live, T, 1.0)          # where #1: prevent T = 0 from reaching sqrt or div
    sqrt_T = jnp.sqrt(T_)
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T_) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T_)
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    intrinsic = jnp.where(is_call, jnp.maximum(F - K, 0.0), jnp.maximum(K - F, 0.0))
    return jnp.where(live, jnp.where(is_call, call, put), intrinsic)   # where #2
```

At $T = 0$ (assuming $\sigma > 0$), the Black-76 branch runs with a safe dummy $T\_ = 1.0$, producing finite intermediates whose zero-weighted adjoints stay zero. Delta returns `1.0` for an in-the-money call, theta, rho and vega return `0.0`, and live options ($T = 0.25$) are unchanged. The SABR implementation uses the same pattern for its at-the-money limit.

Two alternatives exist when you don't need full SIMD masking: `jax.lax.cond` executes only the taken branch at runtime, while a Python `if` on a static argument (as in `black_price_static`) prunes the unused branch at trace time.

## Takeaway

Because every JAX transformation returns an ordinary function, Greeks compose cleanly: `grad` of `grad` gives gamma, `hessian` gives the full second-order block (gamma, vanna, volga) to machine precision, and `vmap` lifts single-option Greeks into a fused strike-ladder kernel. Reliable production numbers still require three habits: enable float64, build the graph from true market inputs, and guard kinks and branches with the double `where` so `NaN`s never leak into your sensitivities.

The next two posts are still options related - we will look at how we can use Autodiff to calibrate a volatility surface , a similar problem to training an MLP