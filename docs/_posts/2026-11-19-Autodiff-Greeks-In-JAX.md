---
layout: post
title: "Autodiff Greeks in JAX: reading the gradient program and compiling it"
date: 2026-11-19
categories: [quant-finance, autodiff]
series: autodiff-for-greeks
part: 3
---
[Part 2]({{ site.baseurl }}{% post_url 2026-11-16-Inside-The-Autodiff-Engine %}) opened up the autodiff engine: a set of primitives with a JVP and a VJP rule each, graph capture by tracing, forward and reverse sweeps, and JAX's trick of deriving reverse mode by linearising and transposing, then generating and running a mini-program called a **jaxpr**. We mentioned that, unlike PyTorch's eager execution of a compute graph, the jaxpr is compiled and optimised by Google's XLA compiler.

This post looks at what that machinery produces in practice. We print the jaxpr JAX generates when we call `jax.grad` on our `black_price` function, check its answers against Part 1's closed forms, and look at what the XLA compiler does to it, which is where the speed comes from. Part 4 then composes JAX's transformations to get second-order Greeks and whole strike ladders, and covers the gotchas that matter when pricing with JAX.

The running example is unchanged: Black-76 in an FX setting, with a USD rate `r_d` discounting a EUR/USD forward.

## The pricing function

Here's our pricing function again, from [`jax_finance/black.py`](https://github.com/martinberridge/JAX_Finance/blob/main/jax_finance/black.py):

```python
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", True)

def black_price(F, K, T, r_d, sigma, is_call=True):
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * (sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T)

    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)

    call_price = df * (F * n_d1 - K * n_d2)
    # N(-d) directly, not 1 - N(d): deep OTM, 1 - N(d) cancels to zero in float64
    put_price = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return jnp.where(is_call, call_price, put_price)
```

The `jax_enable_x64` line turns on 64-bit floats. JAX defaults to 32-bit, which is the right choice for neural networks and too coarse for pricing. Part 4 shows by how much.

The market data in the examples is for a three-month EUR/USD option: $F = 1.0850$, $K = 1.0800$, $T = 0.25$, $r_d = 5\%$, $\sigma = 10\%$. The Greeks come from one call:

```python
jax.grad(black_price, argnums=(0, 2, 3, 4))(1.0850, 1.0800, 0.25, 0.05, 0.10)
```

The contents of `argnums` are positional argument indices: integers saying which arguments to differentiate with respect to. Here they're $F$, $T$, $r_d$ and $\sigma$, giving delta, theta, rho and vega. Arguments left out of `argnums` are treated as constants on the backward pass. Index 1 is left out, so $K$, the strike, is a constant.

## Tracing: why the code looks like this

Part 2 said that JAX captures the graph by **tracing**. That has consequences for how a pricing function has to be written, and it explains two things about the code above: it uses `jax.numpy` and `jax.scipy` rather than NumPy and SciPy, and it chooses between call and put with `jnp.where` rather than `if`.

When you call a transformed function, JAX doesn't simply run your Python on your numbers. It runs it on **tracers**: stand-in objects that record every JAX operation they pass through. Under `jit`, `vmap` and `make_jaxpr` a tracer carries only a shape and a dtype, such as `f64[]` for a 64-bit float scalar, and no value at all. Two things follow.

- **NumPy and SciPy can't see tracers.** `np.sin(tracer)` tries to turn the tracer into a NumPy array and fails with a `TracerArrayConversionError`. `jax.numpy` and `jax.scipy` are drop-in replacements with the interfaces you expect, written in JAX's own primitives, so tracers pass through them and get recorded.
- **Python control flow can't branch on a traced value.** `if is_call:` needs `is_call` to be `True` or `False` right now, and under `jit` it's a tracer with no value. JAX raises a `TracerBoolConversionError`. Under `grad` on its own the tracer still carries the value, so the `if` happens to work there, but the same code breaks the moment you `jit` or `vmap` it. `jnp.where` avoids the question: it records both branches and a select between them, and the choice is made when the program runs.

Recording both branches has a cost, and the gradient program below shows it. Part 4 comes back to the alternatives.

## Reading the gradient program

Part 2 described reverse mode as a forward sweep that saves residuals, followed by a backward sweep that sums adjoints wherever a value fans out. In JAX you can read that sweep directly. `grad` doesn't differentiate your Python code step by step: it writes a new program, a **jaxpr**, and runs that. `jax.make_jaxpr` prints it.

JAX builds a jaxpr whenever a transformation needs to see the whole program before running any of it: `jit` to hand it to a compiler, `vmap` to rewrite every operation for a batch axis, and `grad` to run the dataflow backwards.

We'll start with a function small enough to read whole, then turn to Black-76.

### Anatomy of a jaxpr

The toy function is $f(x) = x\sin x$. It's small, but it has the two things that make reverse mode interesting: a nonlinear step that will need a residual, and an input that's used twice.

```python
def f(x):
    return jnp.sin(x) * x

jax.make_jaxpr(f)(2.0)
```

```text
{ lambda ; a:f64[]. let
    b:f64[] = sin a
    c:f64[] = mul b a
  in (c,) }
```

- **The header**, `{ lambda ; a:f64[]. let`. Names before the `;` are constants captured during tracing, and there are none here. Names after it are the inputs, written as `name:dtype[shape]`. `f64[]` is a 64-bit float scalar.
- **The body.** One primitive per line. Each variable is assigned exactly once and never overwritten.
- **The return**, `in (c,)`. The outputs, as a tuple.

The 2.0 we passed in doesn't appear anywhere. `make_jaxpr` runs `f` on a tracer that carries only a shape and a dtype, so the program is the same for every scalar input.

### From function to gradient program in four steps

Part 2 explained why JAX builds reverse mode by linearising and then transposing. Here's what each step produces for $f$.

1. **Trace.** That's the jaxpr above.

2. **Linearise.** `jax.linearize(f, 2.0)` evaluates $f$ at $x = 2$ and returns the linear map from $\dot{x}$ to $\dot{y}$ as a new jaxpr:

   ```text
   { lambda a:f64[] b:f64[]; c:f64[]. let
       d:f64[] = mul c a
       e:f64[] = mul d 2.0:f64[]
       f:f64[] = mul b c
       g:f64[] = add_any e f
     in (g,) }
   ```

   The constants before the `;` are now filled in: `a` is $\cos 2$ and `b` is $\sin 2$, and $x = 2$ itself has been baked in as the literal `2.0`. These are the **residuals**, fixed by the forward evaluation. The input `c` is the tangent $\dot{x}$, and the body is the product rule, $\dot{y} = x\cos x\,\dot{x} + \sin x\,\dot{x}$. There's no `sin` or `cos` left in it, only `mul` and `add_any`, so it's a linear map.

3. **Transpose.** JAX runs that linear program backwards, replacing each operation with its transpose. Multiplying by a constant transposes to multiplying by the same constant. The sum and the fan-out swap places. The `add_any` that summed the product rule's two terms becomes a copy that sends $\bar{y}$ down both routes. The tangent $\dot{x}$, which was used twice (in `d` and in `f`), becomes an `add_any` that collects both contributions to $\bar{x}$.

4. **Assemble.** JAX joins the forward sweep to the transposed program and seeds it with $\bar{y} = 1$. This is what `jax.make_jaxpr(jax.grad(f))(2.0)` prints:

   ```text
   { lambda ; a:f64[]. let
       b:f64[] = sin a
       c:f64[] = cos a
       _:f64[] = mul b a
       d:f64[] = mul b 1.0:f64[]
       e:f64[] = mul 1.0:f64[] a
       f:f64[] = mul e c
       g:f64[] = add_any d f
     in (g,) }
   ```

Line by line:

- `b` and `c` are the forward sweep. `c = cos a` plays no part in computing $f$: it's a residual, computed now because the backward sweep needs it.
- `_` is $f(x)$ itself, computed and then discarded, because `grad` returns only the gradient.
- `d` is $\bar{x}$ through the right-hand factor $x$: $\bar{y}\sin x$. The `1.0` is the seed $\bar{y}$.
- `e` and `f` are $\bar{x}$ through the left-hand factor and then through `sin`: $\bar{y}\,x\cos x$.
- `g` adds the two routes, because $x$ fanned out: $\bar{x} = \sin x + x\cos x$.

You never call `linearize` yourself to get a gradient: `grad` does all four steps. In the final program the residuals are no longer constants in the header. They're ordinary lines in the forward sweep, like `c`.

### The Black-76 gradient program

The same call on `black_price` prints the program behind our Greeks:

```python
jax.make_jaxpr(jax.grad(black_price, argnums=(0, 2, 3, 4)))(1.0850, 1.0800, 0.25, 0.05, 0.10)
```

The full output is about 280 lines. <!-- TODO: commit the full jaxpr to the repo and link it here --> Below is a redacted version, printed with JAX 0.10 in float64. Type annotations and dtype conversions are stripped, the four `norm.cdf` subnetworks and their transposes are collapsed, and comments are added. Variable names come straight from JAX and will differ between versions.

The inputs `hw hx hy hz ia` are $F$, $K$, $T$, $r_d$ and $\sigma$. `is_call` was left at its default, a plain Python `True`, so it isn't an input. It shows up as the literal `True` in `select_n`.

```
{ lambda ; hw hx hy hz ia. let
  # ── forward sweep: primals, plus residuals for later ─────────────────────
    ib = sqrt hy                  # √T
    ic = div 0.5 ib               # 1/(2√T)      residual: d√T/dT, kept for the backward sweep
    id = div hw hx                # F/K
    ie = log id                   # ln(F/K)
    if = integer_pow[y=2] ia      # σ²
    ig = mul 2.0 ia               # 2σ           residual: dσ²/dσ
    ih = mul 0.5 if               # ½σ²
    ii = mul ih hy                # ½σ²T
    ij = add ie ii                # d1 numerator
    ik = mul ia ib                # σ√T
    il = div ij ik                # d1
    im = integer_pow[y=-2] ik     # 1/(σ√T)²     residual: div's derivative w.r.t. its denominator
    in = mul ia ib                # σ√T again
    io = sub il in                # d2
    ip = neg hz                   # −r
    iq = mul ip hy                # −rT
    ir = exp iq                   # e^(−rT)
    ...                           # N(d1): erf / erfc / select_n, plus three
    ja = custom_jvp_call[exp_neg_sq] iv   #   e^(−x²) calls: residuals for erf′ and erfc′
    ...                           # N(d2): the same again
    kk = mul kj ki                # call price
    kl = neg io                   # −d2
    ...                           # N(−d2): the same again
    lh = neg il                   # −d1
    ...                           # N(−d1): the same again
    mf = mul me md                # put price
    _  = select_n True kk mf      # the price itself, discarded: grad doesn't return it

  # ── the seed: jnp.where, transposed ──────────────────────────────────────
    mh, mi = select_n True ... 1.0    # c̄ = 1 routed: mh = 1 to the call, mi = 0 to the put

  # ── backward sweep ───────────────────────────────────────────────────────
    ml = mul me mi                # put-branch adjoints: every one is scaled by mi = 0
    ...                           # ~50 lines: back through N(−d1) and N(−d2), all zeros
    ol = mul kj mh                # adjoint of F·N(d1) − K·N(d2): e^(−rT)
    om = mul mh ki                # adjoint of e^(−rT) from the call...
    oo = add_any mn on            # ...summed with the put's (zero) contribution
    or = mul ke ol                # adjoint of N(d1): e^(−rT)·F
    os = mul ol jk                # F̄ via node 4.1: e^(−rT)·N(d1)
    ou = add_any mr ot            # ...plus F̄ via the put branch (zero)
    ...                           # back through N(d2), multiplying by its e^(−x²) residuals
    pq = add_any ok pp            # d̄2 = via N(d2) + via N(−d2) (zero)             [fan-out]
    ...                           # back through N(d1), the same way
    qm = add_any nn ql            # d̄1 so far = via N(d1) + via N(−d1) (zero)       [fan-out]
    qn = mul oo ir                # adjoint of −rT
    qo = mul ip qn                # T̄, contribution 1: via e^(−rT)
    qp = mul qn hy
    qq = neg qp                   # r̄: rho                                         → output
    qr = neg pq                   # adjoint of σ√T (in), via d2
    qs = add_any qm pq            # d̄1 = direct paths + path via d2 = d1 − σ√T     [fan-out]
    qt = mul ia qr                # √T̄ via in
    qu = mul qr ib                # σ̄ via in
    ...                           # qx: adjoint of σ√T (ik), via d1's denominator (uses im)
    qy = div qs ik                # adjoint of the d1 numerator
    qz = mul ia qx                # √T̄ via ik
    ra = mul qx ib                # σ̄ via ik
    rb = add_any qu ra            # σ̄ from both σ√T nodes                          [fan-out]
    rc = add_any qt qz            # √T̄ from both σ√T nodes                         [fan-out]
    rd = mul ih qy                # T̄, contribution 2: via ½σ²T
    re = mul qy hy
    rf = add_any qo rd
    rg = mul 0.5 re               # adjoint of σ²
    rh = mul rg ig                # σ̄ via σ², using residual 2σ
    ri = add_any rb rh            # vega                                            → output
    rj = div qy id
    rk = div rj hx                # F̄ via ln(F/K)
    rl = add_any ou rk            # delta                                           → output
    rm = mul rc ic                # T̄, contribution 3: via √T, using residual 1/(2√T)
    rn = add_any rf rm            # ∂c/∂T: three contributions summed               → output
  in (rl, rn, qq, ri) }           # (Δ, ∂c/∂T, ρ, vega)
```

Every idea from Part 2, and from the toy example, is visible in it.

**Residuals are just forward-sweep lines with no forward purpose.** `ic`, `ig`, `im` and the `exp_neg_sq` calls do nothing for the price. They're there because a backward line needs them: `rm` uses `ic`, `rh` uses `ig`, and the `erf` transposes use the $e^{-x^2}$ values. This is the last column of Part 2's derivative-rules table, scheduled into the program.

**The backward sweep is linear.** Read down the bottom half and you'll find only `mul`, `div`, `neg`, `select_n` and `add_any`. There's no `log`, no `exp` and no `erf`, and every `div` divides by a value fixed in the forward sweep. It's the transposed `f_lin` from Part 2, glued onto the forward sweep that produced its constants.

**Fan-outs are `add_any`.** The program contains 23 of them. Theta is the easiest to follow: `qo`, `rd` and `rm` are $T$'s three contributions, through $e^{-rT}$, $\tfrac12\sigma^2T$ and $\sqrt{T}$, and `rf` and `rn` add them up. Vega gathers three paths the same way (`qu`, `ra`, `rh`).

**`jnp.where` sends a zero, not nothing.** The transpose of `select_n` routes $\bar{c} = 1$ to the call and $0$ to the put. The put branch's backward lines still run, scaled by `mi`. Because the put has its own $N(-d_1)$ and $N(-d_2)$, that's two whole `norm.cdf` transposes evaluated only to multiply by zero. It's also the branch gotcha from Part 4 in literal form: if anything on the put side were `inf`, `0 × inf` would put a `NaN` into delta.

**AD doesn't know your identities.** Delta reaches $F$ along two routes. `os` is the direct route, through the $F$ that multiplies $N(d_1)$, and gives the textbook $e^{-rT}N(d_1)$. `rk` is the route through $\ln(F/K)$, which moves $d_1$ and $d_2$ and so both $N$ terms. That route is worth $e^{-rT}\bigl(Fn(d_1) - Kn(d_2)\bigr)/(F\sigma\sqrt{T})$, which is zero because $Fn(d_1) = Kn(d_2)$, the identity that makes Black's delta so tidy. A derivation by hand uses the identity to cancel the term before writing it down. JAX knows derivative rules, not identities, so it computes the term anyway, gets about $5\times10^{-17}$, and adds it to delta in `rl`.

**The price is thrown away.** As with the `_` line in the toy example, the line marked `_` computes the price and discards it, because `grad` only returns the gradient. `value_and_grad` keeps it at no extra cost, which is why the accompanying code uses it.

## Checking the answers

An autodiff result is only as trustworthy as the program it differentiates, so it's worth checking against Part 1's closed forms. The check computes the price and all five first derivatives with one `value_and_grad` call and evaluates the closed forms next to them:

```python
F, K, T, r_d, sigma = 1.0850, 1.0800, 0.25, 0.05, 0.10

# JAX: price and all five first derivatives in one reverse pass
price, (dF, dK, dT, dr, dsigma) = jax.value_and_grad(black_price, argnums=(0, 1, 2, 3, 4))(
    F, K, T, r_d, sigma
)

# Closed forms from Part 1
sqrt_T = jnp.sqrt(T)
d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
d2 = d1 - sigma * sqrt_T
df = jnp.exp(-r_d * T)
N1, N2, n1 = norm.cdf(d1), norm.cdf(d2), norm.pdf(d1)
c = df * (F * N1 - K * N2)

rows = [
    ("Price",      price,  c),
    ("Delta",      dF,     df * N1),
    ("Dual delta", dK,     -df * N2),
    ("Vega",       dsigma, df * F * sqrt_T * n1),
    ("Rho",        dr,     -T * c),
    ("dc/dT",      dT,     df * (F * n1 * sigma / (2 * sqrt_T) + r_d * K * N2 - r_d * F * N1)),
]
for name, ad, closed in rows:
    print(f"{name:<11} {ad:14.10f} {closed:14.10f}")
```

The call option gives:

| Quantity | JAX autodiff | Closed form (Part 1) |
|---|---|---|
| Price | 0.0238821419 | $e^{-rT}[FN(d_1) - KN(d_2)]$ = 0.0238821419 |
| $\partial c/\partial F$ (delta) | 0.5399286386 | $e^{-rT}N(d_1)$ = 0.5399286386 |
| $\partial c/\partial K$ (dual delta) | −0.5203152139 | $-e^{-rT}N(d_2)$ = −0.5203152139 |
| $\partial c/\partial \sigma$ (vega) | 0.2122703392 | $e^{-rT}F\sqrt{T}\,n(d_1)$ = 0.2122703392 |
| $\partial c/\partial r$ (rho) | −0.0059705355 | $-T\cdot c$ = −0.0059705355 |
| $\partial c/\partial T$ | 0.0412599607 | three-term formula = 0.0412599607 |

Agreement is to every printed digit. The code's printed output quotes some of these differently, and the conversions are worth spelling out. Vega and rho are scaled per 1% (×0.01), giving 0.002123 and −0.000060. Theta is quoted as calendar decay per day, $-\frac{1}{365}\,\partial c/\partial T$, giving −0.000113. Gamma, a second derivative, is checked in Part 4 alongside vanna and volga.

The script also checks put-call parity, $C - P - (F - K)e^{-rT} = 0$. It's a good end-to-end check that the traced graph, including both `jnp.where` branches, still encodes the right mathematics.

## Compilation: where the speed comes from

So far everything has run op by op: each primitive executed as soon as Python reached it. The accompanying `FXOptionPricer` class wraps the whole price-and-Greeks computation in `jit`:

```python
self._all_greeks_jit = jit(
    lambda F, K, T, r_d, sigma: compute_greeks_single(F, K, T, r_d, sigma, self.is_call)
)
```

### What `jit` does

The first call to a jitted function goes through four stages:

1. **Trace.** JAX traces `compute_greeks_single` into one jaxpr. That includes the `value_and_grad` program from the previous section and a second, nested `grad` for gamma, which Part 4 looks at.
2. **Lower.** The jaxpr is translated into StableHLO, the input language of **XLA**, Google's optimizing compiler. `jax.jit(f).lower(*args).as_text()` prints this stage.
3. **Optimise.** XLA rewrites the whole program at once. The next section covers the rewrites that matter here.
4. **Compile.** The optimised program becomes machine code for the target device, CPU or GPU: an **executable**. `jax.jit(f).lower(*args).compile().as_text()` prints the optimised program it was built from.

JAX caches the executable, keyed on the shapes and dtypes of the inputs (and on any static arguments, which Part 4 covers). Every later call with the same key skips all four stages and runs the executable directly. A new key, such as a batch of 1,000 options after a batch of 100, means tracing and compiling again. Timing one option through `FXOptionPricer.all_greeks`, the first call took about 450 ms, almost all of it compilation, and later calls took about 0.2 ms.

### What XLA does to the program

Three of XLA's rewrites matter for the "Greeks at the cost of a price" claim:

- **Common subexpression elimination (CSE).** When the same value is computed twice from the same inputs, XLA computes it once. The duplicated $\sigma\sqrt{T}$ in the jaxpr (lines `ik` and `in`) is the small example. The big one is that `compute_greeks_single` runs two differentiated programs, `value_and_grad` for the first-order Greeks and the nested `grad` for gamma. Both start with a forward sweep that computes $d_1$, $d_2$, $N(d_1)$ and $e^{-rT}$, and once they're compiled together XLA computes those once.
- **Dead code elimination (DCE).** Anything computed but never used is dropped. The `_` lines in the jaxprs are the obvious candidates: under `grad` the price is computed and thrown away, so XLA removes whatever the backward sweep doesn't also need.
- **Fusion.** Black-76 is a chain of elementwise operations: on a batch, every step takes arrays of 100,000 values and produces another array of 100,000 values. Run one at a time, each step writes its whole result out to memory and the next step reads it back. XLA fuses the chain into a single **kernel**: one compiled loop that takes each option through all the steps while its intermediate values are still in registers, and writes out only the results. ("Kernel" is the GPU word for a function that runs over a whole array, and XLA uses it for CPUs too.) In a program like this, where each step does very little arithmetic, moving data to and from memory is most of the cost, so fusion is where most of the speed comes from.

To see what this buys, here's a batch of 100,000 options in float64. The script is [`examples/black_benchmark.py`](https://github.com/martinberridge/JAX_Finance/blob/main/examples/black_benchmark.py), run with JAX 0.10 on the CPU of a laptop (Intel Core i7-1360P). Each time is the median of 20 calls after a warm-up call:

| Method | What it computes | Time | Relative to price only |
|---|---|---|---|
| Price only | price | 1.2 ms | 1.0× |
| JAX AD, jitted | price, Δ, Γ, vega, θ, ρ | 4.1 ms | 3.3× |
| Bump-and-reprice, central differences, jitted | same | 7.2 ms | 5.9× |
| JAX AD, *not* jitted | same | 188 ms | 153× |

Milliseconds depend on the machine, so the ratios are what count, and even those move from run to run.

Across seven runs, jitted AD came in at 2.6–4.4× a price, bumping at 4.9–6.4× and un-jitted AD at 136–178×. The price plus five Greeks, one of them second-order, costs about three prices.

Bumping needs nine repricings. It comes in under 9× only because XLA shares work between them, and it still costs one and a half to two times as much as AD. It's also less accurate: at the example option, its gamma matches the AD value to about six significant figures.

Why three prices rather than two? The answer is in the jaxpr. The price-only program never uses the put branch, so XLA's dead code elimination drops it, and the compiled price contains two `erf` calls. The gradient program can't drop it. The backward sweep multiplies the put's value by the zero that `jnp.where` sends (`mm = mul mi md` in the full jaxpr), and a compiler isn't allowed to rewrite `0 × x` as `0`, because `0 × inf` is `NaN`. So the put's $N(-d_1)$ and $N(-d_2)$ survive, the compiled gradient contains four `erf` calls, and every Greek pays for a branch whose contribution is exactly zero. That's the price of an accurate put formula inside a `jnp.where`. `batch_price_and_greeks` already knows the option type at compile time, so choosing the branch with a Python `if` instead would remove the put from the call's program entirely. Part 4 shows how.

The un-jitted row is the cautionary one. Without `jit`, JAX runs the program one primitive at a time. Each of its several hundred primitives is dispatched from Python as its own small computation, and each writes its full 100,000-element result to memory before the next one reads it back. Nothing is fused, shared or eliminated, and autodiff run like this loses to jitted bumping by more than an order of magnitude.

*The mathematics of reverse mode says Greeks can be cheap. It's the compiler that makes them cheap in practice.*

## The memory bill

Reverse mode's speed is paid for in storage, because every residual from the forward sweep is held until the backward sweep uses it. For Black-76 that's a few dozen scalars per option, which is nothing. For a Monte Carlo pricer with 10,000 paths and 250 time steps, it's every intermediate state on every path at every step, and memory becomes the constraint that binds first. `jax.checkpoint` addresses this by trading memory for recomputation: it stores fewer residuals and recomputes the rest during the backward sweep. We'll come back to it when we get to Monte Carlo.

## Takeaway

In JAX, all of Part 1 collapses into a single line, `value_and_grad(black_price, argnums=...)`. Part 2 showed what that line is: one forward sweep that stores residuals and one backward sweep that sums adjoints at every fan-out, derived automatically from forward-mode rules by linearising and transposing. This post printed that program, checked its answers against the closed forms, and showed what happens next. XLA fuses the program into a single kernel, so the price and a full set of Greeks cost about three prices, and the jaxpr shows exactly where the third one goes.

### Coming up in Part 4

Part 4 composes JAX's transformations and covers what can go wrong:

- **Second-order Greeks.** Differentiating a derivative gives gamma, and `jax.hessian` gives gamma, vanna and volga in one line, checked against their closed forms.
- **Forward or reverse.** `jacfwd` and `jacrev`, Part 1's shape rule turned into an API.
- **Strike ladders.** `vmap` turns the single-option code into a whole ladder.
- **Static versus traced arguments,** and how to stop paying for the put inside a call's Greeks.
- **The gotchas that matter for pricing:** why you need float64, why autodiff's "rho" may not be the rho a trader means, what happens at expiry, and how a branch can leak a `NaN` into your Greeks.
