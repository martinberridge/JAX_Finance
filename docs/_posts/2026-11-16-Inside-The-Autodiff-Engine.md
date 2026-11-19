---
layout: post
title: "Inside the autodiff engine: pullbacks, pushforwards and the backward sweep"
date: 2026-11-16
categories: [quant-finance, autodiff]
series: autodiff-for-greeks
part: 2
---
In [Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %}) we saw that the Greeks can be calculated with reverse-mode autodiff, the same machinery that computes gradients and backpropagates errors when training an MLP. We also saw when each mode fits: reverse mode for a function with many inputs and few outputs, and forward mode for one with few inputs and many outputs. An option price, with five inputs and one output, is the textbook case for reverse mode.

[Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %}) wrote the Greeks as a scalar, 1, times a global Jacobian, and built that Jacobian as the product of five local Jacobians, one per layer of the Black-76 model represented as an MLP. In JAX, all of that takes a few lines of code.

First, the Black-76 model in Python:

```python
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

def black_price(F, K, T, r_d, sigma, is_call=True):
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * (sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r_d * T)

    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)

    call_price = df * (F * n_d1 - K * n_d2)
    put_price = df * (K * (1.0 - n_d2) - F * (1.0 - n_d1))

    return jnp.where(is_call, call_price, put_price)
```

Then the call that runs reverse-mode autodiff. Pass the pricing function to `jax.value_and_grad`, choose which arguments you want Greeks for with `argnums`, and pass in the market data:

```python
jax.config.update("jax_enable_x64", True)   # pricing needs float64

price, (delta, dual_delta, dc_dT, rho, vega) = jax.value_and_grad(
    black_price, argnums=(0, 1, 2, 3, 4)
)(1.0850, 1.0800, 0.25, 0.05, 0.10)   # F, K, T, r_d, sigma

theta = -dc_dT
# price 0.0238821, delta 0.5399286, dual delta -0.5203152,
# dc/dT 0.0412600 (theta -0.0412600 per year), rho -0.0059705, vega 0.2122703
```

That's it: the price and all five first-order Greeks from one call.

Which brings us to the promise at the end of [Part 1]({{ site.baseurl }}{% post_url 2026-11-07-Get-The-Greeks-Using_Autodiff %}). Libraries like JAX and PyTorch don't actually build those five matrices. So what do they store and compute instead?

The short answer: they never ask for a Jacobian at all. They only ask what a Jacobian *does* to a vector. Every autodiff library is built around two operations: the **Jacobian-vector product (JVP)**, used in forward mode, and the **vector-Jacobian product (VJP)**, used in reverse mode. Everything else is plumbing to record your program and run those two operations through it efficiently.

This post focuses on the VJP. We'll also need a brief look at the JVP, because it's key to understanding how JAX implements the VJP.

## The maths that lets us dodge the matrix bullet

We'll come back to what the vector in a VJP contains as we go.

What we're implementing is a series of nested vector-matrix products:

$$\Big(\big(\cdots\big((\bar{y}^{\top} J^{(L)})\, J^{(L-1)}\big)\cdots\big)\, J^{(2)}\Big)\, J^{(1)}$$

where $$J^{(L)}$$ is the $$L$$th local Jacobian. In Part 1, for example, $$J^{(1)}$$ is the local Jacobian that turns the market inputs into the log-moneyness, variance, standard deviation and discount factor.

In our option example the result is the row vector of Greeks:

$$\bar{x}^{\top} = \Big(\big(\big((1 \cdot J^{(5)})\, J^{(4)}\big)\, J^{(3)}\big)\, J^{(2)}\Big)\, J^{(1)}$$

where we start the calculation with $$\bar{y} = 1$$. This is the **seed**, covered in step 4.

How do we dodge the bullet of building matrices? Transpose $$\bar{x}^{\top}$$ to get

$$\bar{x} = J^{(1)\top}\Big(J^{(2)\top}\big(J^{(3)\top}\big(J^{(4)\top}\big(J^{(5)\top} \cdot 1\big)\big)\big)\Big)$$

$$= \big(f_1^{*} \circ f_2^{*} \circ f_3^{*} \circ f_4^{*} \circ f_5^{*}\big)(1)$$

Our transposed nested vector-matrix products can be expressed as a composition of functions $$f_i^{*}$$, each of which takes a vector (or a vector-like data structure) and returns a vector. Since all we need from $$f_i^{*}$$ is to take a vector and return a vector, we never have to build a matrix. These $$f_i^{*}$$ are called **pullbacks**. The name comes from differential geometry, where it describes which way something travels relative to a function: a pullback carries a sensitivity *back* from a function's output to its inputs.

The same applies to finding the gradient of an MLP's cost function, where $$J^{(L)}$$ is the local Jacobian of a layer.

Forward mode is the same chain applied in the other direction, $$J^{(5)}\big(\cdots\big(J^{(1)}\,\dot{x}\big)\big)$$. We'll come back to it when we look at how JAX builds the backward sweep.

## How libraries implement this

Reverse mode works in seven steps, each explained in its own section below. Along the way we define the autodiff jargon you'll meet in the literature and in library documentation.

1. **Break the formula into small steps.**
2. **Record the calculation.**
3. **Forward sweep.**
4. **Seed the output.**
5. **Backward sweep:** pass sensitivities back.
6. **Add up at fan-outs.**
7. **Read off the gradient.**

The final section explains how JAX builds the backward sweep from forward mode. Compilation and performance come in the next post.

### 1. Break the formula into small steps

Any pricing formula, however long, reduces to a short list of elementary operations: add, multiply, divide, log, exp, square root, erf. These are the **primitives**. For example, $$N(d_1)$$ in the JAX implementation of Black-76 breaks down into about 14 primitives.

Libraries know how to differentiate each primitive. Each one has a **derivative rule**, which calculates how the primitive's output moves when each input moves. The rule applies the primitive's **local Jacobian** (its tiny table of partial derivatives) to a vector, *without ever building the matrix*, unlike Part 1, which wrote the Jacobians out.

Linked together, the steps form a **compute graph**. Inputs (such as Black-76's forward, strike, vol, rate and time) flow in at one end and the price comes out at the other. For Black-76 the graph has the same layered shape as a small neural network.

### 2. Record the calculation

Before anything can be run backwards, the library needs a record of what happened. There are two ways to generate this record:

| Approach | Used by | How it records | Strength | Cost |
| --- | --- | --- | --- | --- |
| Keep a tape as you go | PyTorch, dco/c++, XAD, AADC | **Operator overloading**: inputs are wrapped in a special type, and every operation on it writes a line on a **tape** (a **Wengert list**) | A **dynamic graph**, rebuilt every run, so ordinary `if` statements just work | The library sees only one step at a time |
| Trace the whole program first | JAX | Runs the function once with **tracers**, placeholders with a shape but no value, producing a complete program called a **jaxpr** | Sees the whole program, so it can optimize and transform it | Control flow must be written in JAX's terms |

**A diversion: why must control flow be written in JAX's terms?** A jaxpr is a program in a small, functional, domain-specific language with strict rules:

- **It's a straight list of equations.** Each line has the form `output = primitive input(s)`: one primitive, named inputs, named outputs, and each variable assigned exactly once.
- **It's pure.** There are no side effects and no mutation. Every result depends only on its inputs.
- **Shapes and types are fixed** when the function is traced, not when it runs.
- **It has no Python control flow.** There's no `if`, `for` or `while` on traced values. Branching or looping on data has to be a *primitive* in the list, like everything else.

That's why our `black_price` function chooses between call and put with:

```python
return jnp.where(is_call, call_price, put_price)
```

instead of:

```python
return call_price if is_call else put_price
```

The Python version works only while `is_call` is a plain Python value. It fails as soon as `is_call` is a traced value, for example when pricing a batch of calls and puts with `vmap`. We'll take a closer look at jaxprs in the next post.

### 3. Forward sweep: compute the price and keep notes

Run the formula step by step. Each step produces its ordinary value, the **primal**.

Alongside it, save whatever the backward sweep will need later: the local derivative of each step, or the values needed to compute it. These saved values are the **residuals**. Some cost nothing extra. For example, the derivative of exp is just its output, so the output is reused.

When the sweep finishes you have the output (the option price in our example, or the cost function's value in code that trains an MLP) plus a full set of **notes**. The notes are everything the backward sweep needs that it can't get from the output alone. They come in two parts:

- **The residuals**: the saved values above. In our example, for $$d_1 = \text{numerator} \div \sigma\sqrt{T}$$, the backward rule needs $$\sigma\sqrt{T}$$ and $$d_1$$, so both are kept. For the discount factor $$e^{-rT}$$, the rule needs $$e^{-rT}$$ itself.
- **The record of the steps**: which step used which inputs, in what order. This is the tape in PyTorch or AAD, or the traced jaxpr in JAX.

The notes must be kept until the backward sweep has used them, so they are reverse mode's **memory bill**.

### 4. Seed the output

Give the output a sensitivity of 1: the output (the price in our example) moves one-for-one with itself. This is the **seed**, and every other sensitivity is measured relative to it. It's the 1 in the nested product at the start of the post.

### 5. Backward sweep: pass sensitivities back

The **adjoint** of a value is the sensitivity of the final output to that value. In our example that's $$\partial\,\text{price}/\partial\,\text{value}$$, written with a bar: $$\bar{d}_1$$ is the adjoint of $$d_1$$. If the value were nudged by a small amount, the price would move by roughly that amount times its adjoint. ML libraries call it the **cotangent**.

An adjoint belongs to a value, such as $$d_1$$, $$\sigma\sqrt{T}$$ or an input like $$T$$, not to a step. A primitive is an operation. It has no sensitivity of its own, only a derivative rule. The price's adjoint is the seed, 1, and each input's adjoint is a Greek.

Walk the record from the last step to the first. At each step $$i$$, apply its pullback $$f_i^{*}$$ to the adjoint of the step's output, and pass the resulting adjoints back to the values that step consumed. In a simple chain, each $$f_i^{*}$$ hands its result straight to $$f_{i-1}^{*}$$.

Concretely: at each step, take the adjoint of the step's output, multiply it by the primitive's local derivative with respect to each input, and add the result to that input's adjoint (written `+=`: add to what's already there rather than overwrite it).

For $$d_2 = d_1 - \sigma\sqrt{T}$$ in Black-76, the sweep adds $$\bar{d}_2 \times 1$$ to $$\bar{d}_1$$ and $$\bar{d}_2 \times (-1)$$ to the adjoint of $$\sigma\sqrt{T}$$, on top of anything they've already received from other steps. A value's adjoint is complete only once every step that used it has been processed.

That's the primitive's **vector-Jacobian product (VJP) rule**: the derivative rule that maps the adjoint of a step's output to the adjoints of its inputs. The pullback $$f_i^{*}$$ is that step's VJP rule. Nothing more complicated happens anywhere in the sweep. Each step multiplies a vector by that step's local Jacobian, the matrices from Part 1, without writing the matrix down.

This is why step 2 keeps a record. Without it, the backward sweep wouldn't know the order to visit the steps in, or where to send each adjoint.

### 6. Add up at fan-outs

Some values feed more than one step. In Black-76, for example, the time to maturity $$T$$ appears in $$\sqrt{T}$$, in $$\tfrac12\sigma^2T$$ and in the discount factor $$e^{-rT}$$. This is a **fan-out**. On the backward sweep, that value receives an adjoint back from every step it fed.

Multiplying by local derivatives chains sensitivities along each route in the backward sweep. **Adjoint accumulation**, the `+=` from step 5, adds the routes together where they meet. Together, they implement the chain rule. The chain rule applies because the Black-76 formula, like an MLP, is a composition of functions. Part 1 wrote it out in matrix form, as a product of layer Jacobians; autodiff evaluates the same product one primitive at a time, without building the matrices.

The two are interleaved in one sweep, not separate phases. Contributions are added to a value's adjoint as they arrive. Only once every contribution is in is the total multiplied on to that value's own inputs, and walking the steps in reverse order guarantees this.

This is why $$\partial c/\partial T$$, and so theta, has three terms: one for each route that $$T$$ takes to the price.

### 7. Read off the gradient

These are the Greeks in our case.

When the sweep reaches the start, each input holds its adjoint, and that adjoint is a Greek: delta on the forward, vega on the vol, rho on the rate, and $$\partial c/\partial T$$ on the time input (theta is its negative). One backward sweep fills in all of them.

Asking for one more input's Greek, such as dual delta on the strike, costs essentially nothing, because the sweep reaches it anyway.

The whole set is one row of the function's **global Jacobian**, again *computed without ever building the matrix*.

## How JAX builds the backward sweep

We saw in step 2 that one difference between JAX and PyTorch is how they record the calculation.

Another difference is how they get the VJP rules. PyTorch uses a hand-written VJP ("backward") rule for each operation. JAX mostly writes only forward-mode (JVP) rules, plus transpose rules for a small set of linear operations, and *derives reverse mode from forward mode*. That's a software engineering choice: one set of rules serves every transformation JAX offers.

#### Forward mode and tangents

The **tangent** of a value is how fast that value moves when a chosen input is nudged: $$\partial\,\text{value}/\partial\,\text{input}$$, written with a dot ($$\dot{d}_1$$ is the tangent of $$d_1$$). The input can also be a chosen combination of inputs, such as a parallel shift of a whole curve. A tangent is the mirror image of an adjoint. A tangent follows one nudge forward to every value, while an adjoint measures how much every value matters to the one final price. That pairing is why the adjoint is also called the **cotangent**.

It come as a no shock that **Forward mode** works the other way round from reverse mode. You nudge one input with a tangent seed of 1. That tangent is carried through every step alongside the primal, using each primitive's **Jacobian-vector product (JVP)** rule, and one Greek comes out at the end.

Using the maths from the start of the post, a forward-mode calculation of an option price sensitivity looks like:

$$\dot{C} = J^{(5)}\Big(J^{(4)}\big(J^{(3)}\big(J^{(2)}\big(J^{(1)}\,\dot{x}\big)\big)\big)\Big), \qquad \dot{x} = (\dot{F}, \dot{K}, \dot{T}, \dot{r}, \dot{\sigma})$$

where $$\dot{x}$$ is the vector of tangent seeds and $$\dot{C}$$ is the price's tangent. With a unit seed, $$\dot{C}$$ is a Greek: to calculate delta, set $$\dot{x} = (1, 0, 0, 0, 0)$$, that is $$\dot{F} = 1$$ and the others 0.

Written as a composition of functions, as at the start of the post:

$$\dot{C} = \big(f_{5*} \circ f_{4*} \circ f_{3*} \circ f_{2*} \circ f_{1*}\big)(\dot{x})$$

These $$f_{i*}$$ are called **pushforwards**, the partner term to pullbacks in differential geometry. A pushforward carries a nudge *forward* along the function, from inputs to output; a pullback carries a sensitivity *back* against it. Like pullbacks, pushforwards are just functions that take a vector-like data structure and return one. Step $$i$$ has both: its pushforward $$f_{i*}$$ applies $$J^{(i)}$$, and its pullback $$f_i^{*}$$ applies $$J^{(i)\top}$$.

*(Aside: we could use forward mode to get the same answers as reverse mode, by running our example once with each of $$\dot{F}, \dot{K}, \dot{T}, \dot{r}, \dot{\sigma}$$ set to 1 and the others set to 0. But that means running it five times instead of once. That's why, for many inputs and few outputs, we prefer reverse mode to forward mode.)*

#### Linearising: getting a linear map

Before we can transpose, we need a linear map. JAX gets one by running the JVP rules at today's market data, with every primal value (the inputs and all the intermediate results) computed and fixed, and only the tangents left free.

Each step's nonlinear work (log, exp, erf) is done once, leaving that step's local Jacobian as fixed numbers: the residuals. What's left is a program that only scales and adds tangents, computing $$J \cdot \dot{x}$$ for any tangent $$\dot{x}$$. In other words, we've linearised the pushforwards.

Let's look at concrete scalar step like $y = x^2$ (specifically, the $\sigma^2$ step inside Black-76) , the JVP rule is $\dot{y} = 2x\,\dot{x}$. 

Evaluated at $x = 3$, the primal $y = 9$ and the local derivative $2x = 6$ are computed once and fixed, leaving the linear map $\dot{y} = 6\,\dot{x}$ with `6` saved as the residual.

#### Transposing: from pushforwards to pullbacks

A linear program can be run backwards mechanically. A step that multiplied a tangent by a constant on the way forward ($$\dot{z} = c\,\dot{x}$$) multiplies the cotangent by the same constant on the way back ($$\bar{x} \mathrel{+}= c\,\bar{z}$$): same number, opposite direction. Every fan-out becomes an addition, and every addition becomes a copy.

Continuing our $y = x^2$ example—the exact operation that computes $\sigma^2$ inside Black-76—the linearised step $\dot{y} = 6\,\dot{x}$ is a tiny $1 \times 1$ local Jacobian $J^{(i)} = [6]$, representing the pushforward $f_{i*}$. Transposing it gives $J^{(i)\top} = [6]$, the pullback $f_i^{*}$, which sends the sensitivity backward via $\bar{x} \mathrel{+}= 6\,\bar{y}$.

Our full Greeks calculation is simply a chain of these steps: five local Jacobians multiplied together, $J^{(5)} J^{(4)} J^{(3)} J^{(2)} J^{(1)}$, one per layer of the model. Transposing each step individually and running them from last to first computes the transpose of the entire chain, because transposing a product reverses its order:

$$\big(J^{(5)} J^{(4)} J^{(3)} J^{(2)} J^{(1)}\big)^{\top} = J^{(1)\top} J^{(2)\top} J^{(3)\top} J^{(4)\top} J^{(5)\top}$$

Applied to the output's adjoint $$\bar{y}$$:

$$\bar{x} = J^{(1)\top}\Big(J^{(2)\top}\big(J^{(3)\top}\big(J^{(4)\top}\big(J^{(5)\top}\, \bar{y}\big)\big)\big)\Big) = \big(f_1^{*} \circ f_2^{*} \circ f_3^{*} \circ f_4^{*} \circ f_5^{*}\big)(\bar{y})$$

Set $$\bar{y} = 1$$ and we've got our reverse-mode calculation of the Greeks. By transposing, we've turned each pushforward $$f_{i*}$$ into its pullback $$f_i^{*}$$.

*TL;DR: reverse mode applies the transpose of the same Jacobian that forward mode applies, so one backward pass gives every Greek.*

When we look at jaxprs in the next post, we'll show how this is implemented in practice.

## Takeaway

An autodiff library never builds the Jacobians from Part 1. It breaks the formula into primitives, each with a derivative rule that applies its local Jacobian to a vector.

Reverse mode runs a forward sweep that computes the price and keeps notes: the residuals and a record of the steps (a tape in PyTorch, a traced jaxpr in JAX). It then seeds the output with 1 and walks the record backwards. Each step's pullback, its VJP rule, multiplies the incoming adjoint by local derivatives, and adjoint accumulation adds the contributions wherever a value fanned out. Multiply along routes, add where they meet: that is the chain rule, evaluated one primitive at a time. When the sweep reaches the inputs, their adjoints are the Greeks, all of them from one pass. The cost is memory: every residual must be kept until the backward sweep uses it.

Forward mode is the same chain in the other direction. Pushforwards carry a tangent from the inputs to the price, one Greek per pass, which is why it suits few inputs and many outputs.

JAX connects the two. It writes the forward-mode rules, linearises them at today's market data, and transposes the resulting linear program to get the pullbacks. Reverse mode applies the transpose of the same Jacobian that forward mode applies.

### Coming up in Part 3

Part 3 takes this machinery out of the abstract and puts it to work on the same Black-76 example:

- **Reading the gradient program.** We print the jaxpr that `grad` actually generates and find every idea from this post in it: residuals saved on the forward sweep, a backward sweep made only of linear operations, fan-outs appearing as `add_any`, theta's three routes summed line by line, and the zero that `jnp.where` sends down the unused branch.
- **Compilation.** We hand that program to XLA, Google's optimizing compiler, and see what fusion and common subexpression elimination buy. For a batch of 100,000 options, the price plus a full set of Greeks costs about three prices. Bumping costs about six, and un-compiled autodiff is slower still. We also look at what the memory bill means for Monte Carlo, and how `jax.checkpoint` trades memory for recomputation.
- **Composing transformations.** Differentiating a derivative gives gamma, `jax.hessian` gives gamma, vanna and volga in one line, and `vmap` turns the single-option code into a whole strike ladder.
- **Checking the answers** against Part 1's closed forms, to every printed digit.
- **The gotchas that matter for pricing:** why you need float64, why autodiff's "rho" may not be the rho a trader means, what happens at expiry and other non-differentiable points, and how branches can leak a `NaN` into your Greeks.

