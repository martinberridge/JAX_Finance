---
layout: post
title: "Getting the Greeks from autodiff: why options look like neural networks"
date: 2026-09-10
categories: [quant-finance, autodiff]
series: autodiff-for-greeks
part: 1
---# Getting the Greeks from autodiff: why options look like neural networks

If you've built software to price options, you've computed Greeks one of two ways: bumping a volatility input by a basis point and repricing, or — if you're working with a complex closed-form formula like Geske's (1979) compound option model — differentiating it by hand and watching the algebra spiral into something gnarly.

Bump-and-reprice and hand-differentiated analytics have been standard practice for decades.

But there's a third way to get a derivative, and it happens to be the same machinery that trains every neural network in production today: automatic differentiation. Researchers like Antoine Savine and Marc Henrard have made the case that, once you look closely, an option pricing formula turns out to be structurally the same kind of computation as a neural network. Same graph shape, same gradient machinery, same tricks for making it fast. This post walks through why.

## Why borrow from machine learning at all?

The instinct might be to reach for an ML *model* — train something to predict prices or Greeks from data. That's not what this is about. The useful thing to borrow isn't the model, it's the plumbing underneath it.

When a neural network trains, it doesn't compute a gradient by guessing and checking, or by writing out a symbolic derivative of the whole network by hand. It computes the sensitivity of a loss function to every parameter in the model, automatically, using the network's own compute graph. That mechanism — sensitivity of an output with respect to inputs, computed automatically over a graph — has nothing to do with prediction or training data. It's a general-purpose tool. Option pricing formulas have compute graphs too, and the same mechanism applies to them.

## Three ways to get a derivative

Before getting into autodiff specifically, it's worth being clear about the alternatives, because each has a real cost.

**Symbolic differentiation** gives you an exact closed-form derivative. The problem is that differentiating a formula by hand — or asking a computer algebra system to do it — tends to produce something far messier than the original. Small formulas can expand into monstrous ones after a few rounds of the product and chain rules. This is usually called *expression swell*, and it's the reason symbolic derivatives of anything beyond a toy formula quickly become impractical to evaluate.

**Numerical differentiation** sidesteps the algebra entirely: bump an input by a small amount, reprice, and divide the difference by the bump size. It's simple and it works on anything you can price, including models with no closed form at all. The tradeoff is a direct accuracy-versus-speed tension — too large a bump and you get a biased estimate of the true derivative, too small a bump and floating-point rounding error dominates. And if you need sensitivities with respect to many inputs, you need to reprice once per input, which gets expensive fast.

**Automatic differentiation** is neither of these. It doesn't produce a symbolic expression, and it doesn't approximate anything numerically. Instead, it decomposes your calculation into a sequence of elementary operations, tracks how each one propagates a derivative, and applies the chain rule mechanically through that sequence. The result is a derivative that's exact to machine precision, computed at a cost comparable to the original calculation itself. It gives you the accuracy of symbolic differentiation without the expression swell, and the generality of numerical differentiation without the bump-size tradeoff.

Autodiff comes in two flavors — forward mode and backward mode — and which one you want depends entirely on the shape of your problem.

## Forward mode: cheap when inputs are few

Forward mode propagates derivative information in the same direction as the original calculation, from inputs toward outputs. It computes, for a single chosen input, how every intermediate quantity — and eventually every output — responds to it. This makes forward mode efficient precisely when you have **few inputs and many outputs**.

Two ways to picture this shape:

- **A robot with one control input and many sensors.** Turn a single dial, and dozens of sensor readings shift in response. Forward mode is the natural way to ask "if I move this one control, how does every sensor respond?" — you sweep the effect of one input forward through the whole system in a single pass.
- **A portfolio of bonds and swaps.** Move a single interest rate, and every instrument's price and every risk number in the portfolio shifts at once. One input, many outputs.

The formal name for what forward mode computes is a **Jacobian-vector product**: given a direction in input space (a single input, or a small combination of them), it produces how the entire output vector moves in response, in one pass.

## Backward mode: cheap when outputs are few

Backward mode runs the opposite direction: it starts at a single output and propagates backward to find how that output depends on every input. This is the mirror image of forward mode's use case — it's efficient when you have **many inputs and one (or few) outputs**.

This is exactly the shape of the machine learning training problem: a network might have millions of parameters (many inputs) feeding into a single scalar loss (one output). Computing the gradient of that loss with respect to every parameter, in one backward sweep, is what the training literature calls **backpropagation** — and it's just backward-mode autodiff applied to a neural network's compute graph.

The formal name here is a **vector-Jacobian product**. Structurally, it's an abstraction over exactly the quantities that matter for hedging: feed in the way you want the output to move (or simply ask for its raw sensitivity), and backward mode hands you back the gradient — which, in a pricing context, is precisely what the Greeks are.

Backward mode also has a natural local/global split. At each individual layer or operation, the computation only needs to know its own local derivative. Stitching those local derivatives together correctly across the entire graph, from output back to every input, is just the chain rule applied globally, layer after layer. The Black-76 example below walks through exactly this, layer by layer.

## A short detour: what is a Jacobian, really?

Both modes are ultimately computing pieces of the same object: the Jacobian, the matrix of all partial derivatives of a set of outputs with respect to a set of inputs. It's worth building intuition for what that matrix actually represents, because it clarifies why forward and backward mode are two different ways of extracting information from the same structure.

A useful analogy comes from computer graphics. An ordinary transformation matrix — the kind used to move or resize a shape — acts globally: every point in the image is moved or scaled by exactly the same rule. A Jacobian is a *local* version of that idea. Instead of one transformation for the whole space, it describes how a nonlinear function behaves in the immediate neighborhood of one particular point — warping and distorting space differently depending on where you are. Zoom into any single point on a curved, nonlinear function and, locally, it looks approximately linear; the Jacobian is that local linear approximation.

This is exactly the sense in which forward and backward mode work: neither one computes a single global "slope" for your entire pricing formula. Each computes local derivatives, layer by layer, and only assembles them into the full input-to-output sensitivity by chaining those local pieces together — which is the chain rule, applied globally across the graph.

## Options, Greeks, and hedging

With that machinery in hand, it's worth stating the obvious, the Greeks are the sensitivities of an option's price to its inputs (spot, volatility, rate, time)  and the object being computed is a derivative — which sets up the natural question: why not get it from autodiff instead, the same way a neural network gets its gradients?

## The payoff: an option price is an MLP

Here's where the two threads — autodiff and options — meet directly. Line up the compute graph of a simple multilayer perceptron next to the compute graph of an option pricing formula, and the structural resemblance is hard to miss:

**An MLP's layers:**

- *Input* — the training set (the values fed into the network)
- *Hidden* — weights, sensitivities, and an activation function transforming values layer to layer
- *Output* — a cost function, the single scalar the network is ultimately being evaluated against

**An option pricing formula's layers:**

- *Input* — market and contract parameters (spot, strike, volatility, rate, time to expiry)
- *Hidden* — a sequence of intermediate transformations: log-moneyness and variance, feeding into probability-flavored metrics like delta, feeding into probabilities of finishing in-the-money and of a given value of the underlying, feeding into an expected intrinsic value built from the expected forward value and expected cost of exercise
- *Output* — the option's price or premium, a single number

Both are directed graphs of simple operations, chained together, terminating in one scalar output computed from many inputs. That is precisely the shape backward mode is built for. Running reverse-mode autodiff through an option pricing formula's compute graph doesn't just give you the price — in the same backward pass that computes the price's sensitivity to itself, it gives you the price's sensitivity to every input simultaneously. In other words: every Greek, in roughly the cost of one backward pass, the same way backpropagation gives a neural network every parameter gradient in one pass.

### Example: Black-76

We'll walk through the Black-76 model as a five-layer network. For each layer below, we first define the forward pass — what the layer computes — then immediately give its local Jacobian, the derivative of that layer's outputs with respect to its own inputs. Chaining those five local Jacobians together at the end gives every Greek in one pass.

**A note on notation.** For layer $l$, $\mathbf{x}^{(l)}$ is the vector of values flowing *into* that layer and $\mathbf{y}^{(l)}$ is the vector flowing *out*. The local Jacobian $J^{(l)} = \dfrac{\partial \mathbf{y}^{(l)}}{\partial \mathbf{x}^{(l)}}$ collects every partial derivative of that layer's outputs with respect to its inputs, with entry $J^{(l)}_{ij} = \dfrac{\partial y_i}{\partial x_j}$. Concretely:

- **Rows** correspond to the layer's *outputs* — one row per output. Reading across a row tells you how that one output responds to every input.
- **Columns** correspond to the layer's *inputs* — one column per input. Reading down a column tells you how every output responds to that one input.
- A Jacobian with $m$ outputs and $n$ inputs is an $m \times n$ matrix — this is why the shape changes from layer to layer below.

The superscript $(l)$ just tags "which layer" and is not an exponent. Where a layer's own outputs get names later (like $d_1$, $d_2$, or $N(d_1)$), those names are used directly in place of $y_i$ once they exist, to keep each layer's own notation readable.

**Inputs (5 nodes)** — the market and contract parameters the network accepts:

| Symbol | Meaning |
|---|---|
| $F$ | Forward price of the underlying asset |
| $K$ | Strike price of the option |
| $\sigma$ | Volatility of the forward price |
| $r$ | Risk-free interest rate |
| $T$ | Time to maturity |

#### Layer 1 — intermediate term extractions

**Forward pass.** This layer processes combinations of inputs to prepare components for the $d_1$ and $d_2$ steps:

- **Node 1.1 (log moneyness):** $\ln(F/K)$
- **Node 1.2 (variance):** $\sigma^2 T$
- **Node 1.3 (standard deviation):** $\sigma\sqrt{T}$
- **Node 1.4 (discount factor):** $e^{-rT}$

**Jacobian.** $\mathbf{x}^{(1)} = [F, K, \sigma, r, T]^T \rightarrow \mathbf{y}^{(1)} = [y_{1.1}, y_{1.2}, y_{1.3}, y_{1.4}]^T$, so $J^{(1)}$ is a $4 \times 5$ matrix: 4 rows for the 4 nodes in this layer (log moneyness, variance, standard deviation, discount factor, in that order), 5 columns for the 5 inputs $(F, K, \sigma, r, T)$. Row 1, for instance, is $\partial y_{1.1}/\partial F, \partial y_{1.1}/\partial K, \ldots$ — the sensitivity of log moneyness to each input in turn:

$$J^{(1)} = \begin{bmatrix} \tfrac{1}{F} & -\tfrac{1}{K} & 0 & 0 & 0 \\ 0 & 0 & 2\sigma T & 0 & \sigma^2 \\ 0 & 0 & \sqrt{T} & 0 & \tfrac{\sigma}{2\sqrt{T}} \\ 0 & 0 & 0 & -T e^{-rT} & -r e^{-rT} \end{bmatrix}$$

Most entries are zero simply because most nodes here only depend on one or two of the five inputs — e.g. log moneyness (row 1) only involves $F$ and $K$, so the rest of that row vanishes.

#### Layer 2 — probability metrics ($d_1$ and $d_2$)

**Forward pass.** This layer consolidates layer 1's outputs into the z-scores used for delta and the probability of exercise:

- **Node 2.1:** $d_{1}=\dfrac{\ln (F/K)+\frac{1}{2}\sigma ^{2}T}{\sigma \sqrt{T}}$
- **Node 2.2:** $d_{2}=d_{1}-\sigma \sqrt{T}$

**Jacobian.** $\mathbf{x}^{(2)} = [v_1, v_2, v_3]^T$ (layer 1's first three outputs; $y_{1.4}$ bypasses this layer) $\rightarrow \mathbf{y}^{(2)} = [d_1, d_2]^T$, so $J^{(2)}$ is a $2 \times 3$ matrix: row 1 is $d_1$'s sensitivity to each of $v_1, v_2, v_3$; row 2 is $d_2$'s. Note both rows share the same first two columns, since $d_2 = d_1 - v_3$ inherits $d_1$'s dependence on $v_1$ and $v_2$ unchanged — only the third column differs, by the extra $-1$ from that subtraction:

$$J^{(2)} = \begin{bmatrix} \tfrac{1}{v_3} & \tfrac{1}{2v_3} & -\tfrac{v_1 + \frac{1}{2}v_2}{v_3^2} \\ \tfrac{1}{v_3} & \tfrac{1}{2v_3} & -\tfrac{v_1 + \frac{1}{2}v_2}{v_3^2} - 1 \end{bmatrix} = \begin{bmatrix} \tfrac{1}{\sigma\sqrt{T}} & \tfrac{1}{2\sigma\sqrt{T}} & -\tfrac{d_1}{\sigma\sqrt{T}} \\ \tfrac{1}{\sigma\sqrt{T}} & \tfrac{1}{2\sigma\sqrt{T}} & -\tfrac{d_1}{\sigma\sqrt{T}} - 1 \end{bmatrix}$$

(second form substitutes the original variables back in)

#### Layer 3 — cumulative probabilities (activation layer)

**Forward pass.** A custom activation layer passing $d_1, d_2$ through the standard normal CDF $N(x)$:

- **Node 3.1:** $N(d_1)$ — the delta-exposure component
- **Node 3.2:** $N(d_2)$ — the risk-adjusted probability the option finishes in-the-money

**Jacobian.** $\mathbf{x}^{(3)} = [d_1, d_2]^T \rightarrow \mathbf{y}^{(3)} = [N(d_1), N(d_2)]^T$, so $J^{(3)}$ is a $2 \times 2$ matrix: row 1 is $N(d_1)$'s sensitivity to $d_1$ and $d_2$; row 2 is $N(d_2)$'s. Because this layer applies $N(\cdot)$ to each input independently — $N(d_1)$ never depends on $d_2$, and vice versa — the off-diagonal entries are both zero and $J^{(3)}$ is purely diagonal, built from the standard normal PDF $n(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$:

$$J^{(3)} = \begin{bmatrix} n(d_1) & 0 \\ 0 & n(d_2) \end{bmatrix}$$

#### Layer 4 — intrinsic components

**Forward pass.** Parallel multiplications weighting the forward position and strike obligation:

- **Node 4.1:** $F \cdot N(d_1)$ — forward value component
- **Node 4.2:** $K \cdot N(d_2)$ — exercise cost component

**Jacobian.** $\mathbf{x}^{(4)} = [F, K, N(d_1), N(d_2)]^T$ ($F$, $K$ brought forward via skip connections) $\rightarrow \mathbf{y}^{(4)} = [u_1, u_2]^T$, so $J^{(4)}$ is a $2 \times 4$ matrix: row 1 is $u_1 = F \cdot N(d_1)$'s sensitivity to each of the 4 inputs, row 2 is $u_2 = K \cdot N(d_2)$'s. Since $u_1$ only involves $F$ and $N(d_1)$, its row is zero in the $K$ and $N(d_2)$ columns — and symmetrically for $u_2$:

$$J^{(4)} = \begin{bmatrix} N(d_1) & 0 & F & 0 \\ 0 & N(d_2) & 0 & K \end{bmatrix}$$

#### Output layer — present value

**Forward pass.** A single terminal node subtracts the exercise cost component from the forward value component and discounts back to present value:

$$c=e^{-rT}\left[FN(d_{1})-KN(d_{2})\right]$$

**Jacobian.** $\mathbf{x}^{(5)} = [u_1, u_2, e^{-rT}]^T \rightarrow y^{(5)} = c$. There's only one output here — the price $c$ — so $J^{(5)}$ collapses to a single row: a $1 \times 3$ gradient vector, with one entry per input giving $c$'s sensitivity to that input:

$$J^{(5)} = \begin{bmatrix} e^{-rT} & -e^{-rT} & u_1 - u_2 \end{bmatrix}$$

### Chaining it together: the global Jacobian

Multiplying the five local Jacobians via the chain rule, $J^{(5)} \cdot J^{(4)} \cdot J^{(3)} \cdot J^{(2)} \cdot J^{(1)}$, gives a **$1 \times 5$** row vector — the sensitivity of the option price $c$ to each of the five inputs. In quantitative finance, these are the Greeks:

$$J_{\text{global}} = \frac{\partial c}{\partial \mathbf{x}^{(1)}} = \begin{bmatrix} \frac{\partial c}{\partial F} & \frac{\partial c}{\partial K} & \frac{\partial c}{\partial \sigma} & \frac{\partial c}{\partial r} & \frac{\partial c}{\partial T} \end{bmatrix} = \begin{bmatrix} \Delta & \text{Dual }\Delta & \mathcal{V} & \rho & \Theta \end{bmatrix}$$

Carrying out that multiplication — and using the Black-76 identity $F \cdot n(d_1) = K \cdot n(d_2)$ to cancel internal terms — gives the closed-form Greeks directly:

| Greek | Sensitivity to | Formula |
|---|---|---|
| Delta ($\Delta$) | Forward price | $\dfrac{\partial c}{\partial F} = e^{-rT} N(d_1)$ |
| Dual delta | Strike price | $\dfrac{\partial c}{\partial K} = -e^{-rT} N(d_2)$ |
| Vega ($\mathcal{V}$) | Volatility | $\dfrac{\partial c}{\partial \sigma} = e^{-rT} F \sqrt{T}\, n(d_1)$ |
| Rho ($\rho$) | Risk-free rate | $\dfrac{\partial c}{\partial r} = -T e^{-rT}\left[F N(d_1) - K N(d_2)\right] = -T \cdot c$ |
| Theta ($\Theta$) | Time to maturity | $\dfrac{\partial c}{\partial T} = e^{-rT}\left[\dfrac{F n(d_1) \sigma}{2\sqrt{T}} + r K N(d_2) - r F N(d_1)\right]$ |

## The forward pass: populating these Jacobian matrices

Both forward mode and reverse mode start with a forward pass: computing the function you actually want to differentiate. This step matters for a reason that's easy to gloss over — the derivative of a nonlinear function isn't a fixed number, it depends on *where* you evaluate it. The derivative of $x^2$ is $2x$, which requires knowing $x$. So a Jacobian can't be computed in a vacuum; it has to be evaluated at a specific numerical point, and the forward pass is what supplies that point.

Every entry in the Black-76 Jacobians above — the $\frac{1}{F}$ term in $J^{(1)}$, the $N(d_1)$ term in $J^{(4)}$, and so on — is only a formula until actual numbers for $F, K, \sigma, r, T$ are plugged in. The forward pass is what produces those numbers, layer by layer.

The two modes use that forward pass differently, though:

- **In forward mode,** the Jacobian is computed *alongside* the forward pass itself — as each intermediate value is calculated, its derivative with respect to the chosen input is calculated right along with it. The forward pass *is* the Jacobian calculation.
- **In reverse mode,** the forward pass instead gathers the raw ingredients the Jacobian calculation will need — every intermediate value, at every node, gets stored as it's produced. The actual Jacobian entries are only computed afterward, working backward through the layers using those stored values — this is the backward pass, chaining the local Jacobians the way $J^{(5)} \cdot J^{(4)} \cdot \ldots \cdot J^{(1)}$ was chained above.

We'll go into more detail on this in the next post, when we look at how particular machine learning libraries implement forward and reverse mode.

## Real-world implementation

How are these calculations actually implemented in machine learning libraries? One question remains, given how the forward pass populates each Jacobian above: when these same techniques are used to train a modern LLM, don't we end up multiplying matrices with hundreds of millions of elements?

In practice, libraries like PyTorch and JAX don't materialize full Jacobian matrices the way we've written them out above. Instead, they implement algorithms that *behave like* matrix and vector multiplication and Jacobian evaluation, computing Jacobian-vector products and vector-Jacobian products directly — without ever forming the intermediate matrix. That's what keeps autodiff tractable at the scale of a network with hundreds of millions of parameters.

The next post takes a closer look at these implementations and explores JAX in particular, including an option pricing example implemented in JAX.

## Takeaway

Forward mode wins when you have few inputs and many outputs — sweeping one interest rate across a whole portfolio. Backward mode wins when you have many inputs and one output — and pricing a single option from a handful of market and contract parameters down to one premium is squarely that shape. Once you see the pricing formula as a compute graph instead of a formula on a page, reaching for autodiff isn't borrowing a trick from machine learning — it's just recognizing that you already had the right kind of graph for it.
