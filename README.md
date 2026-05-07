# Neural Network Visualizer

**Live demo → [kickereb.github.io/nn-visualizer](https://kickereb.github.io/nn-visualizer)**

Two interactive experiments that let you build, configure, and train neural networks in real time — directly in the browser, with no libraries, no build step, and no server.

---

## Experiments

### [Experiment 01 — 2D Regression](experiment-2d.html)
A configurable neural network fits a noisy sine wave.

- 1 input → N hidden layers → 1 output
- Residual lines from each data point to the curve visualize MSE live
- Manual weight sliders + live gradient descent training
- Watch the curve bend into shape as it trains

### [Experiment 02 — 3D Surface Fitting](experiment-3d.html)
Two inputs map to one output — the network's prediction becomes a **surface**.

- 2 inputs → N hidden layers → 1 output
- Interactive rotatable 3D plot (Plotly)
- Vertical residual lines from each data point to the surface
- Target: `y = sin(x₁) · cos(x₂) + noise`

---

## Architecture

Both experiments share the same core design:

| Component | Detail |
|-----------|--------|
| Hidden activation | `tanh` |
| Output activation | Linear (none) |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Full-batch gradient descent |
| Weight init | Xavier-ish: `±√(1/fanIn) × scale` |
| Learning rate | 0.03 (2D) · 0.04 (3D) |
| Max hidden layers | 4 |
| Max neurons/layer | 8 (2D) · 10 (3D) |

**Why tanh over sigmoid for hidden layers?**
tanh is zero-centered (range [−1, 1]) which keeps gradients symmetric and speeds up training compared to sigmoid's [0, 1] output. For regression, the linear output layer is essential — sigmoid would clamp predictions to [0, 1] and the network couldn't fit arbitrary y values.

---

## File structure

```
nn-visualizer/
├── index.html          # Landing page with links to both experiments
├── experiment-2d.html  # 2D regression experiment
├── experiment-3d.html  # 3D surface fitting experiment
└── README.md
```

No npm, no bundler, no framework. Pure HTML + JS + two CDN scripts (Chart.js for 2D, Plotly for 3D).

---

## Hosting on GitHub Pages

1. Fork or clone this repo
2. Go to **Settings → Pages**
3. Source: **Deploy from a branch** → `main` → `/ (root)`
4. Save — your site will be live at `https://YOUR_USERNAME.github.io/nn-visualizer` within ~60 seconds
5. Replace all `YOUR_USERNAME` placeholders in `index.html` and this README

---

## How the training works

Both experiments implement **backpropagation from scratch** in plain JavaScript — no ML libraries. Here's the forward pass in pseudocode:

```
for each layer l:
  for each neuron n:
    z = bias + Σ(weight[p] × activation[p])   // weighted sum
    a = tanh(z)                                 // activation (hidden)
    a = z                                       // activation (output, linear)
```

The backward pass computes δ per neuron using the chain rule:

```
δ_output = 2 × (ŷ − y) / N                    // MSE gradient
δ_hidden = (Σ δ_next × w) × (1 − a²)          // tanh derivative
```

Weights update via gradient descent:

```
w = w − lr × (Σ δ × a_prev)
b = b − lr × δ
```

---

## Why this is interesting

- **Universal approximation theorem**: With enough neurons, even a single hidden layer can approximate any continuous function. Try 1 layer with 6+ neurons on the 2D experiment — you'll see it fit surprisingly complex curves.
- **Vanishing gradients**: Try 5 layers × 2 neurons and randomize. You'll occasionally get a flat output — the sigmoid-chain squashing problem, visible in the forward pass.
- **Depth vs width**: Compare 1 × 6 vs 3 × 2 neurons. Same parameter count, different expressivity.

---

## Credits

Built as an educational visualization for a LinkedIn post about how neural networks generalize beyond the classic sigmoid diagram.
Target audience: ML practitioners who want to build intuition for what networks are actually learning.
