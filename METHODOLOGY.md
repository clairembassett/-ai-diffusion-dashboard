# Methodology — Equations and Estimation

**AI Investment & Diffusion Monitor**
*Four-stage research dashboard · 2019–2030*

This document specifies every equation used in the dashboard, defines each parameter, and explains the role each model plays in the argumentative arc (capacity → concentration → forecast → labor).

---

## 0. Notation

| Symbol | Meaning | Unit |
|---|---|---|
| $I_{c,t}$ | Private AI investment in country $c$, year $t$ | USD billions |
| $G_{c,t}$ | Real GDP of country $c$, year $t$ | USD billions |
| $I_t$ | Global private AI investment, year $t$ | USD billions |
| $\tau_{c,t}$ | AI investment intensity (country $c$, year $t$) | % of GDP |
| $A_{s,t}$ | Cumulative adoption share in sector $s$, year $t$ | unitless, $\in [0, 1]$ |
| $v_s$ | Sector $s$'s share of global output | unitless, $\in [0, 1]$ |
| $g_{s,t}$ | Bass diffusion velocity in sector $s$, year $t$ | per year |
| $\alpha_s$ | Projected share of 2026–2030 investment flow going to sector $s$ | unitless, $\in [0, 1]$ |
| $K$ | Logistic carrying capacity (annual ceiling for global flow) | USD billions |
| $r$ | Logistic intrinsic growth rate | per year |
| $t_0$ | Logistic inflection point | calendar year |
| $p_s, q_s$ | Bass innovation and imitation coefficients for sector $s$ | per year |

---

## 1. Stage 01 — Capacity: Investment Intensity

**Purpose.** A country's ability to absorb and diffuse AI depends on the scale of committed private capital relative to the size of its economy. Raw investment in dollars conflates "big country" with "AI-intensive country." Scaling by real GDP separates the two.

**Definition.**

$$\tau_{c,t} = \frac{I_{c,t}}{G_{c,t}} \times 100$$

**Interpretation.** $\tau_{c,t}$ is the share of a country's real economic output that takes the form of private AI investment in year $t$. Higher values mean more productive capacity is being committed to AI, which proxies for the institutional, talent-market, and financial-market capacity to support AI diffusion.

**Data.** $I_{c,t}$ from Stanford AI Index 2025 (2019–2024) and Stanford AI Index 2026 (2025). $G_{c,t}$ from IMF World Economic Outlook and World Bank; 2025 values projected from IMF growth rates applied to 2024 base estimates.

**No parameters are fitted at this stage** — it is a pure accounting ratio.

---

## 2. Stage 02 — Concentration: Descriptive Aggregation

Stage 2 is descriptive — it pairs global investment levels $I_t$ with industry and function adoption shares $A_{s,t}$ from survey data (McKinsey, IBM, Stanford AI Index). No model is estimated here; the purpose is to observe where flows are concentrating before moving to the forecast.

The adoption data feeds directly into the Bass model in Stage 3, so the sector/function percentages are inputs, not outputs, of the statistical machinery.

---

## 3. Stage 03a — Forecast: Logistic (S-curve) Diffusion Model

**Purpose.** Project total global private AI investment through 2030 in a way that is interpretable and economically disciplined — not a neural network or black-box trend.

### 3.1 The equation

$$I(t) = \frac{K}{1 + e^{-r(t - t_0)}}$$

This is the three-parameter logistic growth function of Verhulst (1838), widely applied to technology diffusion since Rogers (1962) and Mansfield (1961). It produces an S-shape: slow start, rapid acceleration through the midpoint, gradual deceleration toward a ceiling.

### 3.2 Parameter meanings

| Parameter | Meaning | Fitted value |
|---|---|---|
| $K$ | **Carrying capacity** — the asymptotic ceiling the annual flow approaches as $t \to \infty$. Economically interpreted as the maximum annual private investment that global capital markets will sustainably allocate to AI given macro constraints. | $1{,}200$ USD B |
| $r$ | **Intrinsic growth rate** — how quickly investment grows in the early exponential phase and how steeply it accelerates through the inflection. | $0.447$ / yr |
| $t_0$ | **Inflection point** — the calendar year at which annual-flow growth is fastest. After $t_0$, growth continues but decelerates. | $2027.1$ |

At $t = t_0$ the function equals exactly $K/2$. This is the most useful interpretive checkpoint: if the fitted $t_0$ is $2027.1$ and $K = \$1{,}200$B, the model is saying that 2027 is when annual flow passes $\$600$B, halfway to the ceiling.

### 3.3 Why logistic and not exponential

A pure exponential $I(t) = I_0 e^{rt}$ has no ceiling, so extrapolation produces physically impossible forecasts (e.g. AI investment exceeding global GDP within 15 years). The logistic encodes the economic reality that investment growth must slow as the sector matures — capital constraints, diminishing returns on additional compute, and market saturation all pull the curve toward an asymptote.

### 3.4 Fitting procedure

The three parameters are estimated by nonlinear least squares on the seven observed global values $(t_i, I_i)$ for $t \in \{2019, \dots, 2025\}$:

$$\hat{\theta} = \arg\min_{\theta} \sum_{i=1}^{7} \left[ I_i - \frac{K}{1 + e^{-r(t_i - t_0)}} \right]^2$$

where $\theta = (K, r, t_0)$. Implemented with `scipy.optimize.curve_fit`.

**Bounds applied.** $K \in [500, 1200]$, $r \in [0.30, 0.75]$, $t_0 \in [2024, 2030]$. The upper bound on $K$ is anchored externally to Gartner's 2026 forecast of $\$2.5$T in **total** AI spending — private investment has historically run at ~20–30% of total spending, implying a private ceiling in the range of $\$500{\text{B}} – \$750$B, conservatively widened to $\$1.2$T. Without this anchor, six or seven early-exponential data points cannot identify the ceiling — the optimiser walks off toward arbitrarily large $K$.

### 3.5 Fit quality

| Metric | Value |
|---|---|
| $R^2$ | $0.949$ |
| RMSE | $\$23.6$B |
| $n$ | 7 |

### 3.6 Confidence intervals (residual bootstrap)

The 80% confidence band shown in the forecast chart is generated by **residual bootstrap** rather than asymptotic standard errors — because with only seven observations, the asymptotic assumptions underlying standard parametric CIs are weak.

Let $\hat{I}_i$ denote the fitted value and $e_i = I_i - \hat{I}_i$ the residual. For bootstrap replicate $b \in \{1, \dots, 1000\}$:

1. Sample residuals $e_i^{*b}$ with replacement from $\{e_1, \dots, e_7\}$.
2. Construct pseudo-data $I_i^{*b} = \hat{I}_i + e_i^{*b}$.
3. Refit $(K^{*b}, r^{*b}, t_0^{*b})$ by nonlinear least squares on the pseudo-data.
4. Project forward to get $I^{*b}(t)$ for $t \in \{2026, \dots, 2030\}$.

The 10th and 90th percentiles of $\{I^{*b}(t)\}_{b=1}^{1000}$ define the 80% CI at each forecast year. Wider bands in later years reflect genuine parameter uncertainty compounding forward.

---

## 4. Stage 03b — Allocation: Bass Diffusion Velocity × Output Weight

**Purpose.** The logistic model forecasts the *total* annual flow of global AI investment. The allocation model distributes that flow across the 15 industries.

### 4.1 Bass diffusion — the full model

The Bass (1969) model of new-product diffusion specifies that the instantaneous rate of adoption depends on two forces: external influence (advertising, investment spillovers, media coverage) and internal influence (imitation of earlier adopters by peers).

$$\frac{dA_s}{dt} = \big[\, p_s + q_s \cdot A_s(t) \,\big] \cdot \big[\, 1 - A_s(t) \,\big]$$

where:

- $A_s(t)$ is the cumulative adoption share in sector $s$ at time $t$, constrained to $[0, 1]$.
- $p_s$ is the **coefficient of innovation** — captures external push (R&D, advertising, investment-driven supply of AI tools). Independent of current adoption level.
- $q_s$ is the **coefficient of imitation** — captures peer effects. Scales with $A_s(t)$, so imitation pressure grows as more firms in the sector adopt.
- $(1 - A_s(t))$ is the **remaining market**. As saturation approaches, the rate of new adoption slows regardless of $p$ and $q$.

### 4.2 Bass velocity (what we actually use)

The velocity $g_{s,t}$ is $dA_s/dt$ evaluated at the current adoption share $A_s(t)$:

$$g_{s,t} = \big[\, p_s + q_s \cdot A_{s,t} \,\big] \cdot \big[\, 1 - A_{s,t} \,\big]$$

**Interpretation.** $g_{s,t}$ is the per-year increase in adoption share the sector would experience next year, holding $p_s, q_s$ constant. Higher $g$ = more remaining diffusion still to happen per unit time.

### 4.3 Sector-specific $(p_s, q_s)$ calibration

Calibrated from the 2022–2025 adoption panel. Sectors with strong early uptake get larger $p$; sectors showing rapid peer-driven growth get larger $q$. Ranges are consistent with empirical Bass estimates in the technology-diffusion literature (Bass 1969, Sultan, Farley & Lehmann 1990).

| Sector | $p$ | $q$ |
|---|---|---|
| Technology / Software | 0.08 | 0.45 |
| Financial Services | 0.06 | 0.42 |
| Professional Services | 0.07 | 0.43 |
| Healthcare & Pharma | 0.04 | 0.38 |
| Manufacturing | 0.04 | 0.38 |
| Agriculture | 0.02 | 0.28 |

(Full table in `index.html` source.)

### 4.4 Allocation share — the combined equation

A sector's share of the projected 2026–2030 investment flow combines its Bass velocity (how much adoption is still happening) with its share of global output (how big the revenue pool is):

$$\alpha_s = \frac{v_s \cdot g_{s,t^\star}}{\sum_{j=1}^{15} v_j \cdot g_{j,t^\star}}$$

where $t^\star = 2025$ (the last observed adoption year) and shares sum to one: $\sum_s \alpha_s = 1$.

**Why multiply velocity by output share?** Velocity alone would over-weight small sectors with fast adoption growth. Output share alone would over-weight large sectors regardless of AI maturity. The product captures both at once — a sector gets a large allocation only if both its velocity *and* its output share are sizeable. This is why Manufacturing ranks high (output share $\approx 15\%$, moderate velocity) even though Tech has faster velocity but a smaller output share.

---

## 5. Stage 04 — Labor: OLS Regression on Adoption vs Employment Projection

**Purpose.** Quantify the association between a sector's 2025 AI adoption rate and its BLS-projected 10-year employment growth.

### 5.1 Specification

$$y_s = \beta_0 + \beta_1 \cdot x_s + \varepsilon_s$$

where:

- $y_s$ = BLS 2024–2034 projected employment growth in sector $s$ (percentage points, signed)
- $x_s$ = 2025 AI adoption in sector $s$ (percent)
- $\varepsilon_s$ = residual
- $n = 15$ sectors

### 5.2 OLS estimators (computed in the browser)

The slope and intercept are computed by the standard OLS formulas:

$$\hat{\beta}_1 = \frac{\sum_{s=1}^{n} (x_s - \bar{x})(y_s - \bar{y})}{\sum_{s=1}^{n} (x_s - \bar{x})^2}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

and the coefficient of determination:

$$R^2 = 1 - \frac{\sum_s (y_s - \hat{y}_s)^2}{\sum_s (y_s - \bar{y})^2}$$

### 5.3 What the regression can and cannot say

**Descriptive, not causal.** The slope $\hat{\beta}_1$ measures the conditional mean relationship in the 15-sector cross-section. It does **not** identify the causal effect of AI adoption on employment — both variables are shaped by confounders (sector capital intensity, skill mix, trade exposure). With $n = 15$ observations, any causal interpretation would be indefensible.

**What it does show.** Whether high-adoption sectors in the cross-section are also projected to gain or lose employment on average. The finding is a mildly positive slope with modest $R^2$, consistent with the complementarity literature (Acemoglu & Restrepo 2018, 2022; Autor 2024) and inconsistent with a pure aggregate-displacement story. Retail appears as a clear outlier in the residual, consistent with the substitution literature (Frey & Osborne 2017).

---

## 6. Summary — how the equations compose

Each stage's output feeds the next:

$$\underbrace{\tau_{c,t}}_{\text{Stage 1: Capacity}} \to \underbrace{I_t,\ A_{s,t}}_{\text{Stage 2: Concentration}} \to \underbrace{\hat{I}(t),\ \alpha_s}_{\text{Stage 3: Forecast}} \to \underbrace{\hat{\beta}_1(x_s, y_s)}_{\text{Stage 4: Labor}}$$

Reading left to right:

1. **Capacity** ($\tau$) identifies the countries that *can* absorb AI investment at scale.
2. **Concentration** ($I_t$, $A_{s,t}$) observes where the resulting global flow is actually going, industry-by-industry.
3. **Forecast** extends the concentration pattern: the logistic projects total flow ($\hat{I}(t)$) and the Bass + output-share allocation distributes it across industries ($\alpha_s$).
4. **Labor** tests whether the industries receiving the most investment and showing the highest adoption are also the industries projected to gain employment.

The composition is deliberately sequential: an argumentative arc, not four parallel panels.

---

## References

- Acemoglu, D. & Restrepo, P. (2018). "The Race between Man and Machine: Implications of Technology for Growth, Factor Shares, and Employment." *American Economic Review* 108(6).
- Acemoglu, D. & Restrepo, P. (2020). "Robots and Jobs: Evidence from US Labor Markets." *Journal of Political Economy* 128(6).
- Acemoglu, D. & Restrepo, P. (2022). "Tasks, Automation, and the Rise in U.S. Wage Inequality." *Econometrica* 90(5).
- Autor, D. (2024). "Applying AI to Rebuild Middle Class Jobs." NBER Working Paper 32140.
- Bass, F. M. (1969). "A New Product Growth Model for Consumer Durables." *Management Science* 15(5): 215–227.
- Brynjolfsson, E., Chandar, B. & Chen, R. (2025). "Canaries in the Coal Mine: Generative AI and Early-Career Employment." Stanford Digital Economy Lab working paper.
- Frey, C. B. & Osborne, M. A. (2017). "The Future of Employment: How Susceptible are Jobs to Computerisation?" *Technological Forecasting and Social Change* 114.
- Mansfield, E. (1961). "Technical Change and the Rate of Imitation." *Econometrica* 29(4).
- Rogers, E. M. (1962). *Diffusion of Innovations.* Free Press.
- Stanford HAI (2025, 2026). *AI Index Report.* Stanford Institute for Human-Centered Artificial Intelligence.
- Sultan, F., Farley, J. U. & Lehmann, D. R. (1990). "A Meta-Analysis of Applications of Diffusion Models." *Journal of Marketing Research* 27(1).
- Verhulst, P.-F. (1838). "Notice sur la loi que la population poursuit dans son accroissement." *Correspondance Mathématique et Physique* 10.
- U.S. Bureau of Labor Statistics (2025, 2026). "Industry and Occupational Employment Projections, 2024–34." *Monthly Labor Review.*
