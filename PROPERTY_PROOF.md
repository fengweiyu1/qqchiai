# Pass and fail score ordering for the five outcome measures

This supplement analyzes Property 1 for Confidence (Conf), Probability Deficit (Def), Cross-Entropy (CE), Log-Transformed Margin (Mar), and Cosine Deviation (Cos). It derives the score of a passing test and a failing test separately, then compares their scores. Mar guarantees the required ordering; the other four measures do not guarantee it in the multiclass tasks used in the experiments.

## Notation and Property 1

For a test with true label $y$, let $\mathbf p$ be its predicted probability vector and define

$$
c=p_y,\qquad u=p_{\max}=\max_k p_k.
$$

A passing test predicts the true class, so $u=c$. A failing test with a unique maximum predicts another class, so $u>c$. If the maximum is tied, a failing test can also have $u=c$; the selected class determines whether it passes. The arguments below allow such ties, and the constructions establishing violations use unique maxima.

Consider two independent tests: a passing test $t_p$ and a failing test $t_f$. Subscripts $p$ and $f$ identify their respective quantities. In particular, $u_p=c_p$ and $u_f\ge c_f$. These within-test relations do not themselves order $c_p$ and $c_f$ across the two tests.

**Property 1.** For every passing test $t_p$ and failing test $t_f$,

$$
r(t_p)\le r(t_f).
$$

The class counts are fixed by the experimental datasets: 10 for Fashion-MNIST and CIFAR-10, and 43 for GTSRB. Where needed, $C$ denotes this existing class count, rather than a tunable parameter. The probability entries sum to one. The logarithm is natural, and $\varepsilon>0$ is the same fixed stabilizing constant for both tests.

## Confidence

Define $h(x)=\operatorname{clip}(x,0.01,0.99)$. Confidence uses the predicted class probability, with different formulas for passing and failing tests:

$$
r_{\mathrm{Conf}}(t_p)=1-h(u_p)=1-h(c_p),\qquad
r_{\mathrm{Conf}}(t_f)=h(u_f).
$$

For a passing test, the score measures uncertainty in the correct prediction. For a failing test, it measures confidence in the incorrect prediction. Their difference is

$$
r_{\mathrm{Conf}}(t_f)-r_{\mathrm{Conf}}(t_p)
=h(u_f)+h(c_p)-1.
$$

Consequently, the required ordering holds for a given pair exactly when $h(u_f)+h(c_p)\ge1$. Neither passing nor failing requires the predicted class probability to exceed $1/2$ in these multiclass tasks. The classification conditions therefore do not enforce this inequality. The attainable probability distributions derived below establish that the difference can be negative or positive. Conf does not guarantee Property 1.

## Probability Deficit

Def uses the probability assigned to the true class for both outcomes:

$$
r_{\mathrm{Def}}(t_p)=1-c_p,\qquad
r_{\mathrm{Def}}(t_f)=1-c_f.
$$

In either case, less probability assigned to the true class gives a larger score. Comparing the two tests gives

$$
r_{\mathrm{Def}}(t_f)-r_{\mathrm{Def}}(t_p)=c_p-c_f.
$$

Thus $r_{\mathrm{Def}}(t_p)\le r_{\mathrm{Def}}(t_f)$ if and only if $c_p\ge c_f$. Passing means that $c_p$ is maximal within its own probability vector; failing means that some competing class has probability at least $c_f$ within a different vector. Neither condition guarantees $c_p\ge c_f$. The difference can have either sign, as established below. Def does not guarantee Property 1.

## Cross Entropy

The two scores are

$$
r_{\mathrm{CE}}(t_p)=-\log(c_p+\varepsilon),\qquad
r_{\mathrm{CE}}(t_f)=-\log(c_f+\varepsilon).
$$

Their difference is

$$
r_{\mathrm{CE}}(t_f)-r_{\mathrm{CE}}(t_p)
=\log\frac{c_p+\varepsilon}{c_f+\varepsilon}.
$$

Since the logarithm is strictly increasing, the required ordering holds exactly when $c_p\ge c_f$, the same condition as for Def. The logarithm changes the magnitude of score differences but preserves this ordering. The classification conditions do not impose the required cross-test probability relation, and the difference can have either sign. CE does not guarantee Property 1.

## Log Transformed Margin

Mar additionally uses the largest probability among classes other than the true class:

$$
m=\max_{k\ne y}p_k.
$$

This quantity differs from $u=p_{\max}$ on a passing test: $m_p\le c_p=u_p$. On a failing test, $m_f=u_f\ge c_f$. Define $d=m-c$. The score can then be written as

$$
r_{\mathrm{Mar}}=-\log\left(\frac{1-d}{2}+\varepsilon\right).
$$

Separately for the two outcomes,

$$
r_{\mathrm{Mar}}(t_p)
=-\log\left(\frac{1+c_p-m_p}{2}+\varepsilon\right),
$$

$$
r_{\mathrm{Mar}}(t_f)
=-\log\left(\frac{1+c_f-m_f}{2}+\varepsilon\right).
$$

The classification conditions imply

$$
\frac{1+c_p-m_p}{2}+\varepsilon\ge\frac12+\varepsilon,
\qquad
\frac{1+c_f-m_f}{2}+\varepsilon\le\frac12+\varepsilon.
$$

Because $-\log(x)$ is strictly decreasing for $x>0$,

$$
r_{\mathrm{Mar}}(t_p)
\le-\log\left(\frac12+\varepsilon\right)
\le r_{\mathrm{Mar}}(t_f).
$$

Hence Mar satisfies Property 1 for every pair, including ties. When the failing test has a unique incorrect maximum, the second inequality is strict. Replacing $m$ with $u$ would set the margin of every passing test to zero and would change the measure defined in the paper.

## Cosine Deviation

Let $\mathbf e_y$ be the one-hot vector for the true class, and write

$$
s_p=\|\mathbf p_{t_p}\|_2,\qquad s_f=\|\mathbf p_{t_f}\|_2.
$$

Since $\|\mathbf e_y\|_2=1$, the two scores are

$$
r_{\mathrm{Cos}}(t_p)=1-\frac{c_p}{s_p},\qquad
r_{\mathrm{Cos}}(t_f)=1-\frac{c_f}{s_f}.
$$

Both measure deviation of the full probability vector from the ideal true-class direction. Their difference is

$$
r_{\mathrm{Cos}}(t_f)-r_{\mathrm{Cos}}(t_p)
=\frac{c_p}{s_p}-\frac{c_f}{s_f}.
$$

The required ordering is therefore equivalent to $c_p/s_p\ge c_f/s_f$. Passing and failing constrain which coordinate is maximal, whereas these ratios also depend on the squared probabilities of all coordinates. The classification conditions do not enforce their ordering. The following derivation establishes attainable distributions with either sign, so Cos does not guarantee Property 1.

## Attainable orderings for the four measures

To establish the absence of a universal ordering mathematically, it is necessary to show that both signs are attainable, rather than merely observe that the formulas contain different variables. We use probability families with the fixed class count $C\in\{10,43\}$.

For a passing test, set the true-class probability to $c_p=1/C+\delta$ and each other probability to $(1-c_p)/(C-1)$. For sufficiently small $\delta>0$, the true class is the unique maximum. As $\delta\to0^+$,

$$
c_p\to\frac1C,\qquad \frac{c_p}{s_p}\to\frac1{\sqrt C}.
$$

For a failing test, set $c_f=1/2-\delta$, give one incorrect class probability $u_f=1/2$, and distribute the remaining probability $\delta$ equally among the other $C-2$ classes. For sufficiently small $\delta>0$, the incorrect class is the unique maximum. As $\delta\to0^+$,

$$
c_f\to\frac12,\qquad \frac{c_f}{s_f}\to\frac1{\sqrt2}.
$$

The resulting limits of the fail score minus the pass score are

$$
\begin{aligned}
\Delta_{\mathrm{Conf}}&\longrightarrow\frac1C-\frac12<0,\\
\Delta_{\mathrm{Def}}&\longrightarrow\frac1C-\frac12<0,\\
\Delta_{\mathrm{CE}}&\longrightarrow\log\frac{1/C+\varepsilon}{1/2+\varepsilon}<0,\\
\Delta_{\mathrm{Cos}}&\longrightarrow\frac1{\sqrt C}-\frac1{\sqrt2}<0.
\end{aligned}
$$

For Conf, clipping is inactive near $1/C$ and $1/2$ for both experimental class counts. All four limits are strictly negative. By continuity, sufficiently small positive $\delta$ gives strictly positive probability vectors, unique predicted classes, and $r(t_p)>r(t_f)$ for each measure. These are violations of Property 1 within the existing class counts.

The reverse ordering is also attainable. Let a passing distribution approach probability one on the true class, and a failing distribution approach probability one on an incorrect class. The differences approach $0.98$ for Conf, $1$ for Def, $\log((1+\varepsilon)/\varepsilon)>0$ for CE, and $1$ for Cos. Continuity again gives strictly positive distributions with $r(t_p)<r(t_f)$.

Therefore, Conf, Def, CE, and Cos admit both score orderings in the experimental tasks. Mar alone among these five measures guarantees that every passing test has a score no greater than every failing test. This guarantee concerns separation of passing and failing tests; it does not define a universal ordering of severity among all failing tests.
