# Modeling the dynamics of acute and chronic hepatits B with optimal control

Scientific Reports  
(2023) 13:14980  
doi: 10.1038/s41598-023-39582

$$
\Large
\left\{
\begin{equation*}
\begin{aligned}
\frac{dS}{dt} &= \{1 - \eta B(t)\}\Lambda - \{\nu + \mu_0\}S(t) - \{A(t) + \gamma B(t)\}\alpha S(t), \\
\frac{dA}{dt} &= \alpha S(t)A(t) + \gamma\alpha S(t)B(t) - \{\gamma_1 + \beta + \mu_0\}A(t), \\
\frac{dB}{dt} &= \beta A(t) - \{\mu_1 + \gamma_2 + \mu_0\}B(t) + \eta\Lambda B(t), \\
\frac{dR}{dt} &= \gamma_2 B(t) - \mu_0 R(t) + \gamma_1 A(t) + \nu S(t). 
\end{aligned}
\end{equation*}
\right.
$$
