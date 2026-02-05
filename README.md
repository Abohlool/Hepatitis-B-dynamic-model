# Modeling the dynamics of acute and chronic hepatits B with optimal control

Scientific Reports  
(2023) 13:14980  
doi: 10.1038/s41598-023-39582

## Deterministic Model

$$
\Large
\begin{cases}
\frac{dS}{dt} &= \{1 - \eta B\}\Lambda - (\nu + \mu_0)S - (A + \gamma B)\alpha S, \\
\frac{dA}{dt} &= \alpha SA + \gamma\alpha SB - (\gamma_1 + \beta + \mu_0)A, \\
\frac{dB}{dt} &= \beta A - (\mu_1 + \gamma_2 + \mu_0 - \eta\Lambda)B, \\
\frac{dR}{dt} &= \gamma_2 B - \mu_0 R + \gamma_1 A + \nu S. 
\end{cases}
$$

$\large\Lambda\colon\text{The rate of newborns,}$  
$\large\nu\colon\text{The vaccination parameter,}$  
$\large\eta\colon\text{The maternally infected,}$  
$\large\gamma\colon\text{The reduced transmission rate,}$  
$\large\mu_0\colon\text{The proportion of natural death,}$  
$\large\mu_1\colon\text{The portion of death due to the disease,}$  
$\large\alpha\colon\text{The contact parameter,}$  
$\large\gamma_1\colon\text{The recovery rate from acute class,}$  
$\large\gamma_2\colon\text{The recovery rate from chronic class,}$  
$\large\beta\colon\text{The proportion who move from acute class to chronic.}$  


$\Lambda = 0.0121, \quad \eta = 0.5, \quad \mu_0 = 0.01693, \quad \nu = 0.02, \quad \alpha = 0.95$  
$\gamma = 0.16, \quad \gamma_1 = 0.05, \quad \beta = 0.23, \quad \gamma_2 = 0.002, \quad \mu_1 = 0.8$

$S_0 = 80, \quad A_0 = 10, \quad B_0 = 5, \quad R_0 = 5$

![Deterministic-Model](https://github.com/Abohlool/Hepatitis-B-dynamic-model/blob/main/plots/Deterministic.png)

## Controlled Mode

$$
\Large
\begin{cases}
\frac{dS}{dt} &= (1 - \eta B)\Lambda - (\mu_0 + u_1)S - \alpha S (A + \gamma B), \\
\frac{dA}{dt} &= \alpha S (A + \gamma B) - (u_2 + \mu_0 + \gamma_1 + \beta)A, \\
\frac{dB}{dt} &= \beta A - (\mu_1 + \gamma_2 + \mu_0 + \eta\Lambda + u_2) B, \\
\frac{dR}{dt} &= \gamma_2 B - \mu_0 R + \gamma_1 A + u_1 S + (B + A)u_2. 
\end{cases}
$$

$u_1(t)$: Vaccination effort  
$u_2(t)$: Treatment effort  

$$
\large
0 \le u_1(t), u_2(t) \le 1
$$

$w_1 = 0.1, \quad w_2 = 0.6, \quad w_3 = 0.001, \quad w_4 = 0.9$

![Controlled-Model](https://github.com/Abohlool/Hepatitis-B-dynamic-model/blob/main/plots/Controlled.png)

## Controlled vs Uncontrolled (Deterministic)

![Controlled-vs-Deterministic-model](https://github.com/Abohlool/Hepatitis-B-dynamic-model/blob/main/plots/Deterministic-vs-Controlled.png)

## Control Functions

![Controll-Functions-over-time](https://github.com/Abohlool/Hepatitis-B-dynamic-model/blob/main/plots/Control-functions.png)

## Reproductive Number

$$
\Large
R_0 = \frac{\alpha\Lambda (\gamma(\mu_0 + \gamma_1 + \beta + u_2) + \beta)}{(\mu_0 + u_1)(\mu_0 + \gamma_1 + \beta + u_2)(\mu_0 + \mu_1 + \gamma_2 + u_2 - \eta\Lambda)}
$$

![R_0-over-controls](https://github.com/Abohlool/Hepatitis-B-dynamic-model/blob/main/plots/R_0-control.png)
