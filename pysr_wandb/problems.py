# "\rowcolor{white}            Keijzer-1","1","$0.3*x^1*sin(2*\pi*x_1)$","$E[-1, 1, 0.1]$","$E[-1, 1, 0.001]$","\cite{krawiec2013approximating,krawiec2014behavioral,demelo2014kaizen,miranda2017how,liskowski2017discovery}"
# "\rowcolor{gray!25}          Keijzer-2","1","$0.3*x^1*sin(2*\pi*x_1)$","$E[-2, 2, 0.1]$","$E[-2, 2, 0.001]$","\cite{demelo2014kaizen,miranda2017how}"
# "\rowcolor{white}            Keijzer-3","1","$0.3*x^1*sin(2*\pi*x_1)$","$E[-3, 3, 0.1]$","$E[-3, 3, 0.001]$","\cite{demelo2014kaizen,miranda2017how}"
# "\rowcolor{gray!25}          Keijzer-4","1","$x_1^3*e^{-x_1}*cos(x_1)*sin(x_1)*(sin^2(x_1)*cos(x_1)-1)$","$E[0, 10, 0.1]$","$E[0.05, 10.05, 0.1]$","\cite{krawiec2013approximating,krawiec2014behavioral,demelo2014kaizen,szubert2016reducing,chen2016improving,sotto2017probabilistic,miranda2017how,liskowski2017discovery}"
# "\rowcolor{white}","","","$x_1, x_2: U[-1, 1, 1000]$","$x_1, x_2: U[-1, 1, 10000]$",""
# "\multirow{-2}{*}{Keijzer-5}","\multirow{-2}{*}{3}","\multirow{-2}{*}{$\frac{30*x_1*x_3}{(x_1-10)*x_2^2}$}","$x_3: U[1, 2, 1000]$","$x_3: U[1, 2, 10000]$","\multirow{-2}{*}{\cite{krawiec2014behavioral,demelo2014kaizen,sotto2017probabilistic,oliveira2016dispersion}} "
# "Keijzer-6","1","$\sum_{i=1}^{x_1}i$","$E[1, 50, 1]$","$E[1, 120, 1]$","\cite{demelo2014kaizen,lacava2015genetic,nicolau2016managing,oliveira2016dispersion,miranda2017how,medvet2017evolvability}"
# "Keijzer-7","1","$ln x_1$","$E[1, 100, 1]$","$E[1, 100, 0.1]$","\cite{demelo2014kaizen,oliveira2016dispersion,miranda2017how}"
# "Keijzer-8","1","$\sqrt{x_1}$","$E[0, 100, 1]$","$E[0, 100, 0.1]$","\cite{demelo2014kaizen,miranda2017how,liskowski2017discovery}"
# "Keijzer-9","1","$arcsinh(x_1) = ln(x_1+\sqrt{x_1^2+1})$","$E[0, 100, 1]$","$E[0, 100, 0.1]$","\cite{demelo2014kaizen,miranda2017how}"
# "Keijzer-10","2","$x_1^{x_2}$","$U[0, 1, 100]$","$E[0, 1, 0.01]$","\cite{wieloch2013running,demelo2014kaizen,thuong2017combining}"
# "Keijzer-11","2","$x_1*x_2+sin((x_1-1)*(x_2-1))$","$U[-3, 3, 20]$","$E[-3, 3, 0.01]$","\cite{krawiec2014behavioral,demelo2014kaizen,chen2016improving,thuong2017combining} "
# "Keijzer-12 (*)","2","$x_1^4-x_1^3+(\frac{x_2^2}{2})-x_2$","$U[-3, 3, 20]$","$E[-3, 3, 0.01]$","\cite{wieloch2013running,krawiec2014behavioral,demelo2014kaizen,chen2016improving,thuong2017combining}"
# "Keijzer-13","2","$6*sin(x_1)*cos(x_2)$","$U[-3, 3, 20]$","$E[-3, 3, 0.01]$","\cite{krawiec2014behavioral,demelo2014kaizen,thuong2017combining}"
# "Keijzer-14","2","$\frac{8}{2+x_1^2+x_2^2}$","$U[-3, 3, 20]$","$E[-3, 3, 0.01]$","\cite{krawiec2014behavioral,demelo2014kaizen,chen2016improving,liskowski2017discovery,thuong2017combining}"
# "Keijzer-15","2","$\frac{x_1^3}{5}+\frac{x_2^3}{2}-x_2-x_1$","$U[-3, 3, 20]$","$E[-3, 3, 0.01]$","\cite{krawiec2014behavioral,demelo2014kaizen,chen2016improving,liskowski2017discovery,thuong2017combining}"
import numpy as np


def keijzer11(seed):
    """Keijzer-11."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(20, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = x1 * x2 + np.sin((x1 - 1) * (x2 - 1))

    return X, y


def keijzer12(seed):
    """Keijzer-12."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(20, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = x1**4 - x1**3 + (x2**2 / 2) - x2

    return X, y


def keijzer13(seed):
    """Keijzer-13."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(20, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = 6 * np.sin(x1) * np.cos(x2)

    return X, y


def keijzer4(seed):
    """Keijzer-4."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0, 10, size=(100, 1))
    x1 = X[:, 0]
    y = (
        x1**3
        * np.exp(-x1)
        * np.cos(x1)
        * np.sin(x1)
        * (np.sin(x1) ** 2 * np.cos(x1) - 1)
    )

    return X, y


def keijzer14(seed):
    """Keijzer-14."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(20, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = 8 / (2 + x1**2 + x2**2)

    return X, y


# "Vladislavleva-1","2","$\frac{e^{-(x_1-1)^2}}{1.2+(x_2-2.5)^2}$","$U[0.3, 4, 100]$","$E[-0.2, 4.2, 0.1]$","\cite{oliveira2016dispersion,miranda2017how,liskowski2017discovery,thuong2017combining}"
# "Vladislavleva-2","1","$e^{-x_1}*x_1^3*(cos x_1*sin x_1)*(cos x_1*sin^2 x_1-1)$","$E[0.05, 10, 0.1]$","$E[-0.5, 10.5, 0.05]$","\cite{miranda2017how}"
# "","","","$x_1: E[0.05, 10, 0.1]$","$x_1: E[-0.5, 10.5, 0.05]$",""
# "\rowcolor{white} \multirow{-2}{*}{Vladislavleva-3}","\multirow{-2}{*}{2}","\multirow{-2}{*}{$e^{-x_1}*x_1^3*(cos x_1*sin x_1)*(cos x_1*sin^2 x_1-1)*(x_2-5)$}","$x_2: E[0.05, 10.05, 2]$","$x_2: E[-0.5, 10.5, 0.5]$","\multirow{-2}{*}{\cite{miranda2017how}}"
# "\rowcolor{gray!25}            Vladislavleva-4","5","$\frac{10}{5+\sum_{i=1}^5 (x_i-3)^2}$","$U[0.05, 6.05,1024]$","$U[-0.25, 6.35, 5000]$","\cite{lacava2015genetic,medernach2016new,nicolau2016managing,lacava2016epsilon,oliveira2016dispersion}"
# "\rowcolor{white}","","","$x_1: U[0.05, 2,300]$","$x_1: E[-0.05, 2.1, 0.15]$",""
# "","","","$x_2: U[1, 2,300]$","$x_2: E[0.95, 2.05, 0.1]$ "
# "\rowcolor{white} \multirow{-3}{*}{Vladislavleva-5}","\multirow{-3}{*}{3}","\multirow{-3}{*}{$30*(x_1-1)*\frac{x_3-1}{(x_1-10)*x_2^2}$}","$x_3: U[0.05, 2,300]$","$x_3: E[-0.05, 2.1, 0.15]$","\multirow{-3}{*}{\cite{chen2016improving,thuong2017combining}} "
# "\rowcolor{gray!25}          Vladislavleva-6","2","$6*sin(x_1)*cos(x_2)$","$U[0.1, 5.9, 30]$","$E[-0.05, 6.05, 0.02]$","\cite{chen2016improving,thuong2017combining}"
# "\rowcolor{white}            Vladislavleva-7","2","$(x_1-3)*(x_2-3)+2*sin((x_1-4)*(x_2-4))$","$U[0.05, 6.05,300]$","$U[-0.25, 6.35,1000]$","\cite{miranda2017how}"
# "\rowcolor{gray!25}          Vladislavleva-8","2","$\frac{(x_1-3)^4+(x_2-3)^3-(x_2-3)}{(x_2-2)^4+10}$","$U[0.05, 6.05,50]$","$E[-0.25, 6.35, 0.2]$","\cite{chen2016improving,thuong2017combining}"


def vlad1(seed):
    """Vladislavleva-1."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.3, 4, size=(100, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = np.exp(-((x1 - 1) ** 2)) / (1.2 + (x2 - 2.5) ** 2)

    return X, y


def vlad2(seed):
    """Vladislavleva-2."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.05, 10, size=(100, 1))
    x1 = X[:, 0]
    y = (
        np.exp(-x1)
        * x1**3
        * (np.cos(x1) * np.sin(x1))
        * (np.cos(x1) * np.sin(x1) ** 2 - 1)
    )

    return X, y


def vlad3(seed):
    """Vladislavleva-3."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.05, 10, size=(100, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = (
        np.exp(-x1)
        * x1**3
        * (np.cos(x1) * np.sin(x1))
        * (np.cos(x1) * np.sin(x1) ** 2 - 1)
        * (x2 - 5)
    )

    return X, y


def vlad4(seed):
    """Vladislavleva-4."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.05, 6.05, size=(1024, 5))
    y = 10 / (5 + np.sum((X - 3) ** 2, axis=1))

    return X, y


def vlad5(seed):
    """Vladislavleva-5."""
    rstate = np.random.RandomState(seed)
    x1 = rstate.uniform(0.05, 2, size=300)
    x2 = rstate.uniform(1, 2, size=300)
    x3 = rstate.uniform(0.05, 2, size=300)
    X = np.vstack((x1, x2, x3)).T
    y = 30 * (x1 - 1) * (x3 - 1) / ((x1 - 10) * x2**2)

    return X, y


# "\textbf{Dataset}","\textbf{Variables}","\textbf{Objective Function}","\textbf{Training Set}","\textbf{Testing Set}","\textbf{Source}"
# "\cmidrule(r{0em}){1-6}
# "\rowcolor{gray!25}            Burks (*)","1","$4*x_1^4+3*x_1^3+2*x_1^2+x_1$","$U[-1,1,20]$","None","\cite{burks2015efficient,szubert2016reducing}"


# "Korns-1","1","$1.57+24.3*x_4$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-2\textsuperscript{1}","3","$0.23+14.2*\frac{x_4+x_2}{3*x_5}$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-3\textsuperscript{1}","4","$-5.41+4.9*\frac{x_4-x_1+\frac{x_2}{x_5}}{3*x_5}$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized,sotto2017probabilistic}"
# "Korns-4","1","$-2.3+0.13*sin(x_3)$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-5\textsuperscript{1}","1","$3+2.13*ln(x_5)$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized,sotto2017probabilistic}"
# "Korns-6\textsuperscript{1}","1","$1.3+0.13*\sqrt{x_1}$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-7","1","$213.80940889*(1-e^{-0.54723748542*x_1})$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-8\textsuperscript{1}","3","$6.87+11*\sqrt{7.23*x_1*x_4*x_5}$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-9\textsuperscript{1}","4","$\frac{\sqrt{x_1}}{ln(x_2)}*\frac{e^{x_3}}{x_4^2}$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-10\textsuperscript{1}","4","$0.81+24.3*\frac{2*x_2+3*x_3^2}{4*x_4^3+5*x_5^4}$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-11","1","$6.87+11*cos(7.23*x_1^3)$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Korns-12","2","$2-2.1*cos(9.8*x_1)*sin(1.3*x_5)$","$U[-50,50,10000]$","$U[-50,50,10000]$","\cite{worm2013prioritized}"
# "Koza-2 (*)","1","$x_1^5-2*x_1^3+x_1$","$U(-1, 1, 20)$","None","\cite{meier2013accelerating}"
# "Koza-3 (*)","1","$x_1^6-2*x_1^4+x_1^2$","$U(-1, 1, 20)$","None","\cite{meier2013accelerating, harada2014asynchronously}"
# "Meier-3","2","$\frac{x_1^2*x_2^2}{x_1+x_2}$","$U[-1,1,50]$","None","\cite{meier2013accelerating}"
# "Meier-4","2","$\frac{x_1^5}{x_2^3}$","$U[-1,1,50]$","None","\cite{meier2013accelerating}"
# "Nguyen-1 (*)","1","$x_1^3+x_1^2+x_1$","$U(-1, 1, 20)$","None","\cite{worm2013prioritized,demelo2014kaizen,sotto2017probabilistic}"
# "Nguyen-2 (*)","1","$x_1^4+x_1^3+x_1^2+x_1$","$U(-1, 1, 20)$","None","\cite{worm2013prioritized,lopes2013gearnet,harada2014asynchronously,demelo2014kaizen,whigham2015examining,sotto2017probabilistic,medvet2017evolvability}"
# "Nguyen-3","1","$x_1^5+x_1^4+x_1^3+x_1^2+x_1$","$U(-1, 1, 20)$","None","\cite{worm2013prioritized,wieloch2013running,krawiec2014behavioral,demelo2014kaizen,sotto2017probabilistic,liskowski2017discovery}"
# "Nguyen-4","1","$x_1^6+x_1^5+x_1^4+x_1^3+x_1^2+x_1$","$U(-1, 1, 20)$","None","\cite{worm2013prioritized,wieloch2013running,krawiec2014behavioral,demelo2014kaizen,sotto2017probabilistic,liskowski2017discovery}"
# "Nguyen-5","1","$sin(x_1^2)*cos(x_1)-1$","$U(-1, 1, 20)$","None","\cite{worm2013prioritized,wieloch2013running,harada2014asynchronously,krawiec2014behavioral,demelo2014kaizen,liskowski2017discovery}"
# "Nguyen-6","1","$sin(x_1)+sin(x_1+x_1^2)$","$U(-1, 1, 20)$","None","\cite{worm2013prioritized,wieloch2013running,krawiec2014behavioral,demelo2014kaizen,sotto2017probabilistic,liskowski2017discovery}"
# "Nguyen-7","1","$ln(x_1+1)+ln(x_1^2+1)$","$U[0,2,20]$","None","\cite{krawiec2013approximating},\cite{worm2013prioritized,wieloch2013running,harada2014asynchronously,krawiec2014behavioral,demelo2014kaizen,lacava2015genetic,liskowski2017discovery}"
# "Nguyen-8","1","$\sqrt{x_1}$","$U[0,4,20]$","None","\cite{worm2013prioritized,wieloch2013running,demelo2014kaizen,liskowski2017discovery}"
# "Nguyen-9","2","$sin(x_1)+sin(x_2^2)$","$U(-1, 1, 100)$","None","\cite{worm2013prioritized,wieloch2013running,krawiec2014behavioral,demelo2014kaizen,liskowski2017discovery}"
# "Nguyen-10","2","$2*sin(x_1)*cos(x_2)$","$U(-1, 1, 100)$","None","\cite{worm2013prioritized,wieloch2013running,krawiec2014behavioral,demelo2014kaizen}"
# "Nonic","1","$\sum_{i=1}^9 x_1^i$","$E[-1,1,20]$","$U[-1,1,20]$","\cite{krawiec2013approximating,szubert2016reducing}"
# "Pagie-1","2","$\frac{1}{1+x_1^{-4}}+\frac{1}{1+x_2^{-4}}$","$E[-5, 5, 0.4]$","None","\cite{mcphee2015impact,lacava2015genetic,liskowski2017discovery}"
# "Poly-10","9","$x_1*x_2+x_3*x_4+x_5*x_6+x_1*x_7*x_9+x_3*x_6*x_{10}$","$U[0,1,330]$","$U[0,1,170] $","\cite{medernach2016new}"
# "R1","1","$\frac{(x_1+1)^3}{x_1^2-x_1+1}$","$E[-1,1,20]$","$U[-1,1,20]$","\cite{krawiec2013approximating,szubert2016reducing,liskowski2017discovery}"
# "R2","1","$\frac{x_1^5-3*x_1^3+1}{x_1^2+1}$","$E[-1,1,20]$","$U[-1,1,20]$","\cite{krawiec2013approximating,szubert2016reducing,liskowski2017discovery}"
# "R3","1","$\frac{x_1^6+x_1^5}{x_1^4+x_1^3+x_1^2+x_1+1}$","$E[-1,1,20]$","$U[-1,1,20]$","\cite{liskowski2017discovery}"
# "Sine","1","$sin(x_1)+sin(x_1+x_1^2)$","$E[0,6.2,0.1]$","None","\cite{mcphee2015impact}"
