**Portfolio clustering**
================
**author**: Maciej Lecicki
19.10.2021

<br/> <br/> <br/>

### Purpose and objectives

<br/>

High customization of products results in portfolio growth. The number
of SKUs has been increasing and this trend is unlikely to change. This
is one of the challenges for Supply Chain Management (SCM).

SC professionals diversify portfolio management efforts in line with ABC
classification based 80/20 rule. Although it helps manage workload, this
classification method has been used since 1950s to bring structure in
times where there were no ERP or APS systems. In today’s world this
approach seems to be outdated and even adding XYZ elements to ABC
classification (which bring demand variability factor resulting in
ABC/XYZ matrix with 9 clusters of products from A/X - most
valuable/least volatile to C/Z - least valuable/most volatile) does not
change this fact.

The reasons for this are:<br/> - ABC segmentation is based on historic
data and from that point of view it does not add any additional
information about the product itself (volume, cost and even volatility
are known),<br/> - it does not capture physical aspect of goods,<br/> -
it does not capture any manufacturing or upstream supply chain
constraints,<br/> - it does not clearly answer a question about optimal
number of clusters or thresholds used for the split.

With this in mind, let’s investigate alternative segmentation
methodology using Unsupervised Machine Learning56, evaluate outcome,
compare with classic ABC (or at least approximation of ABC
classification as we do not have this information in our dataset) and
brainstorm pros and cons behind this methodology or its deployment in
SCM.

As Unsupervised Machine learning is considered part of Exploratory Data
Analysis even if we reject this idea it’s still good opportunity to
learn something new about our data.

Portfolio clustering is a practical example of using R in Supply Chain
Management.

<br/> <br/>

### Libraries and data examination

<br/>

List of libraries.

``` r
library(clustertend)
library(fpc)
library(psych)
library(skimr)
library(factoextra)
#library(dendextend)
library(patchwork)
library(wesanderson)
library(caret)
library(tidyverse)
library(patchwork)
```

Let’s first read in data.

``` r
data <- read_rds("data/data.rds")
```
