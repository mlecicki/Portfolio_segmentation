**Portfolio clustering**
================
**author**: Maciej Lecicki
19.10.2021

<br/> <br/> <br/>

### Purpose and objectives

<br/>

Portfolio clustering is a practical example of using R in Supply Chain
Management.

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

Data.

``` r
data <- read_rds("data/data.rds")
```
