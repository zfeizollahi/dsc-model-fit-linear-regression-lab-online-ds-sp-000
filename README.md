
# Model Fit in Linear Regression - Lab

## Introduction
In this lab, you'll learn how to evaluate your model results, and you'll learn methods to select the appropriate features using stepwise selection.

## Objectives
You will be able to:
* Analyze the results of regression and R-squared and adjusted-R-squared 
* Understand and apply forward and backward predictor selection

## The Boston Housing Data once more

We pre-processed the Boston Housing Data the same way we did before:

- We dropped "ZN" and "NOX" completely
- We categorized "RAD" in 3 bins and "TAX" in 4 bins
- We used min-max-scaling on "B", "CRIM" and "DIS" (and logtransformed all of them first, except "B")
- We used standardization on "AGE", "INDUS", "LSTAT" and "PTRATIO" (and logtransformed all of them first, except for "AGE") 


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_features = boston_features.drop(["NOX","ZN"],axis=1)

# first, create bins for based on the values observed. 3 values will result in 2 bins
bins = [0,6,24]
bins_rad = pd.cut(boston_features['RAD'], bins)
bins_rad = bins_rad.cat.as_unordered()
#bins_rad = bins_rad.astype('category').cat.codes.astype('category')

# first, create bins for based on the values observed. 4 values will result in 3 bins
bins = [0, 270, 360, 712]
bins_tax = pd.cut(boston_features['TAX'], bins)
bins_tax = bins_tax.cat.as_unordered()

tax_dummy = pd.get_dummies(bins_tax, prefix="TAX")
rad_dummy = pd.get_dummies(bins_rad, prefix="RAD")
boston_features = boston_features.drop(["RAD","TAX"], axis=1)
boston_features = pd.concat([boston_features, rad_dummy, tax_dummy], axis=1)

age = boston_features["AGE"]
b = boston_features["B"]
logcrim = np.log(boston_features["CRIM"])
logdis = np.log(boston_features["DIS"])
logindus = np.log(boston_features["INDUS"])
loglstat = np.log(boston_features["LSTAT"])
logptratio = np.log(boston_features["PTRATIO"])

# minmax scaling
boston_features["B"] = (b-min(b))/(max(b)-min(b))
boston_features["CRIM"] = (logcrim-min(logcrim))/(max(logcrim)-min(logcrim))
boston_features["DIS"] = (logdis-min(logdis))/(max(logdis)-min(logdis))

#standardization
boston_features["AGE"] = (age-np.mean(age))/np.sqrt(np.var(age))
boston_features["INDUS"] = (logindus-np.mean(logindus))/np.sqrt(np.var(logindus))
boston_features["LSTAT"] = (loglstat-np.mean(loglstat))/np.sqrt(np.var(loglstat))
boston_features["PTRATIO"] = (logptratio-np.mean(logptratio))/(np.sqrt(np.var(logptratio)))
```

## Perform stepwise selection

The code for stepwise selection is copied below.


```python
import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
```


```python
boston_features['MEDV'] = boston.target
```


```python
stepwise_selection(boston_features.drop(['MEDV'], axis=1), boston_features['MEDV'])
```

    Add  LSTAT                          with p-value 9.27989e-122
    Add  RM                             with p-value 1.98621e-16
    Add  PTRATIO                        with p-value 2.5977e-12
    Add  DIS                            with p-value 2.85496e-09
    Add  B                              with p-value 2.77572e-06
    Add  TAX_(0, 270]                   with p-value 0.000855799
    Add  CHAS                           with p-value 0.00151282
    Add  INDUS                          with p-value 0.00588575





    ['LSTAT', 'RM', 'PTRATIO', 'DIS', 'B', 'TAX_(0, 270]', 'CHAS', 'INDUS']



### Build the final model again in Statsmodels


```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
```


```python
y = pd.DataFrame(boston.target, columns= ["price"])
X_fin = boston_features[["LSTAT", "RM", "PTRATIO", "DIS", "B", "TAX_(0, 270]", "CHAS", "INDUS"]]
X_with_intercept = sm.add_constant(X_fin)
model = sm.OLS(y,X_with_intercept).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.776</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.773</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   215.7</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 21 Jul 2019</td> <th>  Prob (F-statistic):</th> <td>2.69e-156</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:19:59</td>     <th>  Log-Likelihood:    </th> <td> -1461.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   2941.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   497</td>      <th>  BIC:               </th> <td>   2979.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>    4.8980</td> <td>    2.813</td> <td>    1.742</td> <td> 0.082</td> <td>   -0.628</td> <td>   10.424</td>
</tr>
<tr>
  <th>LSTAT</th>        <td>   -5.5932</td> <td>    0.319</td> <td>  -17.538</td> <td> 0.000</td> <td>   -6.220</td> <td>   -4.967</td>
</tr>
<tr>
  <th>RM</th>           <td>    2.8294</td> <td>    0.386</td> <td>    7.333</td> <td> 0.000</td> <td>    2.071</td> <td>    3.587</td>
</tr>
<tr>
  <th>PTRATIO</th>      <td>   -1.3265</td> <td>    0.226</td> <td>   -5.878</td> <td> 0.000</td> <td>   -1.770</td> <td>   -0.883</td>
</tr>
<tr>
  <th>DIS</th>          <td>   -9.1984</td> <td>    1.333</td> <td>   -6.898</td> <td> 0.000</td> <td>  -11.818</td> <td>   -6.579</td>
</tr>
<tr>
  <th>B</th>            <td>    3.9052</td> <td>    0.931</td> <td>    4.195</td> <td> 0.000</td> <td>    2.076</td> <td>    5.734</td>
</tr>
<tr>
  <th>TAX_(0, 270]</th> <td>    1.4418</td> <td>    0.552</td> <td>    2.614</td> <td> 0.009</td> <td>    0.358</td> <td>    2.526</td>
</tr>
<tr>
  <th>CHAS</th>         <td>    2.7988</td> <td>    0.791</td> <td>    3.539</td> <td> 0.000</td> <td>    1.245</td> <td>    4.353</td>
</tr>
<tr>
  <th>INDUS</th>        <td>   -0.9574</td> <td>    0.346</td> <td>   -2.766</td> <td> 0.006</td> <td>   -1.637</td> <td>   -0.277</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>114.307</td> <th>  Durbin-Watson:     </th> <td>   1.088</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 482.579</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.945</td>  <th>  Prob(JB):          </th> <td>1.62e-105</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.395</td>  <th>  Cond. No.          </th> <td>    96.8</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Where our stepwise procedure mentions that "CHAS" was added with a p-value of 0.00151282, but our statsmodels output returns a p-value of 0.000. What is the intuition behind this?

## Use Feature ranking with recursive feature elimination

Use feature ranking to select the 5 most important features


```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
```


```python
linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select=5)
selector = selector.fit(boston_features.drop(['MEDV'], axis=1) , boston_features['MEDV'])
```


```python
boston_features.columns
```




    Index(['CRIM', 'INDUS', 'CHAS', 'RM', 'AGE', 'DIS', 'PTRATIO', 'B', 'LSTAT',
           'RAD_(0, 6]', 'RAD_(6, 24]', 'TAX_(0, 270]', 'TAX_(270, 360]',
           'TAX_(360, 712]', 'MEDV'],
          dtype='object')




```python
selector.support_
```




    array([False, False,  True,  True, False,  True, False,  True,  True,
           False, False, False, False, False])



Fit the linear regression model again using the 5 columns selected


```python
linreg1 = LinearRegression().fit(boston_features[['CHAS', 'RM', 'DIS', 'B', 
                                                 'LSTAT']] , boston_features['MEDV'])

```

Now, predict $\hat y$ using your model. you can use `.predict()` in scikit-learn


```python
y_hat = linreg1.predict(boston_features[['CHAS', 'RM', 'DIS', 'B', 
                                                 'LSTAT']])
```

Now, using the formulas of R-squared and adjusted-R-squared below, and your Python/numpy knowledge, compute them and contrast them with the R-squared and adjusted-R-squared in your statsmodels output using stepwise selection. Which of the two models would you prefer?

$SS_{residual} = \sum (y - \hat{y})^2 $

$SS_{total} = \sum (y - \bar{y})^2 $

$R^2 = 1- \dfrac{SS_{residual}}{SS_{total}}$

$R^2_{adj}= 1-(1-R^2)\dfrac{n-1}{n-p-1}$


```python
ss_residual = sum( (boston_features['MEDV'] - y_hat) ** 2)
ss_residual
```




    10978.90917178611




```python
ss_total = sum( (boston_features['MEDV'] - boston_features['MEDV'].mean()) ** 2)
ss_total
```




    42716.2954150198




```python
rsquared = 1 - ( ss_residual / ss_total)
rsquared
```




    0.7429807743129866




```python
n = len(boston_features)
adj_rsquare = 1 - (1 - rsquared) * ((n - 1) / (n - 5 - 1) )
adj_rsquare
```




    0.7404105820561164



## Level up - Optional

- Perform variable selection using forward selection, using this resource: https://planspace.org/20150423-forward_selection_with_statsmodels/. Note that this time features are added based on the adjusted-R-squared!
- Tweak the code in the `stepwise_selection()`-function written above to just perform forward selection based on the p-value.

## Summary
Great! You now performed your own feature selection methods!
