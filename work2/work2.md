# 数据挖掘大作业二：关联规则挖掘


## 1. 数据预处理
对数据集进行处理，转换成适合关联规则挖掘的形式
数据集：dataset1：wine-reviews，来源https://www.kaggle.com/zynicide/wine-reviews
包含两个csv文件：
   winemag-data-130k-v2.csv：包含14个属性（3个数值属性，11个标称属性），129970条数据记录；
   winemag-data_first150k.csv：包含11个属性（3个数值属性，8个标称属性），150930条数据记录
该数据集是包括了许多葡萄酒的点评，本次作业打算分析红酒产地、品种的关系。
首先对数据集缺失值进行处理，采用的是用最高频率值来填补缺失值的方法
对数值属性使用'属性=百分比'的编码方式进行离散化，使用四分位数将原属性切分为四部分，使用0-0.25,0.25-0.5,0.5-0.75,0.75-1.0共4个离散化属性来替代原属性


```python
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import pylab
%matplotlib inline


df = pd.read_csv('assignment1/wine-reviews/winemag-data-130k-v2.csv', index_col=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
  </tbody>
</table>
</div>




```python
transactions = []
for index, row in df.iterrows():
    transactions += [(row['country'], row['variety'], row['winery'])]

transactions[:20]
```




    [('Italy', 'White Blend', 'Nicosia'),
     ('Portugal', 'Portuguese Red', 'Quinta dos Avidagos'),
     ('US', 'Pinot Gris', 'Rainstorm'),
     ('US', 'Riesling', 'St. Julian'),
     ('US', 'Pinot Noir', 'Sweet Cheeks'),
     ('Spain', 'Tempranillo-Merlot', 'Tandem'),
     ('Italy', 'Frappato', 'Terre di Giurfo'),
     ('France', 'Gewürztraminer', 'Trimbach'),
     ('Germany', 'Gewürztraminer', 'Heinz Eifel'),
     ('France', 'Pinot Gris', 'Jean-Baptiste Adam'),
     ('US', 'Cabernet Sauvignon', 'Kirkland Signature'),
     ('France', 'Gewürztraminer', 'Leon Beyer'),
     ('US', 'Cabernet Sauvignon', 'Louis M. Martini'),
     ('Italy', 'Nerello Mascalese', 'Masseria Setteporte'),
     ('US', 'Chardonnay', 'Mirassou'),
     ('Germany', 'Riesling', 'Richard Böcking'),
     ('Argentina', 'Malbec', 'Felix Lavaque'),
     ('Argentina', 'Malbec', 'Gaucho Andino'),
     ('Spain', 'Tempranillo Blend', 'Pradorey'),
     ('US', 'Meritage', 'Quiévremont')]




```python
# 把数值属性列作离散化
def ParticalDiscretization(df, nume_attr=[], bina_attr=[]):
    new_df = copy.deepcopy(df[bina_attr]);
    for i in nume_attr:
        new_attr = [i + '=0~0.25', i + '=0.25~0.5', i + '=0.5~0.75', i + '=0.75~1.0'];
        tmp = pandas.DataFrame(columns=new_attr, index=df.index);
        k = 0;
        for j in df[i]:
            if j >= df[i].quantile(0.75):
                tmp[i + '=0.75~1.0'][k] = 1;
            elif j >= df[i].quantile(0.5):
                tmp[i + '=0.5~0.75'][k] = 1;
            elif j >= df[i].quantile(0.25):
                tmp[i + '=0.25~0.5'][k] = 1;
            elif j >= df[i].quantile(0):
                tmp[i + '=0~0.25'][k] = 1;
            k = k + 1;
        new_df = pandas.concat([new_df, tmp], axis=1);
    new_df = new_df.fillna(value=0);
    return new_df;
```

# 2. 找出频繁模式

使用apriori算法


```python
from collections import defaultdict
import itertools


def apriori(transactions, support=0.1, confidence=0.8, lift=1, minlen=2, maxlen=2):
    item_2_tranidxs = defaultdict(list)
    itemset_2_tranidxs = defaultdict(list)

    for tranidx, tran in enumerate(transactions):
        for item in tran:
            item_2_tranidxs[item].append(tranidx)
            itemset_2_tranidxs[frozenset([item])].append(tranidx)

    item_2_tranidxs = dict([(k, frozenset(v)) for k, v in item_2_tranidxs.items()])
    itemset_2_tranidxs = dict([
        (k, frozenset(v)) for k, v in itemset_2_tranidxs.items()])

    tran_count = float(len(transactions))
    # print('Extracting rules in {} transactions...'.format(int(tran_count)))

    valid_items = set(item
        for item, tranidxs in item_2_tranidxs.items()
            if (len(tranidxs) / tran_count >= support))

    pivot_itemsets = [frozenset([item]) for item in valid_items]
    freqsets = []

    if minlen == 1:
        freqsets.extend(pivot_itemsets)

    for i in range(maxlen - 1):
        new_itemset_size = i + 2
        new_itemsets = []

        for pivot_itemset in pivot_itemsets:
            pivot_tranidxs = itemset_2_tranidxs[pivot_itemset]
            for item, tranidxs in item_2_tranidxs.items():
                if item not in pivot_itemset:
                    common_tranidxs = pivot_tranidxs & tranidxs
                    if len(common_tranidxs) / tran_count >= support:
                        new_itemset = frozenset(pivot_itemset | set([item]))
                        if new_itemset not in itemset_2_tranidxs:
                            new_itemsets.append(new_itemset)
                            itemset_2_tranidxs[new_itemset] = common_tranidxs

        if new_itemset_size > minlen - 1:
            freqsets.extend(new_itemsets)

        pivot_itemsets = new_itemsets

    # print('{} frequent patterns found'.format(len(freqsets)))

    for freqset in freqsets:
        for item in freqset:
            rhs = frozenset([item])
            lhs = freqset - rhs
            support_rhs = len(itemset_2_tranidxs[rhs]) / tran_count
            if len(lhs) == 0:
                lift_rhs = float(1)
                if support_rhs >= support and support_rhs > confidence and lift_rhs > lift:
                    yield (lhs, rhs, support_rhs, support_rhs, lift_rhs)
            else:
                confidence_lhs_rhs = len(itemset_2_tranidxs[freqset]) \
                    / float(len(itemset_2_tranidxs[lhs]))
                lift_lhs_rhs = confidence_lhs_rhs / support_rhs

                if confidence_lhs_rhs >= confidence and lift_lhs_rhs > lift:
                    support_lhs_rhs = len(itemset_2_tranidxs[freqset]) / tran_count
                    yield (lhs, rhs, support_lhs_rhs, confidence_lhs_rhs, lift_lhs_rhs)
```

频繁项集（support>0.03, confidence>0.1, lift>1）如下：


```python
rules = apriori(transactions, support=0.03, confidence=0.1, lift=1)
rules_sorted = sorted(rules, key=lambda x: (x[4], x[3], x[2]), reverse=True) # ORDER BY lift DESC, confidence DESC, support DESC

for r in rules_sorted:
    print(r)

```

    (frozenset({'Bordeaux-style Red Blend'}), frozenset({'France'}), 0.03635426364342815, 0.6832971800433839, 4.019771773295553)
    (frozenset({'France'}), frozenset({'Bordeaux-style Red Blend'}), 0.03635426364342815, 0.2138686461775223, 4.019771773295553)
    (frozenset({'Cabernet Sauvignon'}), frozenset({'US'}), 0.05628178593686284, 0.7722761824324325, 1.841580575864628)
    (frozenset({'US'}), frozenset({'Cabernet Sauvignon'}), 0.05628178593686284, 0.13421033318655512, 1.841580575864628)
    (frozenset({'Pinot Noir'}), frozenset({'US'}), 0.07605542774926714, 0.7448010849909584, 1.7760630745882846)
    (frozenset({'US'}), frozenset({'Pinot Noir'}), 0.07605542774926714, 0.18136283575517392, 1.7760630745882844)
    (frozenset({'Chardonnay'}), frozenset({'US'}), 0.052327057574381976, 0.5786607674636263, 1.3798825518863749)
    (frozenset({'US'}), frozenset({'Chardonnay'}), 0.052327057574381976, 0.12477983267283135, 1.3798825518863749)
    

## 3. 导出关联规则及其支持度，置信度


```python
import csv 

with open('result.csv', 'wt') as f:
    f_csv = csv.writer(f, delimiter=',')
    f_csv.writerow(['rule', 'sup', 'conf', 'lift'])
    for r in rules_sorted:
        f_csv.writerow([f'{str(list(r[0])[0])} => {str(list(r[1])[0])}', r[2], r[3], r[4]])

pd.read_csv('result.csv')

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rule</th>
      <th>sup</th>
      <th>conf</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bordeaux-style Red Blend =&gt; France</td>
      <td>0.036354</td>
      <td>0.683297</td>
      <td>4.019772</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France =&gt; Bordeaux-style Red Blend</td>
      <td>0.036354</td>
      <td>0.213869</td>
      <td>4.019772</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cabernet Sauvignon =&gt; US</td>
      <td>0.056282</td>
      <td>0.772276</td>
      <td>1.841581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US =&gt; Cabernet Sauvignon</td>
      <td>0.056282</td>
      <td>0.134210</td>
      <td>1.841581</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pinot Noir =&gt; US</td>
      <td>0.076055</td>
      <td>0.744801</td>
      <td>1.776063</td>
    </tr>
    <tr>
      <th>5</th>
      <td>US =&gt; Pinot Noir</td>
      <td>0.076055</td>
      <td>0.181363</td>
      <td>1.776063</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chardonnay =&gt; US</td>
      <td>0.052327</td>
      <td>0.578661</td>
      <td>1.379883</td>
    </tr>
    <tr>
      <th>7</th>
      <td>US =&gt; Chardonnay</td>
      <td>0.052327</td>
      <td>0.124780</td>
      <td>1.379883</td>
    </tr>
  </tbody>
</table>
</div>



## 4. 对规则进行评价，使用Lift， Kulc

上一步已经计算了Lift，这里再计算一下Kulc。


```python
res = []
for r in rules_sorted:
    conf1 = r[3]
    for r2 in rules_sorted:
        if r2[0] == r[1] and r2[1] == r[0]:
            conf2 = r2[3]
    kulc = (conf1 + conf2) / 2
    res.append(kulc)

res
```




    [0.4485829131104531,
     0.4485829131104531,
     0.4532432578094938,
     0.4532432578094938,
     0.46308196037306615,
     0.46308196037306615,
     0.3517203000682288,
     0.3517203000682288]



## 5. 对挖掘结果进行分析

这里以Bordeaux-style Red Blend => France为例。

由关联规则可知Bordeaux-style Red Blend这个品种的葡萄酒基本上产自法国，那么我们就来检验一下：


```python
df[df['variety'] == 'Bordeaux-style Red Blend'].sample(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>109896</th>
      <td>France</td>
      <td>Named after the Swedish royal family, this est...</td>
      <td>NaN</td>
      <td>89</td>
      <td>NaN</td>
      <td>Bordeaux</td>
      <td>Haut-Médoc</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Bernadotte 2013  Haut-Médoc</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Bernadotte</td>
    </tr>
    <tr>
      <th>31828</th>
      <td>France</td>
      <td>Ripe fruit, juicy acidity and a fine balance o...</td>
      <td>Boha</td>
      <td>90</td>
      <td>17.0</td>
      <td>Bordeaux</td>
      <td>Blaye Côtes de Bordeaux</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Magdeleine Bouhou 2015 Boha  (Blaye Cô...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Magdeleine Bouhou</td>
    </tr>
    <tr>
      <th>64213</th>
      <td>France</td>
      <td>Wood and smoke aromas precede ripe and dusty t...</td>
      <td>Famille Lapalu</td>
      <td>86</td>
      <td>10.0</td>
      <td>Bordeaux</td>
      <td>Médoc</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaines Lapalu 2008 Famille Lapalu  (Médoc)</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Domaines Lapalu</td>
    </tr>
    <tr>
      <th>116444</th>
      <td>France</td>
      <td>This is rounded, with its ripe fruit dominatin...</td>
      <td>Cuvée Prestige</td>
      <td>87</td>
      <td>14.0</td>
      <td>Bordeaux</td>
      <td>Bordeaux Supérieur</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château de Cornemps 2009 Cuvée Prestige  (Bord...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château de Cornemps</td>
    </tr>
    <tr>
      <th>77402</th>
      <td>France</td>
      <td>This wine is firm with plenty of structured ta...</td>
      <td>NaN</td>
      <td>93</td>
      <td>NaN</td>
      <td>Bordeaux</td>
      <td>Margaux</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Giscours 2013  Margaux</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Giscours</td>
    </tr>
    <tr>
      <th>86644</th>
      <td>France</td>
      <td>This ripe, bold and generous Gonfrier Frères w...</td>
      <td>NaN</td>
      <td>89</td>
      <td>14.0</td>
      <td>Bordeaux</td>
      <td>Bordeaux</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Tassin 2015  Bordeaux</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Tassin</td>
    </tr>
    <tr>
      <th>36585</th>
      <td>US</td>
      <td>This wine is made from a majority of Cabernet ...</td>
      <td>Winston Hill</td>
      <td>92</td>
      <td>150.0</td>
      <td>California</td>
      <td>Rutherford</td>
      <td>Napa</td>
      <td>Virginie Boone</td>
      <td>@vboone</td>
      <td>Frank Family 2012 Winston Hill Red (Rutherford)</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Frank Family</td>
    </tr>
    <tr>
      <th>51398</th>
      <td>US</td>
      <td>Full bodied, with structured tannins and brigh...</td>
      <td>New World Red</td>
      <td>85</td>
      <td>34.0</td>
      <td>Virginia</td>
      <td>Monticello</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kluge Estate 2009 New World Red Red (Monticello)</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Kluge Estate</td>
    </tr>
    <tr>
      <th>103898</th>
      <td>US</td>
      <td>A proprietary blend of 36% Cabernet Sauvignon,...</td>
      <td>Contrarian</td>
      <td>91</td>
      <td>135.0</td>
      <td>California</td>
      <td>Napa Valley</td>
      <td>Napa</td>
      <td>Virginie Boone</td>
      <td>@vboone</td>
      <td>Blackbird Vineyards 2013 Contrarian Red (Napa ...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Blackbird Vineyards</td>
    </tr>
    <tr>
      <th>78741</th>
      <td>France</td>
      <td>This is a firmly structured wine, solid with f...</td>
      <td>NaN</td>
      <td>92</td>
      <td>NaN</td>
      <td>Bordeaux</td>
      <td>Pessac-Léognan</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Carbonnieux 2014  Pessac-Léognan</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Carbonnieux</td>
    </tr>
    <tr>
      <th>96518</th>
      <td>US</td>
      <td>Based on Merlot, this Bordeaux-style blend is ...</td>
      <td>Bastille</td>
      <td>86</td>
      <td>38.0</td>
      <td>California</td>
      <td>Sonoma County</td>
      <td>Sonoma</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>De Novo 2009 Bastille Red (Sonoma County)</td>
      <td>Bordeaux-style Red Blend</td>
      <td>De Novo</td>
    </tr>
    <tr>
      <th>122324</th>
      <td>US</td>
      <td>After initial scents of smoke and dark toast s...</td>
      <td>Corchaug Estate Ben's Blend</td>
      <td>89</td>
      <td>48.0</td>
      <td>New York</td>
      <td>North Fork of Long Island</td>
      <td>Long Island</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>McCall 2007 Corchaug Estate Ben's Blend Red (N...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>McCall</td>
    </tr>
    <tr>
      <th>56833</th>
      <td>US</td>
      <td>This wine is a blend of Merlot (44%), Cabernet...</td>
      <td>Two Blondes Vineyard</td>
      <td>91</td>
      <td>64.0</td>
      <td>Washington</td>
      <td>Yakima Valley</td>
      <td>Columbia Valley</td>
      <td>Sean P. Sullivan</td>
      <td>@wawinereport</td>
      <td>Andrew Will 2013 Two Blondes Vineyard Red (Yak...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Andrew Will</td>
    </tr>
    <tr>
      <th>63552</th>
      <td>France</td>
      <td>An impressive blend of Merlot and Malbec, this...</td>
      <td>Comtesse de Ségur</td>
      <td>90</td>
      <td>19.0</td>
      <td>Southwest France</td>
      <td>Montravel</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Laulerie 2012 Comtesse de Ségur  (Mont...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Laulerie</td>
    </tr>
    <tr>
      <th>75613</th>
      <td>France</td>
      <td>94-96 Barrel sample. Full of blackcurrant frui...</td>
      <td>Barrel sample</td>
      <td>95</td>
      <td>NaN</td>
      <td>Bordeaux</td>
      <td>Margaux</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Giscours 2009 Barrel sample  (Margaux)</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Giscours</td>
    </tr>
    <tr>
      <th>33862</th>
      <td>US</td>
      <td>High alcohol, softness and tremendously ripe f...</td>
      <td>Maquette</td>
      <td>87</td>
      <td>38.0</td>
      <td>California</td>
      <td>Paso Robles</td>
      <td>Central Coast</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sculpterra 2010 Maquette Red (Paso Robles)</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Sculpterra</td>
    </tr>
    <tr>
      <th>99251</th>
      <td>France</td>
      <td>This structured wine has 25-year old vines as ...</td>
      <td>NaN</td>
      <td>88</td>
      <td>10.0</td>
      <td>Bordeaux</td>
      <td>Bordeaux</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Arnaucosse 2012  Bordeaux</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Arnaucosse</td>
    </tr>
    <tr>
      <th>65728</th>
      <td>Argentina</td>
      <td>This is an elegant wine that shows that Argent...</td>
      <td>Pasionado Cuatro Cepas</td>
      <td>91</td>
      <td>50.0</td>
      <td>Mendoza Province</td>
      <td>Mendoza</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Andeluna 2005 Pasionado Cuatro Cepas Red (Mend...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Andeluna</td>
    </tr>
    <tr>
      <th>57233</th>
      <td>France</td>
      <td>The vineyard surrounds a 14th century castle, ...</td>
      <td>NaN</td>
      <td>90</td>
      <td>36.0</td>
      <td>Bordeaux</td>
      <td>Puisseguin Saint-Émilion</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Langlais 2010  Puisseguin Saint-Émilion</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Langlais</td>
    </tr>
    <tr>
      <th>129193</th>
      <td>France</td>
      <td>Layered tannins and wood flavors have tended t...</td>
      <td>NaN</td>
      <td>84</td>
      <td>18.0</td>
      <td>Bordeaux</td>
      <td>Bordeaux Supérieur</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Château Laville 2015  Bordeaux Supérieur</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Château Laville</td>
    </tr>
  </tbody>
</table>
</div>



再绘制一个直方图如下，可以看出，Bordeaux-style Red Blend这个品种的葡萄酒产自法国的确比较多。


```python
df[df['variety'] == 'Bordeaux-style Red Blend']['country'].value_counts().plot(kind='bar')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-cd95f9fe0e4f> in <module>
    ----> 1 df[df['variety'] == 'Bordeaux-style Red Blend']['country'].value_counts().plot(kind='bar')
    

    NameError: name 'df' is not defined


对规则进行可视化如下


```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('result.csv')
plt.scatter(df['sup'], df['conf'], c=df['lift'], s=20, cmap='Reds')
plt.xlabel('sup')
plt.ylabel('conf')
cb = plt.colorbar()
cb.set_label('lift')
plt.show()
```


![png](output_17_0.png)



```python

```
