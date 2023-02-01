```python
#Importing Pandas for data manipulations
#numpy for working with arrays and use it library for tranforms works.
#matplot for ploting graphs and seaborn for statistical analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt
import seaborn as sb
```


```python
#lets import the data set
housing_data = pd.read_csv("~/Downloads/housing.csv")
```


```python
#to check few rows of data on the csv file
housing_data.head()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
#to check the number of rows and columns
housing_data.shape
```




    (20640, 10)




```python
#the data set CSV files has 20640 rows and 10 columns.

#lets findout if there is any missing data in the file.

housing_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB



```python
#from the output we can see that total_bedrooms has 207 data missing.

#this null fields can be filled using python function called fillna()

#by calculating median value of total_bedrooms column.

median = housing_data['total_bedrooms'].median()
housing_data['total_bedrooms'] = housing_data['total_bedrooms'].fillna(median)
```


```python
#now check the info of the dataset again
housing_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20640 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB



```python
#there is no null values now.

#from the info we can see that, ocean_proximity is the one field we can use to categorize the house.

#that how closer each house closer to the beach. Normally the more closer to the beach the costlier it would be.

#so lets count that field data too see how many houses closer to beach or away from it.

housing_data['ocean_proximity'].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
#ok, now lets get the overview of our dataset to understand it futher by using the describe() method.
housing_data.describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>536.838857</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>419.391878</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>297.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>643.250000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#here we can see the min and max value of the median_house_value which is 14999 to 500001.

#Note:- Ocean proximity field is not numerical data hence its not taken by the describe() method.

#we can use sklearn library to numarize it later

#lets visualise the dataset to understand it further using matplot library

housing_data.hist(bins=50, figsize=(20, 10))
mplt.show()
```


    
![png](output_9_0.png)
    



```python
#lets check the media_house_value graph. 
#it shows there are more numbers of house between the prices 100000 to 200000.
#and fewer house are priced high.

#lets plot one more map to show the house price based on the lat and lang, polulation density
housing_data.plot(kind="scatter", x="longitude", y="latitude", 
                  s=housing_data['population']/100, label="population", 
                  figsize=(10, 7), c="median_house_value", cmap=mplt.get_cmap('jet'), colorbar=True)
mplt.show()
```


    
![png](output_10_0.png)
    



```python
#we can see "median_house_value" based on "longitude" and "latitude", 
#with marker size representing "population" and color representing "median_house_value". 
#It includes a colorbar and uses "jet" colormap

y = housing_data["median_house_value"]
housing_data.drop("median_house_value", axis=1, inplace=True)
```


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(housing_data, y, test_size=0.2, random_state=42)
print("Training dataset size:", x_train.shape)
print("Test dataset size:", x_test.shape)
```

    Training dataset size: (16512, 9)
    Test dataset size: (4128, 9)



```python
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoder.fit(x_train["ocean_proximity"].values)
x_train_op_lb =  encoder.transform(x_train["ocean_proximity"].values)
x_test_op_lb = encoder.transform(x_test["ocean_proximity"].values)

print("x_train_op_lb:", x_train_op_lb.shape)
print("x_test_op_lb:", x_test_op_lb.shape)
```

    x_train_op_lb: (16512, 5)
    x_test_op_lb: (4128, 5)



```python
from sklearn.preprocessing import StandardScaler
x_train_num = x_train.drop("ocean_proximity", axis = 1)
x_test_num = x_test.drop("ocean_proximity", axis = 1)
standardscaler = StandardScaler()
standardscaler.fit(x_train_num)
x_train_num_ss = standardscaler.transform(x_train_num)
x_test_num_ss = standardscaler.transform(x_test_num)

print("x_train_num_ss_shape:", x_train_num_ss.shape)
print("x_test_num_ss_shape:", x_test_num_ss.shape)
```

    x_train_num_ss_shape: (16512, 8)
    x_test_num_ss_shape: (4128, 8)



```python
x_tr = np.hstack((x_train_num_ss, x_train_op_lb))
x_ts = np.hstack((x_test_num_ss, x_test_op_lb))
```


```python
x_tr
```




    array([[ 1.27258656, -1.3728112 ,  0.34849025, ...,  0.        ,
             0.        ,  1.        ],
           [ 0.70916212, -0.87669601,  1.61811813, ...,  0.        ,
             0.        ,  1.        ],
           [-0.44760309, -0.46014647, -1.95271028, ...,  0.        ,
             0.        ,  1.        ],
           ...,
           [ 0.59946887, -0.75500738,  0.58654547, ...,  0.        ,
             0.        ,  0.        ],
           [-1.18553953,  0.90651045, -1.07984112, ...,  0.        ,
             0.        ,  0.        ],
           [-1.41489815,  0.99543676,  1.85617335, ...,  0.        ,
             1.        ,  0.        ]])




```python
x_ts
```




    array([[ 0.28534728,  0.1951    , -0.28632369, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.06097472, -0.23549054,  0.11043502, ...,  0.        ,
             0.        ,  0.        ],
           [-1.42487026,  1.00947776,  1.85617335, ...,  0.        ,
             1.        ,  0.        ],
           ...,
           [-1.23041404,  0.78014149, -0.28632369, ...,  0.        ,
             0.        ,  0.        ],
           [-0.08860699,  0.52740357,  0.58654547, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.60445493, -0.66608108, -0.92113763, ...,  0.        ,
             0.        ,  0.        ]])




```python
x_tr.shape
```




    (16512, 13)




```python
x_ts.shape
```




    (4128, 13)




```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
for_reg = RandomForestRegressor()
for_reg.fit(x_tr, y_train)

predictions = for_reg.predict(x_tr)
msme = mean_squared_error(predictions, y_train)
rsme = np.sqrt(msme)
rsme
```




    18202.02359925946



