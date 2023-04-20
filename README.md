# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
Program Developed By : Praveen kumar.S
Register Number : 212222230108
```

## Data.csv
```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```

## Encoding.csv

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```

## Titanic.csv

```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```

# OUPUT
## Data.csv:
## Initial dataset:
![1](https://user-images.githubusercontent.com/119559827/233379619-6f3b61c4-7ebf-4b81-961a-dd74b42089a6.png)

## Binary Encoding:
![2](https://user-images.githubusercontent.com/119559827/233380015-92550473-c267-4f80-83a0-118aa378245a.png)

## Encoded Dataset:
![3](https://user-images.githubusercontent.com/119559827/233380346-3a39c437-5b61-48fe-a72e-2410695279e4.png)

## Data Scaling using MinMaxScaler:
![4](https://user-images.githubusercontent.com/119559827/233380589-9f718684-8956-42dd-b9be-7f3817ead2cd.png)

## Data Scaling using StandardScaler:
![5](https://user-images.githubusercontent.com/119559827/233380930-b2159a15-d008-4fe0-9360-fc85649721a5.png)

## Data Scaling using MaxAbsScaler:
![6](https://user-images.githubusercontent.com/119559827/233381310-56dab29d-4c04-412f-9da1-4ff165dd2ad6.png)

## Data Scaling using RobustScaler:
![7](https://user-images.githubusercontent.com/119559827/233381573-48007d34-44c1-47bb-8ebb-a8ebb639dbd7.png)

# Encoding.csv :

## Initial Dataset:
![1](https://user-images.githubusercontent.com/119559827/233381934-330f873c-bc0c-4777-aa32-8047767970f1.png)

## Binary Encoding:
![2](https://user-images.githubusercontent.com/119559827/233382294-17897583-ddd2-4acc-a9a6-a646dd81e042.png)

## Encoded Dataset:
![3](https://user-images.githubusercontent.com/119559827/233382472-a234e52d-a7e5-4af2-bead-9dbcaf47347e.png)

## Data Scaling using MinMaxScaler:
![4](https://user-images.githubusercontent.com/119559827/233382642-dfd84c6a-dd8b-41c5-8619-edc325cd420d.png)

## Data Scaling using StandardScaler:
![5](https://user-images.githubusercontent.com/119559827/233382881-7b6673f7-f882-4763-88aa-01bdd72bd791.png)

## Data Scaling using MaxAbsScaler:
![6](https://user-images.githubusercontent.com/119559827/233383510-5b277fbe-45b3-4f01-8ab6-409356be57a4.png)

## Data Scaling using RobustScaler:
![7](https://user-images.githubusercontent.com/119559827/233383797-f88a114b-b26f-4ffb-8edf-134708dcb84a.png)

# Titanic.csv :
## Initial Dataset:
![1](https://user-images.githubusercontent.com/119559827/233384442-bb748548-0d96-4663-b5db-6d65c0d4192e.png)

## Data cleaning before encoding:
![2](https://user-images.githubusercontent.com/119559827/233384592-8c0aeef8-85af-4056-9504-374e22fdf5b3.png)

## Cleaned Dataset:
![3](https://user-images.githubusercontent.com/119559827/233385010-6602cd76-17df-497d-a382-8109c4484c6a.png)

## Binary Encoding:
![4](https://user-images.githubusercontent.com/119559827/233385191-8b854d05-5349-4bb3-a4b9-abd10995e76b.png)

## Encoded Dataset:
![5](https://user-images.githubusercontent.com/119559827/233385367-6f87a7ce-58ad-4a3e-a7e3-09e30025baed.png)

## Data Scaling using MinMaxScaler:
![6](https://user-images.githubusercontent.com/119559827/233385668-f22a5d43-5372-4848-a836-d0a33ccf565f.png)

## Data Scaling using StandardScaler:
![7](https://user-images.githubusercontent.com/119559827/233385978-460c680d-4df5-425d-aa3e-c6775362e4fd.png)

## Data Scaling using MaxAbsScaler:
![8](https://user-images.githubusercontent.com/119559827/233386240-889f0848-5083-4e62-8c6b-0518fc2cc7a8.png)

## Data Scaling using RobustScaler:
![9](https://user-images.githubusercontent.com/119559827/233386464-432b0b60-5d20-432f-9fac-2b2a8efcb165.png)

# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.

