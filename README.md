## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.


# FEATURE ENCODING:

1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

3. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

5. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

7. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.



# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
     # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
import pandas as pd

df=pd.read_csv("/content/Encoding Data.csv")

df

![image](https://github.com/user-attachments/assets/235a3d63-0263-4162-9fa8-8efe8efa529f)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

![image](https://github.com/user-attachments/assets/c11ec7d7-0f41-499c-9eaa-c622067e9d58)

df['bo2']=e1.fit_transform(df[["ord_2"]])

df

![image](https://github.com/user-attachments/assets/2b70f161-33f1-47c8-a0ac-691616af7e71)

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc

![image](https://github.com/user-attachments/assets/8488d188-9845-4795-b837-74702f1d0120)

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))


from sklearn.preprocessing import OneHotEncoder

# Use sparse_output instead of sparse

ohe=OneHotEncoder(sparse_output=False) 

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))


df2=pd.concat([df2,enc],axis=1)

df2

![image](https://github.com/user-attachments/assets/ef3969d8-4ab8-4084-899c-d4ea053a1e7b)

pd.get_dummies(df2,columns=["nom_0"])

![image](https://github.com/user-attachments/assets/946ec041-80bd-4ce0-b012-3121c7a9d6ef)

pip install --upgrade category_encoders

![image](https://github.com/user-attachments/assets/1cd07bb8-cb41-41bf-a7c5-a6de41040c92)

from category_encoders import BinaryEncoder

df=pd.read_csv("/content/data.csv")

df

![image](https://github.com/user-attachments/assets/d40d19d0-53e9-45ba-9491-cd1e327bfd66)

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb

![image](https://github.com/user-attachments/assets/c7b40ec0-8c18-40bb-91a3-f2ac559c22c5)

from category_encoders import TargetEncoder

te=TargetEncoder()

CC=df.copy()

new=te.fit_transform(X=CC["City"],y=CC["Target"])

CC=pd.concat([CC,new],axis=1)

CC

![image](https://github.com/user-attachments/assets/8ddad87b-5de5-4d80-a782-698f840323a3)

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("/content/Data_to_Transform.csv")

df

![image](https://github.com/user-attachments/assets/5c58dd3f-9502-41db-8478-94236dd43031)

df.skew()

![image](https://github.com/user-attachments/assets/31b90a34-322a-4b55-85e4-1f92d0eb95d7)

np.log(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/6d9964f7-6fab-4a71-bfa5-4857d8bf4c1f)

np.reciprocal(df["Moderate Positive Skew"])

![image](https://github.com/user-attachments/assets/89f18943-ca2d-48d9-9c28-2cc0e5197673)

np.sqrt(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/b8ed6c8a-99f1-44e1-90d7-50d36a1a6185)

np.square(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/4a240dd0-c84f-4a33-88f1-683b18faaeeb)

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])

df

![image](https://github.com/user-attachments/assets/b53243fe-316e-4bb0-bf1e-138681ec4423)

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])

![image](https://github.com/user-attachments/assets/f4f827cd-e74b-4965-a9fd-95b4763446d2)

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/379c254f-9ae6-42da-b00a-7c0acc3d4d23)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()

![image](https://github.com/user-attachments/assets/f7cbda12-8692-4ee8-b6ca-16794757cb70)

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/7aef38ce-43d4-4449-bf18-c9ef0422e012)

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/894760af-f6f7-438f-a105-8ab94627d1f3)

sm.qqplot(df["Highly Negative Skew_1"],line='45')

plt.show()

![image](https://github.com/user-attachments/assets/6487359f-a409-40e4-acb2-b76ce5e2eb35)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()

![image](https://github.com/user-attachments/assets/cc6b7eee-3bfd-46e3-b657-03a58e356c9e)


# RESULT:
       # INCLUDE YOUR RESULT HERE
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       

       
