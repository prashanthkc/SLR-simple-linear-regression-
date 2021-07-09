# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

calory_data = pd.read_csv("C:/Users/hp/Desktop/SLR assi/calories_consumed.csv")

# Exploratory data analysis:
calory_data.describe()
cols = {'Weight gained (grams)':'Weight_gained', 'Calories Consumed':'Calories_Consumed'}
calory_data.rename(cols,axis=1,inplace = True)
#Graphical Representation
import matplotlib.pyplot as plt

plt.bar(height = calory_data['Weight_gained'] , x = np.arange(1, 15, 1))
plt.hist(calory_data['Weight_gained']) #histogram
plt.boxplot(calory_data['Weight_gained']) #boxplot

plt.bar(height = calory_data['Calories_Consumed'], x = np.arange(1, 15, 1))
plt.hist(calory_data['Calories_Consumed']) #histogram
plt.boxplot(calory_data['Calories_Consumed']) #boxplot

# Scatter plot
plt.scatter(x = calory_data['Weight_gained'], y = calory_data['Calories_Consumed'], color = 'green') 

# correlation
np.corrcoef(calory_data['Weight_gained'], calory_data['Calories_Consumed']) 

# Covariance
cov_output = np.cov(calory_data['Weight_gained'], calory_data['Calories_Consumed'])[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Calories_Consumed ~ Weight_gained', data = calory_data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(calory_data['Weight_gained']))

# Regression Line
plt.scatter(calory_data.Weight_gained, calory_data.Calories_Consumed)
plt.plot(calory_data.Weight_gained, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = calory_data.Calories_Consumed - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(calory_data['Weight_gained']), y = calory_data['Calories_Consumed'], color = 'brown')
np.corrcoef(np.log(calory_data.Weight_gained), calory_data.Calories_Consumed) #correlation

model2 = smf.ols('Calories_Consumed ~ np.log(Weight_gained)', data = calory_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(calory_data['Weight_gained']))

# Regression Line
plt.scatter(np.log(calory_data.Weight_gained), calory_data.Calories_Consumed)
plt.plot(np.log(calory_data.Weight_gained), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = calory_data.Calories_Consumed - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = calory_data['Weight_gained'], y = np.log(calory_data['Calories_Consumed']), color = 'orange')
np.corrcoef(calory_data.Weight_gained, np.log(calory_data.Calories_Consumed)) #correlation

model3 = smf.ols('np.log(Calories_Consumed) ~ Weight_gained', data = calory_data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(calory_data['Weight_gained']))
pred3_at = np.exp(pred3)

# Regression Line
plt.scatter(calory_data.Weight_gained, np.log(calory_data.Calories_Consumed))
plt.plot(calory_data.Weight_gained, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = calory_data.Calories_Consumed - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Calories_Consumed) ~ Weight_gained + I(Weight_gained*Weight_gained)', data = calory_data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(calory_data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = calory_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

plt.scatter(calory_data.Weight_gained, np.log(calory_data.Calories_Consumed))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res4 = calory_data.Calories_Consumed - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# The best model
from sklearn.model_selection import train_test_split

train, test = train_test_split(calory_data, test_size = 0.2)

finalmodel = smf.ols('Calories_Consumed ~ Weight_gained', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))

# Model Evaluation on Test data
test_res = test.Calories_Consumed - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.Calories_Consumed - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

############################################Problem 2##############################################
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

delivery_data = pd.read_csv("C:/Users/hp/Desktop/SLR assi/delivery_time.csv")

# Exploratory data analysis:
delivery_data.describe()
cols = {'Delivery Time':'Delivery_Time', 'Sorting Time':'Sorting_Time'}
delivery_data.rename(cols,axis=1,inplace = True)
#Graphical Representation
import matplotlib.pyplot as plt

plt.bar(height = delivery_data['Sorting_Time'] , x = np.arange(1, 22, 1))
plt.hist(delivery_data['Sorting_Time']) #histogram
plt.boxplot(delivery_data['Sorting_Time']) #boxplot

plt.bar(height = delivery_data['Delivery_Time'], x = np.arange(1, 22, 1))
plt.hist(delivery_data['Delivery_Time']) #histogram
plt.boxplot(delivery_data['Delivery_Time']) #boxplot

# Scatter plot
plt.scatter(x = delivery_data['Sorting_Time'], y = delivery_data['Delivery_Time'], color = 'green') 

# correlation
np.corrcoef(delivery_data['Sorting_Time'], delivery_data['Delivery_Time']) 

# Covariance
cov_output = np.cov(delivery_data['Sorting_Time'], delivery_data['Delivery_Time'])[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Delivery_Time ~ Sorting_Time', data = delivery_data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(delivery_data['Sorting_Time']))

# Regression Line
plt.scatter(delivery_data.Sorting_Time, delivery_data.Delivery_Time)
plt.plot(delivery_data.Sorting_Time, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = delivery_data.Delivery_Time - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(delivery_data['Sorting_Time']), y = delivery_data['Delivery_Time'], color = 'brown')
np.corrcoef(np.log(delivery_data.Sorting_Time), delivery_data.Delivery_Time) #correlation

model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = delivery_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(delivery_data['Sorting_Time']))

# Regression Line
plt.scatter(np.log(delivery_data.Sorting_Time), delivery_data.Delivery_Time)
plt.plot(np.log(delivery_data.Sorting_Time), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = delivery_data.Delivery_Time - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = delivery_data['Sorting_Time'], y = np.log(delivery_data['Delivery_Time']), color = 'orange')
np.corrcoef(delivery_data.Sorting_Time, np.log(delivery_data.Delivery_Time)) #correlation

model3 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time', data = delivery_data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(delivery_data['Sorting_Time']))
pred3_at = np.exp(pred3)

# Regression Line
plt.scatter(delivery_data.Sorting_Time, np.log(delivery_data.Delivery_Time))
plt.plot(delivery_data.Sorting_Time, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = delivery_data.Delivery_Time - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = delivery_data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(delivery_data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = delivery_data.iloc[:, 1:2].values
X_poly = poly_reg.fit_transform(X)

plt.scatter(delivery_data.Sorting_Time, np.log(delivery_data.Delivery_Time))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res4 = delivery_data.Delivery_Time - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# The best model
from sklearn.model_selection import train_test_split

train, test = train_test_split(delivery_data, test_size = 0.2)

finalmodel = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
# test_pred = finalmodel.predict(pd.DataFrame(test))
# pred_test_Delivery_Time = np.exp(test_pred)
# pred_test_Delivery_Time

# Model Evaluation on Test data
test_res = test.Delivery_Time - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.Delivery_Time - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

###########################################Problem 3##########################################
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

emp_data = pd.read_csv("C:/Users/hp/Desktop/SLR assi/emp_data.csv")

# Exploratory data analysis:
emp_data.describe()

#Graphical Representation
import matplotlib.pyplot as plt

plt.bar(height = emp_data['Salary_hike'] , x = np.arange(1, 11, 1))
plt.hist(emp_data['Salary_hike']) #histogram
plt.boxplot(emp_data['Salary_hike']) #boxplot

plt.bar(height = emp_data['Churn_out_rate'], x = np.arange(1, 11, 1))
plt.hist(emp_data['Churn_out_rate']) #histogram
plt.boxplot(emp_data['Churn_out_rate']) #boxplot

# Scatter plot
plt.scatter(x = emp_data['Salary_hike'], y = emp_data['Churn_out_rate'], color = 'green') 

# correlation
np.corrcoef(emp_data['Salary_hike'], emp_data['Churn_out_rate']) 

# Covariance
cov_output = np.cov(emp_data['Salary_hike'], emp_data['Churn_out_rate'])[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Churn_out_rate ~ Salary_hike', data = emp_data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(emp_data['Salary_hike']))

# Regression Line
plt.scatter(emp_data.Salary_hike, emp_data.Churn_out_rate)
plt.plot(emp_data.Salary_hike, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = emp_data.Churn_out_rate - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(emp_data['Salary_hike']), y = emp_data['Churn_out_rate'], color = 'brown')
np.corrcoef(np.log(emp_data.Salary_hike), emp_data.Churn_out_rate) #correlation

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data = emp_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(emp_data['Salary_hike']))

# Regression Line
plt.scatter(np.log(emp_data.Salary_hike), emp_data.Churn_out_rate)
plt.plot(np.log(emp_data.Salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = emp_data.Churn_out_rate - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = emp_data['Salary_hike'], y = np.log(emp_data['Churn_out_rate']), color = 'orange')
np.corrcoef(emp_data.Salary_hike, np.log(emp_data.Churn_out_rate)) #correlation

model3 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike', data = emp_data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(emp_data['Salary_hike']))
pred3_at = np.exp(pred3)

# Regression Line
plt.scatter(emp_data.Salary_hike, np.log(emp_data.Churn_out_rate))
plt.plot(emp_data.Salary_hike, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = emp_data.Churn_out_rate - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = emp_data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(emp_data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = emp_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

plt.scatter(emp_data.Salary_hike, np.log(emp_data.Churn_out_rate))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res4 = emp_data.Churn_out_rate - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# The best model
from sklearn.model_selection import train_test_split

train, test = train_test_split(emp_data, test_size = 0.2)

finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Churn_out_rate = np.exp(test_pred)
pred_test_Churn_out_rate

# Model Evaluation on Test data
test_res = test.Churn_out_rate - pred_test_Churn_out_rate
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Churn_out_rate = np.exp(train_pred)
pred_train_Churn_out_rate

# Model Evaluation on train data
train_res = train.Churn_out_rate - pred_train_Churn_out_rate
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

########################################Problem 4#################################################
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

salary_data = pd.read_csv("C:/Users/hp/Desktop/SLR assi/Salary_Data.csv")

# Exploratory data analysis:
salary_data.describe()

#Graphical Representation
import matplotlib.pyplot as plt

plt.bar(height = salary_data['YearsExperience'] , x = np.arange(1, 31, 1))
plt.hist(salary_data['YearsExperience']) #histogram
plt.boxplot(salary_data['YearsExperience']) #boxplot

plt.bar(height = salary_data['Salary'], x = np.arange(1, 31, 1))
plt.hist(salary_data['Salary']) #histogram
plt.boxplot(salary_data['Salary']) #boxplot

# Scatter plot
plt.scatter(x = salary_data['YearsExperience'], y = salary_data['Salary'], color = 'green') 

# correlation
np.corrcoef(salary_data['YearsExperience'], salary_data['Salary']) 

# Covariance
cov_output = np.cov(salary_data['YearsExperience'], salary_data['Salary'])[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Salary ~ YearsExperience', data = salary_data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(salary_data['YearsExperience']))

# Regression Line
plt.scatter(salary_data.YearsExperience, salary_data.Salary)
plt.plot(salary_data.YearsExperience, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = salary_data.Salary - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(salary_data['YearsExperience']), y = salary_data['Salary'], color = 'brown')
np.corrcoef(np.log(salary_data.YearsExperience), salary_data.Salary) #correlation

model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = salary_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(salary_data['YearsExperience']))

# Regression Line
plt.scatter(np.log(salary_data.YearsExperience), salary_data.Salary)
plt.plot(np.log(salary_data.YearsExperience), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = salary_data.Salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = salary_data['YearsExperience'], y = np.log(salary_data['Salary']), color = 'orange')
np.corrcoef(salary_data.YearsExperience, np.log(salary_data.Salary)) #correlation

model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = salary_data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(salary_data['YearsExperience']))
pred3_at = np.exp(pred3)

# Regression Line
plt.scatter(salary_data.YearsExperience, np.log(salary_data.Salary))
plt.plot(salary_data.YearsExperience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = salary_data.Salary - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = salary_data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(salary_data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = salary_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

plt.scatter(salary_data.YearsExperience, np.log(salary_data.Salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res4 = salary_data.Salary - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# The best model
from sklearn.model_selection import train_test_split

train, test = train_test_split(salary_data, test_size = 0.2)

finalmodel = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Salary = np.exp(test_pred)
pred_test_Salary

# Model Evaluation on Test data
test_res = test.Salary - pred_test_Salary
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Salary = np.exp(train_pred)
pred_train_Salary

# Model Evaluation on train data
train_res = train.Salary - pred_train_Salary
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

###############################Problem 5###########################################################
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

sat_data = pd.read_csv("C:/Users/hp/Desktop/SLR assi/SAT_GPA.csv")

# Exploratory data analysis:
sat_data.describe()

#Graphical Representation
import matplotlib.pyplot as plt

plt.bar(height = sat_data['GPA'] , x = np.arange(1, 201, 1))
plt.hist(sat_data['GPA']) #histogram
plt.boxplot(sat_data['GPA']) #boxplot

plt.bar(height = sat_data['SAT_Scores'], x = np.arange(1, 201, 1))
plt.hist(sat_data['SAT_Scores']) #histogram
plt.boxplot(sat_data['SAT_Scores']) #boxplot

# Scatter plot
plt.scatter(x = sat_data['GPA'], y = sat_data['SAT_Scores'], color = 'green') 

# correlation
np.corrcoef(sat_data['GPA'], sat_data['SAT_Scores']) 

# Covariance
cov_output = np.cov(sat_data['GPA'], sat_data['SAT_Scores'])[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('SAT_Scores ~ GPA', data = sat_data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(sat_data['GPA']))

# Regression Line
plt.scatter(sat_data.GPA, sat_data.SAT_Scores)
plt.plot(sat_data.GPA, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = sat_data.SAT_Scores - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(sat_data['GPA']), y = sat_data['SAT_Scores'], color = 'brown')
np.corrcoef(np.log(sat_data.GPA), sat_data.SAT_Scores) #correlation

model2 = smf.ols('SAT_Scores ~ np.log(GPA)', data = sat_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(sat_data['GPA']))

# Regression Line
plt.scatter(np.log(sat_data.GPA), sat_data.SAT_Scores)
plt.plot(np.log(sat_data.GPA), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = sat_data.SAT_Scores - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = sat_data['GPA'], y = np.log(sat_data['SAT_Scores']), color = 'orange')
np.corrcoef(sat_data.GPA, np.log(sat_data.SAT_Scores)) #correlation

model3 = smf.ols('np.log(SAT_Scores) ~ GPA', data = sat_data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(sat_data['GPA']))
pred3_at = np.exp(pred3)

# Regression Line
plt.scatter(sat_data.GPA, np.log(sat_data.SAT_Scores))
plt.plot(sat_data.GPA, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = sat_data.SAT_Scores - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(SAT_Scores) ~ GPA + I(GPA*GPA)', data = sat_data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(sat_data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = sat_data.iloc[:, 1:2].values
X_poly = poly_reg.fit_transform(X)

plt.scatter(sat_data.GPA, np.log(sat_data.SAT_Scores))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res4 = sat_data.SAT_Scores - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# The best model
from sklearn.model_selection import train_test_split

train, test = train_test_split(sat_data, test_size = 0.2)

finalmodel = smf.ols('SAT_Scores ~ np.log(GPA)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))

# Model Evaluation on Test data
test_res = test.SAT_Scores - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.SAT_Scores - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
###########################################END##########################################