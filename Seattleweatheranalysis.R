#Catarina Jesus
#June 18th, 2023

# Importing all required libraries library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2) 
library(tidyverse) 
library(tseries) 
library(rugarch) 
library(dplyr) 
library(corrplot)

# importing the data source
df <- read.csv("seattle-weather.csv", encoding="UTF-8")

#See nulls 
colSums(is.na(df))

#See blanks 
colSums(df == "")

#Check names columns names(df)
unique(df$weather) 

#####################################################################

# Extract the year from the date column 
df$month <- month(as.Date(df$date)) 
df$year <- year(as.Date(df$date))

#Replace the weather column by 1 and 0
df$binary <- ifelse(df$weather %in% c("rain", "drizzle", "fog", "snow"), 0, 1) 

## Correlation between variables

cor_matrix <- cor(df[,c("binary", "precipitation",
                        "temp_max", "temp_min", "wind")]) 
cor_matrix


# Plot the correlation graph
corrplot(cor_matrix, method = "circle", tl.col = "black", tl.srt = 45)

## Normalize data
## Removing units using normalization 
min_max <- function(x){
  normalize <- (x-min(x))/(max(x)-min(x))
  return(normalize) }

df$precipitation_norm <- min_max(df$precipitation) 
df$temp_max_norm <- min_max(df$temp_max) 
df$temp_min_norm <- min_max(df$temp_min) 
df$wind_norm <- min_max(df$wind)

## sampling into TRAINIG and TESTING
training_idx <- sample(1:nrow(df), size=0.8*nrow(df)) 
my_df_train <- df[training_idx,]
my_df_test <- df[-training_idx,]


#### LOGISTIC REGRESSION MODEL ####

##Normalize variables

## The normalization is made to compare variables between each other
my_logit <- glm(binary ~ precipitation_norm+temp_max_norm+temp_min_norm+wind_norm,
                data = my_df_train, family = "binomial") 
summary(my_logit)

#precipitation_norm 
exp(-3491.5156)-1 #-1 #temp_max_norm 
exp(5.5578)-1 #258.2519 #temp_min_norm 
exp(-3.3214)-1 #-0.9638977 #wind_norm
exp(2.3884)-1 #9.896046

my_prediction <- predict(my_logit, my_df_test, type="response")

confusionMatrix(data= as.factor(as.numeric(my_prediction>0.5)), reference= as.factor(as.numeric(my_df_test$binary)))

#Confusion Matrix and Statistics 
# Reference
#Prediction 0 1
#0       141 3
#1       34 115

#Accuracy : 0.8737
#95% CI : (0.8302, 0.9095)
#No Information Rate : 0.5973 
#P-Value [Acc > NIR] : < 2.2e-16

#Confusion Matrix and Statistics 
# Reference
#Prediction 0 1
#0        13 03
#1       35 125

#Accuracy : 0.8703
#95% CI : (0.8264, 0.9066)
#No Information Rate : 0.5631 
#P-Value [Acc > NIR] : < 2.2e-16

##Not normalize variables
my_logit1 <- glm(binary ~ precipitation + temp_max + temp_min + wind , data = my_df_train, family = "binomial")
summary(my_logit1)

my_prediction1 <- predict(my_logit1, my_df_test, type="response") 

confusionMatrix(data=as.factor(as.numeric(my_prediction1>0.5)), reference=as.factor(as.numeric(my_df_test$binary)))

#Confusion Matrix and Statistics 
# Reference
#Prediction 0 1
#0        13 03
#1      35 125

#Accuracy : 0.8703
#95% CI : (0.8264, 0.9066)
#No Information Rate : 0.5631 
#P-Value [Acc > NIR] : < 2.2e-16

#We exclude precipitation once the p-value is >0.5 which means that they are statically insignificant
my_logit2 <- glm(binary ~ temp_max + temp_min + wind , data = my_df_train, family = "binomial")
summary(my_logit2)

my_prediction2 <- predict(my_logit2, my_df_test, type="response") 

confusionMatrix(data=as.factor(as.numeric(my_prediction2>0.5)),reference=as.factor(as.numeric(my_df_test$binary)))

#Confusion Matrix and Statistics 
# Reference
#Prediction 0 1
#0       134 53
#1        31 75

#Accuracy : 0.7133
#95% CI : (0.6578, 0.7644)
#No Information Rate : 0.5631 
#P-Value [Acc > NIR] : 8.486e-08


#################################################################
#We will create a stratified sample and see which one between random sample and stratified samples gives a better accuracy and confusion matrix
#Stratified sample - the population is divided into homogeneous subgroups based on specific characteristics or attributes
#Random sample - observations are selected randomly from the population without any specific consideration for the population's characteristics or subgroups

#Create a stratified sample 
set.seed(123)

# Create a stratified sample
training_idx_strat <- createDataPartition(df$binary, p = 0.8, list = FALSE)

# Subset the data based on the stratified sample indices 
training_data_strat <- df[training_idx_strat, ] 
testing_data_strat <- df[-training_idx_strat, ]

# building a decision tree
my_tree <- rpart(binary ~ precipitation+temp_max+temp_min+wind, data=training_data_strat, method="class", cp = 0.0008)
summary(my_tree)

#Plot the decision tree 
rpart.plot(my_tree, type=1, extra=1)

#Accuracy and performance for decision tree
# testing performance of your model
my_df_tree_predict <- predict(my_tree, testing_data_strat, type="prob")

#Accuracy
confusionMatrix(data = as.factor(as.numeric(my_df_tree_predict[,2]>0.5)) ,
                reference= as.factor(as.numeric(testing_data_strat$binary)))

#Confusion Matrix and Statistics 
# Reference
#Prediction 0 1
#0       128 10
#1       40 114

#Accuracy : 0.8288
#95% CI : (0.7806, 0.8702)
#No Information Rate : 0.5753 
#P-Value [Acc > NIR] : < 2.2e-16

# building a decision tree with random sample
my_tree1 <- rpart(binary ~ precipitation+temp_max+temp_min+wind, data=my_df_train, method="class", cp = 0.001)
summary(my_tree1)

#Plot the decision tree 
rpart.plot(my_tree1, type=1, extra=1)

## Lets assume that we have a case study: # if explicit is zero - yes left, no right
## if yes means that binary is zero which means is not business success #Accuracy and performance for decision tree

# testing performance of your model
my_df_tree_predict1 <- predict(my_tree1, my_df_test, type="prob")

#Accuracy
confusionMatrix(data = as.factor(as.numeric(my_df_tree_predict1[,2]>0.5)) ,
                reference= as.factor(as.numeric(my_df_test$binary)))

#Confusion Matrix and Statistics 
# Reference
#Prediction 0 1
#0       132 15
#1       33 113

#Accuracy : 0.8362
#95% CI : (0.7887, 0.8767) 
#No Information Rate : 0.5631 
#P-Value [Acc > NIR] : < 2e-16

######################################################## 

##Forecasting

## We will forecast 3 variables, which are the variables with higher influence on the y variable of the prediction (binary)
## The 3 variables are: precipitation, temp_max and wind

#Group by variable by year 
precipitation <- df %>% 
  group_by(year, month) %>%
  summarize(avg_precipitation = mean(precipitation, na.rm = TRUE))

temp_max <- df %>%
  group_by(year, month) %>%
  summarize(avg_temp_max = mean(temp_max, na.rm = TRUE))

wind <- df %>%
  group_by(year, month) %>%
  summarize(avg_wind = mean(wind, na.rm = TRUE))

colSums(is.na(precipitation))

#plot

ggplot(precipitation, aes(x = month, y = avg_precipitation, group = year, color = factor(year))) +
  geom_line(size = 1) +
  labs(x = "Month", y = "Average Precipitation", title = "Average Precipitation by Month") + theme_minimal() +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(title = "Year", nrow = 1, override.aes = list(size = 1.5),
                              label.theme = element_text(size = 12))) + scale_color_discrete(labels = c("2012", "2013", "2014", "2015"))
                

ggplot(temp_max, aes(x = month, y = avg_temp_max, group = year, color = factor(year))) + geom_line(size = 1) +
  labs(x = "Month", y = "Average max Temperature ", title = "Average max Temperature by Month") + theme_minimal() +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(title = "Year", nrow = 1, override.aes = list(size = 1.5),
                              label.theme = element_text(size = 12))) + scale_color_discrete(labels = c("2012", "2013", "2014", "2015"))

ggplot(wind, aes(x = month, y = avg_wind, group = year, color = factor(year))) + geom_line(size = 1) +
  labs(x = "Month", y = "Average wind ", title = "Average wind by Month") + theme_minimal() +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(title = "Year", nrow = 1, override.aes = list(size = 1.5),
                              label.theme = element_text(size = 12))) + scale_color_discrete(labels = c("2012", "2013", "2014", "2015"))

#adf test
adf.test(precipitation$avg_precipitation)
#Since the p-value above .05, we accept the null hypothesis.
#This means the time series is non-stationary.
#In other words, it has some time-dependent structure and does not have constant variance over time.

adf.test(temp_max$avg_temp_max)
#Since the p-value is lower .05 we don't accept the null hypothesis. #This means the time series is stationary.

adf.test(wind$avg_wind)
#Since the p-value is lower .05 we don't accept the null hypothesis. #This means the time series is stationary.


# Decomposition of the non-stationary data

#On the right side, decompose our non-stationary into semi-stationary or stationary
#We can see that precipitation is semi-stationary
ts_precipitation <- ts(precipitation[,c("year", "avg_precipitation")], frequency = 14, start=c(2012)) 
dec_precipitation <- decompose(ts_precipitation)
plot(dec_precipitation)

#We can see that temp_max is stationary
ts_temp_max <- ts(temp_max[,c("year", "avg_temp_max")], frequency = 14, start=c(2012)) 
dec_temp_max <- decompose(ts_temp_max)
plot(dec_temp_max)

#We can see that wind is semi-stationary
ts_wind <- ts(wind[,c("year", "avg_wind")], frequency = 14, start=c(2012)) 
dec_wind <- decompose(ts_wind)
plot(dec_wind)

#acf and pacf 
acf(precipitation$avg_precipitation)

acf(temp_max$avg_temp_max)

acf(wind$avg_wind)

pacf(precipitation$avg_precipitation)

pacf(temp_max$avg_temp_max)

pacf(wind$avg_wind)


############################################################ 

#ARIMA forecasting

##AR - 1
## I - 1 - SEMI-STATIONARY
## MA - 1
precipitation_arima <- arima(precipitation$avg_precipitation,
                             order=c(1,1,1)) 
predict(precipitation_arima, n.ahead =12)

#### forecasting for the next 12 months: 9.012280 9.112194 9.051743 9.088318 9.066189 9.079577 9.071477 9.076378 9.073413 9.075207 9.074121 9.074778
#### with a level of uncertainty of (interval of error): 2.212021 2.987504 3.669911 4.206964 4.702844 5.140114 5.549207 5.926624 6.283414 6.619851 6.940663 7.246901


##AR - 3
## I - 0 - STATIONARY
## MA - 2
temp_max_arima <- arima(temp_max$avg_temp_max,
                        order=c(3,0,2)) 
predict(temp_max_arima, n.ahead =12)

#### forecasting for the next 12 months: 2.002874 2.584622 2.944451 2.963718 2.988545 3.260744 3.613116 3.954603 4.086378 19.513835 15.547872 11.674406 9.540338
#### with a level of uncertainty of (interval of error): 2.002874 2.584622 2.944451 2.963718 2.988545 3.260744 3.613116 3.954603 4.086378 4.101938 4.134502 4.279594


##AR - 1
## I - 1 - SEMI-STATIONARY
## MA - 2
wind_arima <- arima(wind$avg_wind,
                    order=c(1,1,2)) 
predict(wind_arima, n.ahead =12)

#### forecasting for the next 12 months: 3.984740 4.121233 4.049976 4.087176 4.067755 4.077894 4.072601 4.075364 4.073922 4.074675 4.074282 4.074487
#### with a level of uncertainty of (interval of error): 0.4978905 0.5705063 0.6792200 0.7515005 0.8273784 0.8919926 0.9545946 1.0121701 1.0672188 1.1192768 1.1691624 1.2169329


##############################################################

#GARCH model
model_param <- ugarchspec(mean.model=list(armaOrder=c(0,0)), variance.model= list(model="sGARCH", garchOrder=c(1,1)), distribution.model="norm")

### precipitation
garch_model1 <- ugarchfit(data=precipitation$avg_precipitation,
                          spec=model_param, out.sample = 20) 
print(garch_model1)

## forecasting Bootstrapping GARCH
bootstrap1 <- ugarchboot(garch_model1, method = c("Partial", "Full")[1],
                         n.ahead = 500, n.bootpred = 500) 
print(bootstrap1)

### According GARCH model, we expect the forecast to change around 223%
## T+1: 2.2325 - 223%, variance seems high, which means the future volatility will be high ## T+2: 2.2342
## T+3: 2.2360
## T+4: 2.2378
## T+5: 2.2395


#GARCH model
### temp_max
garch_model2 <- ugarchfit(data=temp_max$avg_temp_max,
                          spec=model_param, out.sample = 20) 
print(garch_model2)

## forecasting Bootstrapping GARCH
bootstrap2 <- ugarchboot(garch_model2, method = c("Partial", "Full")[1],
                         n.ahead = 500, n.bootpred = 500) 
print(bootstrap2)

### According GARCH model, we expect the forecast to change around 617%
## T+1: 6.1799 - 617%, variance seems high, which means the future volatility will be high ## T+2: 6.1760
## T+3: 6.1721
## T+4: 6.1681
## T+5: 6.1642


#GARCH model
### wind
garch_model3 <- ugarchfit(data=wind$avg_wind,
                          spec=model_param, out.sample = 20) 
print(garch_model3)

## forecasting Bootstrapping GARCH
bootstrap3 <- ugarchboot(garch_model3, method = c("Partial", "Full")[1],
                         n.ahead = 500, n.bootpred = 500) 
print(bootstrap3)

### According GARCH model, we expect the forecast to change around 56%
## T+1: 0.55868 - 56%, variance seems high, which means the future volatility will be high ## T+2: 0.55988
## T+3: 0.56107
## T+4: 0.56226
## T+5: 0.56345