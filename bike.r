rm(list=ls())
setwd("C:/Users/asus/Desktop/project4")
#load library
library(plyr)
library(corrplot)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(randomForest)
library(party)
library("DMwR")
library("corrgram")
library("scales")
library("psych")
library("gplots")
library(mlr)
library(xgboost)
library(readr)
library(car)
library(stringr)
library(Metrics)
library(rpart)
library(ggplot2)                           # data visualization
library(readr)                             # CSV file I/O, e.g. the read_csv function
library(base)                              # for finding the file names
library(data.table)                        # for data input and wrangling
library(dplyr)                             # for data wranging
library(forecast)                          # for time series forecasting
library(tseries)                           # for time series data    
library(lubridate)                         # for date modifications
library(tidyr)                             # for data wrangling
library(magrittr)                          # for data wrangling
#loading data
#Load data
data = read.csv("day.csv",header = TRUE)
str(data)
#missing value analysis
sapply(data, function(x) sum(is.na(x)))

#rename variables name
names(data)[2:5] <- c("dateday","season","year","month")
names(data)[9] <- "weatherlist" 
names(data)[12] <- "humidity" 
names(data)[16] <- "count" 
data=rename(data,c("dtedata" = "datedata" ,"yr" = "year","mnth" = "month","weathersit" = "weatherlist","hum" = "humidity","cnt" = "count"))

#histograms
hist(data$season)
hist(data$year)
hist(data$weatherlist)
hist(data$temp)
hist(data$atemp)
hist(data$humidity)
hist(data$windspeed)
hist(data$casual)
hist(data$registered)

#setting proper datatype

for(i in c(3,4,5,6,7,8:9)) {
  data[,i] = as.factor(data[,i])
}

#plot

#Multivariate #Scatter Plot
ggplot(data, aes_string(x = data$atemp, y = data$count)) + 
  geom_point(aes_string(colour = data$weekdata),size = 4) +
  theme_bw()+ ylab("total no. of customers") + xlab("temprature") + ggtitle("Weekdata wise daily distribution of count") + 
  theme(text=element_text(size=25)) + 
  scale_x_continuous(breaks=pretty_breaks(n=10)) +
  scale_y_continuous(breaks=pretty_breaks(n=10)) +
  scale_colour_discrete(name="weekdata")

#bar plot
ggplot(data, aes_string(x = data$month, y = data$count)) +
  geom_bar(stat="identity",fill =  "Blue") + theme_bw() +
  xlab("Month") + ylab('Count') + scale_y_continuous(breaks=pretty_breaks(n=10)) +
  ggtitle("Monthly distribution of counts") +  theme(text=element_text(size=15))

#bar plot
ggplot(data, aes_string(x = data$season, y = data$count)) +
  geom_bar(stat="identity",fill =  "Blue") + theme_bw() +
  xlab("Month") + ylab('Count') + scale_y_continuous(breaks=pretty_breaks(n=10)) +
  ggtitle("Seasonly distribution of counts") +  theme(text=element_text(size=15))

#bar plot
ggplot(data, aes_string(x = data$year, y = data$count)) +
  geom_bar(stat="identity",fill =  "Blue") + theme_bw() +
  xlab("Month") + ylab('Count') + scale_y_continuous(breaks=pretty_breaks(n=10)) +
  ggtitle("Yearly distribution of counts") +  theme(text=element_text(size=15))

 

############################################Outlier Analysis#############################################
# ## BoxPlots - Distribution and Outlier Check


ggplot(data, aes_string(x = data$count, y = data$temp, 
                                  fill = data$count)) + 
  geom_boxplot(outlier.colour = "red", outlier.size = 3) + 
  scale_y_continuous(breaks=pretty_breaks(n=10)) + 
  guides(fill=FALSE) + theme_bw() + xlab("Count") + ylab("temprature") +
  ggtitle("Outlier Analysis") +  
  theme(text=element_text(size=20))




df = subset(data, select = c(temp,atemp,humidity,windspeed,casual,registered)) #selecting only numeric
cnames = colnames(df)

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "count"), data = subset(df))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="count")+
           ggtitle(paste("Box plot of count for",cnames[i])))
}

# ## Plotting plots together
#gridExtra::grid.arrange(temp,atemp,humidity,windspeed,ncol=4)
#gridExtra::grid.arrange(gn5,gn6,ncol=2)



## Correlation Plot 
corrgram(df, order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")  


# create a function for denormalisartion
tconvert <- function(min, max, vector){
  result <- vector * (max - min) + min
  return (result)
}

# apply the function and denormalise the temperature values
data$temp <- tconvert(-8, 39, data$temp)
data$atemp <- tconvert(-16, 50, data$atemp)

#ANALYSIS

### TASKS 2, 9

# calculate mean, st.dev and median for each season
# by aggregation with dplyr library
library(dplyr)
data.agg <- data %>%
  group_by(season) %>%
  summarise(
    temp.min = min(temp),
    temp.max = max(temp),
    temp.med = median(temp),
    temp.stdev = sd(temp),
    temp.mean = mean(temp), 
    count = n())
data.agg

### TASK 8

# create a boxplot for temperature by season
boxplot(temp ~ season,
        data = data,
        xlab = "Season",
        ylab = "Temperature",
        main = "Temperature by Season",
        col = "skyblue")

# check seasons and respective months
# fall months
unique(data$month[data$season=="1"])

# winter months
unique(data$month[data$season=="2"])

# spring months
unique(data$month[data$season=="3"])

# summer months
unique(data$month[data$season=="4"])

### TASK 8

# create a beanplot for number of bike rents per each weather condition
library("beanplot")
require("beanplot")
require("RColorBrewer")
bean.cols <- lapply(brewer.pal(6, "Set3"),
                    function(x){return(c(x, "black", "gray", "red"))})
beanplot(count ~ weatherlist,
         data = data,
         main = "Bike Rents by Weather Condition",
         xlab = "Weather Condition",
         ylab = "Number of rentals",
         col = bean.cols,
         lwd = 1,
         what = c (1,1,1,0),
         log = ""
)

### TASK 11

# create a data frame
df <- data.frame(spring = rep(NA, 3),
                 winter = rep(NA, 3),
                 summer = rep(NA, 3),
                 fall = rep(NA, 3))
row.names(df) <- c("mean", "median", "sd")

# fill the data frame with corresponding mean, median and sd values
vec <- c ("mean","median","sd") 
for (n in vec){
  for (i in unique(data$season)) {
    my.fun <- get(n)
    res <- my.fun(data$count[data$season == i])
    df[n,i] <- res
  }
}  
df

# statistics (analysis of variance model)
summary(aov(count ~ season, data = data))

# pairwise comparison of means for seasons
# in order to identify any difference between two means that is greater than the expected standard error
TukeyHSD(aov(count ~ season, data = data))

### TASK 8

# create a boxplot for count~season in order to reveal the most popular season
# for bike rentals

boxplot(count ~ season,
        data = data,
        xlab = "Season",
        ylab = "Count",
        main = "Count by Season",
        col = "yellow3")

### TASK 4

# correlation test for count~atemp
t1 <- cor.test(data$atemp[data$year == 0],
               data$count[data$year == 0])
t1

t2 <- cor.test(data$atemp[data$year == 1], 
               data$count[data$year == 1])
t2

# apa format
library("yarrr")
apa(t1)

apa(t2)

### TASKS 5, 6

# plotting the results in a scatterplot with regression lines

# blank plot
plot(x = 1,
     xlab = "Temperature",
     ylab = "Number of Rents",
     xlim = c(-25,50),
     ylim = c(0,1000),
     main = "Temperature vs. Count")

# draw points for 2011 year
points(x = data$atemp[data$year == 0],
       y = data$count[data$year == 0],
       pch = 16,
       col = "red",
       cex = 0.5
)
# draw points for 2012 year
points(x = data$atemp[data$year == 1],
       y = data$count[data$year == 1],
       pch = 16,
       col = "darkgreen",
       cex = 0.5
)

# add regression lines for two ears
abline(lm(count~atemp, data, subset = year == 0),
       col = "darkgreen",
       lwd = 3)

abline(lm(count~atemp, data, subset = year == 1),
       col = "red",
       lwd = 3)

# add legend
legend("topleft",
       legend = c(2011, 2012),
       col = c("darkgreen","red"),
       pch = c(16, 16),
       bg = "white",
       cex = 1
)

### TASK 5 

# summary on linear model fitting
summary(lm(count~weatherlist, data))

summary(aov(count~weatherlist, data))

### TASK 9

# calculate min, max, mean, st.dev and median for each season
# by aggregation with dplyr library

w.agg <- data %>%
  group_by(weatherlist) %>%
  summarise(
    temp.min = min(temp),
    temp.max = max(temp),
    temp.mean = mean(temp),
    temp.stdev = sd(temp),
    temp.med = median(temp), 
    count = n())
w.agg

### TASKS 7, 11 

# create histograms for each weather condition
# to explore distribution of the bike rentals by 
# weather condition

# create a vector for histograms titles
vec <- c("Clear Weather", "Cloudy Weather", "Rainy Weather", "Thunderstorm Weather")

# parameters for plots combining
par(mfrow = c(2, 2))

# create 4 histograms with a loop
for (i in c(1:4)){
  name.i <- vec[i]
  hist(data$count[data$weatherlist == i],
       main = name.i,
       xlab = "Number of Rents",
       ylab = "Frequency",
       breaks = 10,
       col = "yellow3",
       border = "black")
  
  # the line indicating median value
  abline(v = median(data$count[data$weatherlist == i]),
         col = "black", 
         lwd = 3, 
         lty = 2) 
  
  # the line indicating mean value
  abline(v = mean(data$count[data$weatherlist == i]),
         col = "blue", 
         lwd = 3, 
         lty = 2) 
}

### TASK 3

t <- t.test(data$count[data$holiday == 0],
            data$count[data$holiday == 1])

# apa format
apa(t)

# TASK 8

beanplot(count ~ holiday,
         data = data,
         main = "Bike Rents by Type of a data",
         xlab = "Type of data",
         ylab = "Number of rents",
         col = bean.cols,
         lwd = 1,
         what = c(1,1,1,0),
         log = ""
)

#Divide the data into data and test
#set.seed(123)
data_index = sample(1:nrow(data), 0.8 * nrow(data))
train = data[data_index,]
test = data[-data_index,]

str(train)
str(test)

train_deleted = subset(train,select= -c(instant,dateday,temp))
test_deleted = subset(test,select = -c(instant,dateday,temp))


# ##rpart for regression
#fit = rpart(count ~ ., data = train_deleted, method = "anova")

#Predict for new test cases
#predictions_DT = predict(fit, test_deleted[,-13])


#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

#MAPE(test_deleted[,13], predictions_DT)


#Error Rate: 11.75
#Accuracy: 88.25

#Linear Regression
#check multicollearity
#library(usdm)
#vif(train_deleted[,-13])

#vifcor(train_deleted[,-13], th = 0.9)




#run regression model
#lm_model = lm(count ~., data = train_deleted)

#Summary of the model
#summary(lm_model)

#Predict
#predictions_LR = predict(lm_model, test_deleted[,1:13])

#Calculate MAPE
#MAPE(test[,13], predictions_LR)


#Error Rate: 8.8
#acuracy: 91.2%

###Random Forest
RF_model = randomForest(count ~ ., train_deleted, importance = TRUE, ntree = 500)

#Extract rules fromn random forest
#transform rf object to an inTrees' format
library(RRF)
library(inTrees)
treeList = RF2List(RF_model) 

# 
# #Extract rules
exec = extractRules(treeList, train_deleted[,-13])  # R-executable conditions
# 
# #Visualize some rules
exec[1:2,]
# 
# #Make rules more readable:
readableRules = presentRules(exec, colnames(train_deleted))
readableRules[1:2,]
# 
# #Get rule metrics
ruleMetric = getRuleMetric(exec, train_deleted[,-13], train_deleted$count)  # get rule metrics
# 
# #evaulate few rules
ruleMetric[1:2,]

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test_deleted[,-13])

##Evaluate the performance of regression model
#Calculate MAPE
MAPE(test_deleted[,13], RF_Predictions)


#Accuracy = 94.0%
#Error rate = 6.0%


