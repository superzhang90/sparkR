
1、k-means算法实现：
###算法程序 
library(sparklyr)
library(dplyr)

Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
data("iris")
names(iris) <- c("Sepal_Length","Sepal_Width","Petal_Length","Petal_Width","Species")
head(iris)

irisDF <- suppressWarnings(createDataFrame(iris))
kmeansDF <- irisDF
kmeansTestDF <- irisDF
kmeansModel <- spark.kmeans(kmeansDF, ~ Sepal_Length + Sepal_Width + Petal_Length + Petal_Width,k = 3)
summary(kmeansModel)

showDF(fitted(kmeansModel))
kmeansPredictions <- predict(kmeansModel, kmeansTestDF)
showDF(kmeansPredictions)

###算法程序
library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
df <- createDataFrame(iris)
names(iris) <- c("Sepal_Length","Sepal_Width","Petal_Length","Petal_Width","Species")

model <- spark.kmeans(df, Sepal_Length ~ Sepal_Width, k = 4, initMode = "random")
summary(model)
fitted <- predict(model, df)
head(select(fitted, "Sepal_Length", "prediction"))
path <- "path/to/model"
write.ml(model, path)
savedModel <- read.ml(path)
summary(savedModel)


###顾客活跃度数据聚类分析
library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
curwd = setwd("D:/R/work")
df <- read.csv('activedata.csv',header = TRUE)
head(df)
dataDF <- df[,2:4]
names(dataDF) <- c("counts","timelength","totallength")
head(dataDF)
dfdata <- suppressWarnings(createDataFrame(dataDF))
kmeansModel <- spark.kmeans(dfdata, ~ counts + timelength + totallength,k = 3,initMode = "random")
summary(kmeansModel) 


2、逻辑回归logistic（2.1.0以上版本支持）
#D sparkR.session()
# binary logistic regression
df <- createDataFrame(iris)
training <- df[df$Species %in% c("versicolor", "virginica"), ]
model <- spark.logit(training, Species ~ ., regParam = 0.5)
summary <- summary(model)
# fitted values on training data
fitted <- predict(model, training)
# save fitted model to input path
path <- "path/to/model"
write.ml(model, path)
# can also read back the saved model and predict
# Note that summary deos not work on loaded model
savedModel <- read.ml(path)
summary(savedModel)
# multinomial logistic regression
df <- createDataFrame(iris)
model <- spark.logit(df, Species ~ ., regParam = 0.5)
summary <- summary(model)


3、UDF自定义函数模型试验：
###算法程序
library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
###连接本地spark
sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
###创建数据集 （数据为R自带）
df <- as.DataFrame(faithful)
head(faithful)
schema <- structType(structField("eruptions", "double"), structField("waiting", "double"),
                     structField("waiting_secs", "double"))
df1 <- dapply(df, function(x) { x <- cbind(x, x$waiting * 100) }, schema)
head(collect(df1))

4、UDF自定义函数模型试验： 
###算法程序
library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
###连接本地spark
sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
curwd = setwd("D:/R/work")
df <- read.csv('activedata.csv',header = TRUE)
head(df)
###创建数据集 (数据为顾客活跃度数据)
dfdata <- createDataFrame(df)

ldf <- dapplyCollect(
  dfdata,
  function(x) {
    x <- cbind(x, "T-t" = x$T-x$t)
  })
head(ldf, 6)

5、随机森林算法
###算法程序
# fit a Random Forest Regression Model
library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
###连接本地spark
sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
df <- createDataFrame(longley)
head(df)
model <- spark.randomForest(df, Employed ~ ., type = "regression", maxDepth = 5, maxBins = 16)
# get the summary of the model
summary(model)
# make predictions
predictions <- predict(model, df)
# save and load the model
path <- "path/to/model"
write.ml(model, path)
savedModel <- read.ml(path)
summary(savedModel)
# fit a Random Forest Classification Model
df <- createDataFrame(iris)
model <- spark.randomForest(df, Species ~ Petal_Length + Petal_Width, "classification")


6、朴素贝叶斯
###算法程序
library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
###连接本地spark
sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
data <- as.data.frame(UCBAdmissions)
df <- createDataFrame(data)

# fit a Bernoulli naive Bayes model
model <- spark.naiveBayes(df, Admit ~ Gender + Dept, smoothing = 0)

# get the summary of the model
summary(model)

# make predictions
predictions <- predict(model, df)

# save and load the model
path <- "path/to/model"
write.ml(model, path)
savedModel <- read.ml(path)
summary(savedModel)

7、广义线性模型GLM
###算法程序
library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
###连接本地spark
sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
data(iris)
df <- createDataFrame(iris)
model <- spark.glm(df, Sepal_Length ~ Sepal_Width, family = "gaussian")
summary(model)

# fitted values on training data
fitted <- predict(model, df)
head(select(fitted, "Sepal_Length", "prediction"))

# save fitted model to input path
path <- "path/to/model"
write.ml(model, path)

# can also read back the saved model and print
savedModel <- read.ml(path)
summary(savedModel)

8、加速失效时间（AFT）生存回归模型
###算法程序
library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="C:/Users/Administrator/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
###连接本地spark
sc <- sparkR.session(master = "local", sparkConfig = list(spark.driver.memory = "2g"))
df <- createDataFrame(ovarian)
model <- spark.survreg(df, Surv(futime, fustat) ~ ecog_ps + rx)

# get a summary of the model
summary(model)

# make predictions
predicted <- predict(model, df)
showDF(predicted)

# save and load the model
path <- "path/to/model"
write.ml(model, path)
savedModel <- read.ml(path)
summary(savedModel)














