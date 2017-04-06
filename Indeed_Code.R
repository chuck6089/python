##install and load packages
libs <- c("tm", "plyr", "class")
lapply(libs, install.packages, character.only = T)
lapply(libs, require, character.only = T)

options(stringsAsFactors = F)


##set working derectory
setwd("D:/ML_Py/Indeed-Tagging-Raw-Job-Descriptions")

##load data, train.tsv has 4375 rows,
           #test.tsv has 2921 rows including the header
train <- read.table("train.tsv",
                    sep = '\t',
                    quote = "", 
                    fill = T, 
                    encoding="UTF-8",
                    comment.char = "", 
                    header = T,
                    na.strings = c('',"NA"))
test <- read.table("test.tsv",
                   sep = '\t', 
                   quote = "", 
                   fill = T, 
                   comment.char = "", 
                   encoding="UTF-8",
                   header = T,
                   na.strings = c('',"NA"))


##clean the train dataset by delecting the rows under "tags" without values.
train.clean <- na.omit(train)
test.clean <- na.omit(test)

##turn data into corpus
df.train <- as.data.frame(train.clean)
df.test <- as.data.frame(test.clean)
train.Corpus <- Corpus(VectorSource(df.train$description))
test.Corpus <- Corpus(VectorSource(df.test$description))                      

##clean corpus funtion
clean.corpus <- function (corpus) {
  corpus.tmp <- tm_map (corpus, removePunctuation)
  corpus.tmp <- tm_map (corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map (corpus.tmp, tolower)
  corpus.tmp <- tm_map (corpus.tmp, removeWords, stopwords("en"))
  return (corpus.tmp)
}

train.clean <- clean.corpus(train.Corpus)
test.clean <- clean.corpus(test.Corpus)

##Generate term document matrix
#train.tdm <- removeSparseTerms(TermDocumentMatrix(train.clean),0.9)

train.tdm <- TermDocumentMatrix(train.clean)
#test.tdm <- TermDocumentMatrix(test.clean)

s.t <- t(data.matrix(train.tdm[["train.tdm"]]))

str(train.tdm)


tags <- data.frame(ID=1:3504, tags=train.clean$tags)
tags.split <- data.frame(do.call("rbind",strsplit(as.character(df$tags)," ",fixed=T)))

train.tdm <- TermDocumentMatrix(train.clean$description)

str(df2)



# transform the "tags" variable in the train dataset into a vector
tags.type <- c("art-time-job","full-time-job", "hourly-wage",
                "salary","associate-needed","bs-degree-needed", 
                "ms-or-phd-needed","licence-needed","1-year-experience-needed",
                "2-4-years-experience-needed","5-plus-years-experience-needed","supervising-job")

## build the model using trainnig set and run it on the test set
### k-Nearest Neighbour Classification
# knn function

## calculate the accuracy using confusion matrix
conf.mat <- table ("predictation" = , "Actural" = )
accuracy <- sum(diag(conf.mat))

