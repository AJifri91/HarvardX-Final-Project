pacman::p_load(tidyverse,dslabs, caret, broom) 

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
tinytex::install_tinytex()
options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
#######################################
library(Matrix)
library(matrixStats)
library(irlba)

##need this to load into markdown report
save(edx, file = "edx.RData")
save(final_holdout_test, "final_holdout_test.RData")
###################### first part to get mu, biases and get rmse of base model########################
##calculate overall mean of rating
mu <- mean(edx$rating)

##calculate movie bias call it bi
bi <- edx %>% 
  group_by(movieId) %>% 
  summarise(bi = mean(rating - mu)) %>% as.data.frame()

##calculate user bias call it bu
bu <- edx %>% left_join(bi, by = "movieId") %>% 
  mutate(bu_u = rating - mu - bi) %>% 
  group_by(userId) %>% 
  summarise(bu = mean(bu_u)) %>% as.data.frame()

##we now can use the linear model to predict ratings for users
edx_pred <- final_holdout_test %>% left_join(bi, by = "movieId") %>% 
  left_join(bu, by = "userId") %>% 
  mutate(pred = mu + bi + bu)

RMSE(edx_pred$pred, final_holdout_test$rating)


#################get y matrix and preprocess it to be ready for matrix factorization#################

##create matrix y from edx with unique users as rows and unique movies as columns
##filled with corresponding ratings
y <- edx %>% select(userId, movieId, rating) %>%
  pivot_wider(names_from = movieId, values_from = rating) 
y <- as.matrix(y[,-1])

##need this to load into markdown report
save(y, file = "y.RData")

##get columns names of y to be used later in second part
colnames_y <- colnames(y)
##get userId of edx to be used later in second part
uid <- unique(edx$userId)

str(y)

##the code below iteratively replaces NAs in y with -1 to preserve memory
size <- 1000  #adjust the chunk size based on available memory
n_rows <- nrow(y)

for (i in seq(1, n_rows, size)) {
  chunk_end <- min(i + size - 1, n_rows)
  y[i:chunk_end, ] <- replace(y[i:chunk_end, ],is.na(y[i:chunk_end, ]),-1)
}
rm(chunk_end, size, i, n_rows)

##remove movie bias bi from columns (movies) of y. match bi values to movieId order 
##in y since bi is not ordered
y <- t(t(y)-bi$bi[match(as.integer(colnames(y)), bi$movieId)]) 

##remove user bias bu
y <- y - bu$bu



################################ data visualization part ###########################################
##visualize na values in our matrix sample 1000 rows and columns

ind <- sample(1:nrow(y), 1000)
ind1 <- sample(1:ncol(y), 1000)
image(y[ind, ind1])

##visualize count of users who rated
edx %>% group_by(userId) %>% summarise(n =n()) %>%
  ggplot(aes(n))+
  geom_histogram()+
  scale_x_log10()

##visualize count of movies rated
edx %>% group_by(movieId) %>% summarise(n =n()) %>%
  ggplot(aes(n))+
  geom_histogram()+
  scale_x_log10()

##################### Part two matrix factorisation IRLBA and rmse corrseponding to it###############

##y is ready for matrix factorization after removing bi, NAs and bu
##perform "implicitly restarted Lanczos bidiagonalization algorithm" (IRLBA)
##to approximate 800 right and left singular vectors of svd
##running this code could take up to half a day
## you can just load the file rpca_800_nc which contains the object rpca from which you can get rij

rpca <- irlba(y, nv = 800, center = F, scale = F)

save(rpca, file = "rpca_800_nc.RData")

##get rij which is the matrix multiplication of left and right singular vectors obtained
r <- rpca$u%*%t(rpca$v)

##name columns of with the same original y columns
colnames(r) <- colnames_y

##having tried many variations of irbla trying singular vectors approximations 200,400,600,800. 
##choosing 800 as more than that the gain was too little to consider given the run time of the algorithm

sum(colVars(rpca$u%*%diag(rpca$d)))/4636.547 ##total variance of columns in y
rm(edx, final_holdout_test)

##total_var_exp_200: 2962.988/4636.547 = 0.6390506
##total_var_exp_400: 3241.901/4636.547 = 0.6992059
##total_var_exp_600: 3403.758/4636.5 = 0.7341223
##total_var_exp_800: 3580.26/4636.5 = 0.7721902


##sum overall mean of rating mu and user bias to the resdula matrix r
r <- r+mu+bu$bu
rm(bu)

##sum user bias by chunking r into 2 parts to preserve memory
r1 <- t(t(r[1:35000,])+bi$bi[match(as.integer(colnames(r)), bi$movieId)])
r2 <- t(t(r[35001:69878,])+bi$bi[match(as.integer(colnames(r)), bi$movieId)])
rm(r)

##join the two chunks to form our final y_hat predicted users/movies matrix
r <- rbind(r1,r2)
rm(r1,r2, bi,colnames_y)


rm(edx)

##now we are ready for testing, select the needed columns for test set
final_holdout_test <- final_holdout_test %>%  select(userId, movieId, rating)

##given memory issues, we are going to chunk the testing into 6 parts, getting mean square error for 
##each part by dividing on total numbers of rows in test set.

mse <- function(predicted, observed){
  sum((predicted - observed)^2)/nrow(final_holdout_test)
}

##6 chunks of r our prediction matrix
ind1 <- 1:11646 ; ind2 <- 11647:23293; ind3 <- 23294:34939; ind4 <- 34940:46585;
ind5 <- 46586:58231; ind6 <- 58232:69878

##use ind1:ind6 in the corresponding two places in the code below r[..] and uid[..], each time you get
##an mse for that part set it aside and do the next. once all 6 mse are present sum them and use sqrt
##to get total rmse

r[ind1,] %>% as.data.frame() %>% cbind(userId = uid[ind1]) %>%
  pivot_longer(cols =-"userId" , names_to = "movieId",values_to = "y_hat" ) %>% 
  mutate(movieId = as.integer(movieId)) %>%
  right_join(final_holdout_test, by =c("userId" ,"movieId")) %>% drop_na() %>%
  summarise(mse = mse(y_hat, rating)) %>% as.data.frame()

mse1 = 0.1230827 ; mse2 = 0.1265856 ; mse3 = 0.1239886; 
mse4 = 0.1249978; mse5 = 0.122252; mse6 = 0.1281323
##Final RMSE
sqrt(mse1+mse2+mse3+mse4+mse5+mse6)



#this part does the chunking of r if memory/harware were suffieinct to handle computation

# Define the chunk size and number of chunks
chunk_size <- ceiling(nrow(r) / 6)
num_chunks <- ceiling(nrow(r) / chunk_size)

# Initialize an empty vector to store the MSE values
mse_values <- numeric(num_chunks)

for (i in 1:num_chunks) {
  # Calculate the starting and ending indices for the current chunk
  start_index <- (i - 1) * chunk_size + 1
  end_index <- min(i * chunk_size, nrow(r))
  
  # Perform the pivot_longer and other operations on the chunk
  mse_values[i] <- r[start_index:end_index, ] %>%
    as.data.frame() %>%
    cbind(userId = uid[start_index:end_index]) %>%
    pivot_longer(cols = -"userId", names_to = "movieId", values_to = "y_hat") %>%
    mutate(movieId = as.integer(movieId)) %>%
    right_join(final_holdout_test, by = c("userId", "movieId")) %>%
    drop_na() %>%
    summarise(mse = mse(y_hat, rating)) %>% as.data.frame() %>% .$mse

} 
rm(start_index, end_index, i)
sqrt(sum(mse_values))
