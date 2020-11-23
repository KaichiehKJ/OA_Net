
# Libraries

library(OpenImageR)
library(abind)
library(jpeg)
library(mxnet)

img_info_path <- 'img_info.RData'

load(img_info_path)
OA_data <- read.csv("data/label.csv")

train_table <- OA_data[OA_data[,'group'] == 1, (-2)]
val_table <- OA_data[OA_data[,'group'] == 2, (-2)]
#test_table <- OA_data[OA_data[,'group'] == 3, (-2)]
label_table <- OA_data[, (-2)]

train_table <- train_table[, c(1:7)]


my_iterator_core <- function (batch_size = 6, sample_type = 'train', aug_flip = TRUE,
                              aug_rotate = TRUE, aug_crop = TRUE, oversampling = FALSE) {
  
  batch <-  0
  
  if (sample_type == 'train') {
    sample_ids <- train_table[,'names']
    sample_label <- train_table
  } else if (sample_type == 'val') {
    sample_ids <- val_table[,'names']
    sample_label <- val_table
  } else {
    sample_ids <- label_table[,'names']
    sample_label <- label_table
  }
  
  batch_per_epoch <- floor(length(sample_ids)/batch_size)
  
  reset <- function() {batch <<- 0}
  
  iter.next <- function() {
    
    batch <<- batch + 1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
    
  }
  
  value <- function() {
    
    if (oversampling == TRUE) {
      
      idx_1 = sample(which(sample_label[sample_label[,1] %in% sample_ids, 2] == 1), batch_size/6, replace = TRUE)
      idx_2 = sample(which(sample_label[sample_label[,1] %in% sample_ids, 3] == 1), batch_size/6, replace = TRUE)
      idx_3 = sample(which(sample_label[sample_label[,1] %in% sample_ids, 4] == 1), batch_size/6, replace = TRUE)
      idx_4 = sample(which(sample_label[sample_label[,1] %in% sample_ids, 5] == 1), batch_size/6, replace = TRUE)
      idx_5 = sample(which(sample_label[sample_label[,1] %in% sample_ids, 6] == 1), batch_size/6, replace = TRUE)
      idx_6 = sample(which(sample_label[sample_label[,1] %in% sample_ids, 7] == 1), batch_size/6, replace = TRUE)
      idx = c(idx_1, idx_2, idx_3, idx_4, idx_5, idx_6)
      #idx = c(idx_1, idx_2)
      idx <- sort(idx)
    } else {
      
      idx <- 1:batch_size + (batch - 1) * batch_size
      idx[idx > length(sample_ids)] <- sample(1:(idx[1]-1), sum(idx > length(sample_ids)))
      idx <- sort(idx)
      
    }
    
    img_array_list <- list()
    for (i in 1:batch_size) {
      #print(sample_ids[idx[i]])
      img_array_list[[i]] <- readJPEG(img_list[[sample_ids[idx[i]]]])
    }
    
    img_array <- abind(img_array_list, along = 4)
    
    
    if (aug_flip) {
      
      if (sample(c(TRUE, FALSE), 1)) {
        
        img_array <- img_array[,dim(img_array)[2]:1,,,drop = FALSE]
        
      }
      
    }
    
    
    if (aug_rotate) {
      
      for (i in 1:batch_size) {
        
        ROTATE_ANGLE <- sample(c(0:15, 345:359), 1)
        img_array[,,,i] <- rotateImage(img_array[,,,i], ROTATE_ANGLE)
        
      }
      
    }
    
    if (aug_crop) {
      
      random.row <- sample(0:40, 1)
      random.col <- sample(0:40, 1)
      
      img_array <- img_array[random.row+1:(800-40),random.col+1:(800-40),,,drop = FALSE]
      
    } 
    
    label <- array(0, dim =c(6, batch_size)) 
    for(i in 1:batch_size) {
      label[,i] <- t(t(as.numeric(sample_label[sample_label[,1] == sample_ids[idx[i]], 2:7])))
      #label[,i] <- t(t(as.numeric(sample_label[sample_label[,1] == sample_ids[idx[i]], 2])))
    }
    
    data = mx.nd.array(img_array)
    label = mx.nd.array(label)
    
    return(list(data = data, label = label))
    
  }
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch, sample_ids = sample_ids))
}

my_iterator_func <- setRefClass("Custom_Iter",
                                fields = c("iter", "batch_size", "sample_type", "img_list", "select_sample", 'aug_flip', 'aug_crop', 'aug_rotate', 'oversampling'),
                                contains = "Rcpp_MXArrayDataIter",
                                methods = list(
                                  initialize = function(iter, batch_size = 12, sample_type = 'train', aug_flip = TRUE, aug_crop = TRUE, aug_rotate = TRUE, oversampling = TRUE) {
                                    .self$iter <- my_iterator_core(batch_size = batch_size, sample_type = sample_type,
                                                                   aug_flip = aug_flip, aug_crop = aug_crop, aug_rotate = aug_rotate, oversampling = oversampling)
                                    .self
                                  },
                                  value = function(){
                                    .self$iter$value()
                                  },
                                  iter.next = function(){
                                    .self$iter$iter.next()
                                  },
                                  reset = function(){
                                    .self$iter$reset()
                                  },
                                  finalize=function(){
                                  }
                                )
)

# Test iterator function
# You can delete symbol # for running the test

#my_iter <- my_iterator_func(iter = NULL, batch_size = 12, sample_type = 'train', aug_flip = TRUE,
#                            aug_crop = FALSE, aug_rotate = TRUE, oversampling = TRUE)

#my_iter$reset()

#t0 <- Sys.time()

#my_iter$iter.next()

#test <- my_iter$value()
