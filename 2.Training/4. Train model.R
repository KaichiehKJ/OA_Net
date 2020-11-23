# Libraries

library(mxnet)
library(psych)

#1. Build an executor to train model

source('~/OA/code/2.Training/1.Iterator.R')
source('~/OA/code/2.Training/2.model.R')
source('~/OA/code/2.Training/3.Suppot fuction.R')

Fixed_NAMES = names(res_model$arg.params)[names(res_model$arg.params) %in% names(mx.symbol.infer.shape(flatten0_output, data = c(448, 448, 3, 6))$arg.shapes)]
#Fixed_NAMES = names(chexnet_model$arg.params)[names(chexnet_model$arg.params) %in% names(mx.symbol.infer.shape(densenet0_pool4_fwd_output, data = c(512, 512, 3, 12))$arg.shapes)]


my_executor = mx.simple.bind(symbol = ce_loss,
                             data = c(760, 760, 3, 12), label = c(6, 12), 
                             ctx = mx.gpu(1), grad.req = "write")

num_arg_list <- sapply(lapply(new_arg$arg.params, dim), prod)
message('The total number of parameters = ', sum(num_arg_list))


#2. Set the initial parameters

mx.exec.update.arg.arrays(my_executor, new_arg$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(my_executor, new_arg$aux.params, match.name = TRUE)

#3. Define the updater

my_updater = mx.opt.get.updater(optimizer = my_optimizer, weights = my_executor$ref.arg.arrays)

my_iter <- my_iterator_func(iter = NULL, batch_size = 12, sample_type = 'train', aug_flip = TRUE,
                            aug_crop = TRUE, aug_rotate = FALSE, oversampling = TRUE)

#val_data
val_label <- val_table
val_ids <- unique(val_table[,'names'])
img_array <- array(0, dim = c(800, 800, 3, length(val_ids)))
#for (i in 1:length(val_ids)) {
  
#  img <- readJPEG(img_list[[val_ids[i]]])
  
#  random.row <- sample(0:64, 1)
#  random.col <- sample(0:64, 1)
  
#  img_array[,,,i] <- img[random.row+1:(512-64),random.col+1:(512-64),,drop = FALSE]
  
#}


for (i in 1:length(val_ids)) {

  img_array[,,,i]  <- readJPEG(img_list[[val_ids[i]]])

}

vald.X_list = list()
vald.X_list[[1]] = img_array[1:760, 1:760,,]; dim(vald.X_list[[1]]) = c(760, 760, 3, length(val_ids))
vald.X_list[[2]] = img_array[41:800, 1:760,,]; dim(vald.X_list[[2]]) = c(760, 760, 3, length(val_ids))
vald.X_list[[3]] = img_array[1:760, 41:800,,]; dim(vald.X_list[[3]]) = c(760, 760, 3, length(val_ids))
vald.X_list[[4]] = img_array[41:800, 41:800,,]; dim(vald.X_list[[4]]) = c(760, 760, 3, length(val_ids))
vald.X_list[[5]] = img_array[21:780, 21:780,,]; dim(vald.X_list[[5]]) = c(760, 760, 3, length(val_ids))


KL_pred_val <- matrix(0, ncol = 1, nrow = nrow(val_label))
for (i in 1:nrow(KL_pred_val)) {
  if (val_label[i,2] == 1) {
    KL_pred_val[i,1] <- 1
  } else if (val_label[i,3] == 1) {
    KL_pred_val[i,1] <- 2
  } else if (val_label[i,4] == 1) {
    KL_pred_val[i,1] <- 3
  }else if (val_label[i,5] == 1) {
    KL_pred_val[i,1] <- 4
  }else if (val_label[i,6] == 1) {
    KL_pred_val[i,1] <- 5
  }else if (val_label[i,7] == 1) {
    KL_pred_val[i,1] <- 6
  }
}


prefix = 'model/OA_model/OA_model'

batch_inform_freq = 50
batch_size = 12
kappa_list <- list()

decode_func <- function (x) {
  
  x <- as.array(x)
  out <- max.col(t(x)) - 1
  return(out)
  
}


#message("Start training with ", ndevice, " devices")
for (i in 1:100) {
  
  message('Start training: round = ', i)
  my_iter$reset()
  batch_loss = NULL
  t0 <- Sys.time()
  batch_seq <- 0
  
  while (my_iter$iter.next()) {
    batch_seq <- batch_seq + 1
    my_values <- my_iter$value()
    mx.exec.update.arg.arrays(my_executor, arg.arrays = my_values, match.name = TRUE)
    mx.exec.forward(my_executor, is.train = TRUE)
    mx.exec.backward(my_executor)
    update_args = my_updater(weight = my_executor$ref.arg.arrays, grad = my_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
    batch_loss = c(batch_loss, as.array(my_executor$ref.outputs$ce_loss_output))
    if (batch_seq %% batch_inform_freq == 0 | batch_seq < 10) {
      message(paste0("epoch [", i, "] batch [", batch_seq, "] loss =  ", 
                     formatC(mean(unlist(batch_loss)), 6, format = "f"), " (Speed: ",
                     formatC(batch_seq * batch_size/as.numeric(Sys.time() - t0, units = 'secs'), format = "f", 2), " samples/sec)"))
    }
  }
  message(paste0("epoch = ", i, ": ce_loss = ", formatC(mean(batch_loss), format = "f", 4)))
  my_model <- mxnet:::mx.model.extract.model(symbol = sigmoid,
                                             train.execs = list(my_executor))
  
  #my_model <- mxnet:::mx.model.extract.model(symbol = softmax,
  #                                           train.execs = list(my_executor))
  
  
  #predict_Y1 = predict(my_model, Test.X1)
  #predict_Y2 = predict(my_model, Test.X2)
  #predict_Y3 = predict(my_model, Test.X3)
  #predict_Y4 = predict(my_model, Test.X4)
  #predict_Y5 = predict(my_model, Test.X5)
  
  #predict_Y = (predict_Y1 + predict_Y2 + predict_Y3 + predict_Y4 + predict_Y5) / 5
  
  pred_vald.Y_list = list()
  pred_vald.Y = 0
  for (j in 1:5) {
    pred_vald.Y_list[[j]] = predict(my_model, vald.X_list[[j]], array.batch.size = 16, ctx = mx.gpu(1))
    pred_vald.Y = pred_vald.Y_list[[j]] + pred_vald.Y
  }
  pred_vald.Y = pred_vald.Y/5
  confusion_table = table(factor(max.col(t(pred_vald.Y)), levels = 1:6), factor(KL_pred_val, levels = 1:6))
  print(confusion_table)
  kappas_valid <- cohen.kappa(confusion_table)
  
  kappa_list[i] <- kappas_valid$weighted.kappa
  
  message(paste0("epoch [", i, "] weighted kappa = ", formatC(kappa_list[[i]], format = "f", 4)))
  message(paste0("best epoch [", which.max(kappa_list), "] weighted kappa = ", formatC(kappa_list[[which.max(kappa_list)]], format = "f", 4)))
  
  message(cat("Testing accuracy rate =", sum(diag(confusion_table))/sum(confusion_table)))
  
  
  my_model$arg.params <- append(my_model$arg.params, my_model$arg.params[names(my_model$arg.params) %in%Fixed_NAMES])
  mx.model.save(my_model, prefix, i)
  
}

