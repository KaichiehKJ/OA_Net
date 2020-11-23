library(data.table)
library(magrittr)

# Optimizer

my_optimizer = mx.opt.create(name = "sgd", learning.rate = 0.005, momentum = 0.9, wd = 0)

mx.set.seed(0)
new_arg = mxnet:::mx.model.init.params(symbol = ce_loss,
                                       input.shape = list(data = c(800, 800, 3, 12), label = c(6, 12)),
                                       output.shape = NULL,
                                       initializer = mxnet:::mx.init.uniform(0.01),
                                       ctx = mx.gpu(1))

#for (i in 1:length(new_arg$arg.params)) {
#  pos <- which(names(res_model$arg.params) == names(new_arg$arg.params)[i])
#  if (all.equal(dim(res_model$arg.params[[pos]]), dim(new_arg$arg.params[[i]])) == TRUE) {
#    new_arg$arg.params[[i]] <- res_model$arg.params[[pos]]
#  }
#}

#for (i in 1:length(new_arg$aux.params)) {
#  pos <- which(names(res_model$aux.params) == names(new_arg$aux.params)[i])
#  if (all.equal(dim(res_model$aux.params[[pos]]), dim(new_arg$aux.params[[i]])) == TRUE) {
#    new_arg$aux.params[[i]] <- res_model$aux.params[[pos]]
#  }
#}



for (i in names(new_arg$arg.params)) {
  if (i %in% names(res_model$arg.params)) {
    if (length(new_arg$arg.params[[i]]) == length(res_model$arg.params[[i]])) {
      new_arg$arg.params[[i]] <- res_model$arg.params[[i]]
    }
  }
}

for (i in names(new_arg$aux.params)) {
  if (i %in% names(res_model$aux.params)) {
    if (length(new_arg$aux.params[[i]]) == length(res_model$aux.params[[i]])) {
      new_arg$aux.params[[i]] <- res_model$aux.params[[i]]
    }
  }
}
