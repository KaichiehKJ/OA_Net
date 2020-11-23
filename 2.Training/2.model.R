library(mxnet)
library(magrittr)
# Read Pre-training Model

res_model <- mx.model.load("model/resnet-50", 0)
res_sym <- mx.symbol.load("model/resnet-50-symbol.json")

# Get symbol

all_layers = res_sym$get.internals()
flatten0_output = which(all_layers$outputs == 'flatten0_output') %>% all_layers$get.output()
#mx.symbol.infer.shape(flatten0_output, data = c(448, 448, 3, 12))$out.shapes

# Define Model Architecture
fc1 <- mx.symbol.FullyConnected(data = flatten0_output, num_hidden = 6, name = 'fc1')
#mx.symbol.infer.shape(fc1, data = c(448, 448, 3, 12))$out.shapes
#relu1 <- mx.symbol.Activation(data = fc1, act_type = "relu", name = 'relu1')
#mx.symbol.infer.shape(relu1, data = c(448, 448, 3, 12))$out.shapes
#fc2 <- mx.symbol.FullyConnected(data = flatten0_output, num_hidden = 6, name = 'fc2')
#mx.symbol.infer.shape(fc2, data = c(448, 448, 3, 12))$out.shapes
#dp1 <- mx.symbol.Dropout(data = fc1, p = 0.2, name = 'fc1')
sigmoid = mx.symbol.sigmoid(data = fc1, name = 'sigmoid')
#mx.symbol.infer.shape(sigmoid, data = c(448, 448, 3, 4))$out.shapes
#softmax <- mx.symbol.softmax(data = dp1, axis = 1, name = 'softmax')


label = mx.symbol.Variable(name = 'label')

eps = 1e-8
ce_loss_pos =  mx.symbol.broadcast_mul(mx.symbol.log(sigmoid + eps), label)
#mx.symbol.infer.shape(ce_loss_pos, data = c(256, 256, 3, 4), label = c(1,4))$out.shapes
ce_loss_neg =  mx.symbol.broadcast_mul(mx.symbol.log(1 - sigmoid + eps), 1 - label)
#mx.symbol.infer.shape(ce_loss_neg, data = c(256, 256, 3, 4), label = c(1,4))$out.shapes
ce_loss_mean = 0 - mx.symbol.mean(ce_loss_pos + ce_loss_neg)
#mx.symbol.infer.shape(ce_loss_mean, data = c(256, 256, 3, 4), label = c(1,4))$out.shapes
ce_loss = mx.symbol.MakeLoss(ce_loss_mean, name = 'ce_loss')

#mx.symbol.infer.shape(ce_loss, data = c(256, 256, 3, 4), label = c(1,4))$out.shapes


#eps = 1e-8
#m_log = 0 - mx.symbol.mean(mx.symbol.broadcast_mul(mx.symbol.log(softmax + eps), label))
#m_logloss = mx.symbol.MakeLoss(m_log, name = 'm_logloss')