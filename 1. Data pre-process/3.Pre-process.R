source('~/OA/code/1. Data pre-process/2.Image process.R')

# Libraries

library(jpeg)
library(data.table)
library(magrittr)
library(imager)
library(OpenImageR)

# Image_path
data_path <- 'data/data.csv'
OA_data_path <- 'data/OA_data.csv'

img_info_path <- 'img_info.RData'

# Read data

data <- fread(data_path, data.table = FALSE, stringsAsFactors = FALSE, na = '')
OA_data <- fread(OA_data_path, data.table = FALSE, stringsAsFactors = FALSE, na = '')


#img_preprocess
#process_file:"image" & "example"

img_list <- process_image_fun(data = data, process_file = "image", target_size = 800)

save(img_list, data, OA_data, file = img_info_path)
