#de fun

Show_img <- function (img, plot_xlim = c(0.04, 0.96), plot_ylim = c(0.96, 0.04)) {
  
  par(mar = rep(0, 4))
  plot(NA, xlim = plot_xlim, ylim = plot_ylim, xaxt = "n", yaxt = "n", bty = "n")
  img = (img - min(img))/(max(img) - min(img))
  img = as.raster(img)
  rasterImage(img, 0, 1, 1, 0, interpolate=FALSE)
  
}

resize_fun <- function(img, target_size = 512) {
  
  if(dim(img)[1] > dim(img)[2]) {
    
    long_side <- round(dim(img)[1]/(dim(img)[2]/target_size))
    resized <- resize(img, long_side, target_size)
    # Reshape to format needed by mxnet (width, height, channel, num)
    resized <- as.array(resized)
    crop_size <- dim(resized)[1] - target_size
    img <- resized[(1+(crop_size/2)):(dim(resized)[1]-(crop_size/2)),,,,drop = FALSE]
    
  } else {
    
    long_side <- round(dim(img)[2]/(dim(img)[1]/target_size))
    resized <- resize(img, target_size, long_side)
    # Reshape to format needed by mxnet (width, height, channel, num)
    resized <- as.array(resized)
    crop_size <- dim(resized)[2] - target_size
    img <- resized[,(1+(crop_size/2)):(dim(resized)[2]-(crop_size/2)),,,drop = FALSE]
    
    
  }
  
  return(img)
  
}

process_image_fun <- function(data = data, process_file = "image", target_size = 512) {
  
  img_path <- if(process_file == "example") {"example/"} else {"data/image/"}
  img_names <- unique(as.character(data[,"objectUID"]))
  num_img <- length(img_names)
  img_list <- list() 
  t0 <- Sys.time()
  try_error <- NULL
  times <- 1
  
  
  for(i in 1:num_img) {
    
    img <- readImage(paste0(img_path, img_names[i]))
    renames <- substr(img_names[i], start = 1, stop = nchar(img_names[i]) - 4)
    img_array <- array(0, dim = c(dim(img), 3))
    
    if (dim(img)[3] != 3 | length(dim(img)) < 3) {
      for(j in 1:3) {
        img_array[,,j] <- img
      }
    } else (
      img_array <- img
    )
    
    dim(img_array) <- c(dim(img_array), 1)
    img <- img_array
    
    if (is.array(img)) {
    
    bbox_info <- data[data[,"objectUID"] == img_names[i], c("col_left", "col_right", "row_bot", "row_top")]
    if(nrow(bbox_info) == 2) {
      
      crop_col <- round(sum(bbox_info[,"col_left"], bbox_info[,"col_right"])/4*dim(img)[2])
      img_1 <- img[,1:crop_col,,, drop = FALSE]
      img_2 <- img[,(crop_col+1):dim(img)[2],,, drop = FALSE]
      
      resize_img_1 <- resize_fun(img_1, target_size)
      resize_img_2 <- resize_fun(img_2, target_size)
      
      file_path_1 <- "data/temporary/resize_img_1.jpg"
      file_path_2 <- "data/temporary/resize_img_2.jpg"
      writeJPEG(resize_img_1[,,,1], file_path_1)
      writeJPEG(resize_img_2[,,,1], file_path_2)
      
      
      img_list[[times]] <- readBin(con = file_path_1, what = 'raw', n = file.size(file_path_1))
      names(img_list)[times] <- paste0(renames, "01.jpg")
      times = times + 1
      img_list[[times]] <- readBin(con = file_path_2, what = 'raw', n = file.size(file_path_2))
      names(img_list)[times] <- paste0(renames, "02.jpg")
      times = times + 1
      
      
    } else {
      
      resize_img <- resize_fun(img, target_size)
      file_path_1 <- "data/temporary/resize_img_1.jpg"
      writeJPEG(resize_img[,,,1], file_path_1)
      img_list[[times]] <- readBin(con = file_path_1, what = 'raw', n = file.size(file_path_1))
      names(img_list)[times] <- paste0(renames, "01.jpg")
      times = times + 1
      
    }
    
    
    if (i %% 500 == 0) {
      message(paste0('Current process: ', i, '/', num_img, ' Speed: ',
                     formatC(as.numeric(Sys.time() - t0, units = 'secs')/i, format = 'f', 1), 'sec/img\nEstimated time remaining: ',
                     formatC(as.numeric(Sys.time() - t0, units = 'secs')/i*(num_img - i), format = 'f', 1), 'sec'))
    }
    
  } else {
    
    try_error <- c(try_error, i)
    
    }
    
  }
  
  return(img_list)
  
}



