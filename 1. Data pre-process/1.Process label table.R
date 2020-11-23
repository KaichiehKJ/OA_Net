data <- read.csv("data/Osteoarthritis.csv", header = TRUE, sep = ",", fileEncoding = "CP950")
data <- data[data[,"del_Username"] != "orth2301" & data[,"del_Username"] != "ji9su",]
head(data)
data[,"objectUID"] <- as.character(data[,"objectUID"])
data[2723:2724, "objectUID"] <- c("104040101635346.jpg", "104040101635346.jpg")

# process label

labels <- c("KL=0", "KL=1", "KL=2", "KL=3", "KL=4", "TKR",
          "Joint space of medial condyle", "Joint space of lateral condyle",
          "Spur formation (medial)", "Spur formation (lateral)",
          "Bony defect (femoral condyle flattening)", "Bony defect (tibial plateau bony defect)",
          "Subchondral cyst", "Sclerotic change", "Valgus/ varus knee",
          "Tiabial bowing", "Femoral bowing", "X-ray quality",
          "Intra-articular calcification", "Implant", "Polio")

OA_data <- as.data.frame(matrix(0, nrow = 8241, ncol = 22))
colnames(OA_data) <- c("NO", labels)
OA_data[,1] <- data[,"objectUID"]

for(i in 1:nrow(OA_data)) {
  
  for(j in 1:length(labels)) {
    if(grepl(labels[j], data[i, 3], fixed = TRUE)) {
      
      OA_data[i, labels[j]] <- 1
      
    }
  }
}

img_names <- unique(OA_data[,1])

for(i in 1:length(img_names)) {
  
  if(length(data[data[,"objectUID"] == img_names[i], "col_left"]) == 2) {
    
    if (data[data[,"objectUID"] == img_names[i], "col_left"][1] < data[data[,"objectUID"] == img_names[i], "col_left"][2]) {
      
      renames <- substr(img_names[i], start = 1, stop = nchar(img_names[i]) - 4)
      OA_data[OA_data[,"NO"] == img_names[i], "names"] <- c(paste0(renames, "01.jpg"), paste0(renames, "02.jpg"))
      
    } else {
      
      renames <- substr(img_names[i], start = 1, stop = nchar(img_names[i]) - 4)
      OA_data[OA_data[,"NO"] == img_names[i], "names"] <- c(paste0(renames, "02.jpg"), paste0(renames, "01.jpg"))
      
    }
    
  } else {
    
    renames <- substr(img_names[i], start = 1, stop = nchar(img_names[i]) - 4)
    OA_data[OA_data[,"NO"] == img_names[i], "names"] <- c(paste0(renames, "01.jpg"))
    
  }

}


write.csv(OA_data, "data/OA_data.csv")
write.csv(data, "data/data.csv")






#label <- read.csv("data/label.csv")
#for(i in 1:8239) {
  
#  OA_data[grepl(substr(as.character(label[i,"img_name"]), start = 1, stop = nchar(as.character(label[i,"img_name"])) - 4) ,OA_data[,"names"]), "group"] <- label[i, "group"]
  
#}

#OA_data[c(6571, 7686), "group"] = 1
#label_table <- OA_data[,c("names", "group")]
#label_table[,3:23] <- OA_data[,3:23]
#write.csv(label_table, "data/label.csv", row.names=F)


