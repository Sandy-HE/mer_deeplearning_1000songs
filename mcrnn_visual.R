library(tidyverse)
library(data.table)
library("plotrix")  # for drawing circle

#This code file is for data visualization related to 1000-song dataset.


#The file "mcrnn_eva_10fold.csv" contain 10 fold test results of MCRNN model in terms of rmse and r2.
#The file "dnn_eva_10fold.csv" contain 10 fold test results of DNN model in terms of rmse and r2.
#The file "index.csv" give the all index(705 songs) of 1000 songs which are 44100Hz rather than 48000Hz. 

#Every groundtruth file includes 744 songs, we need to filter them to 705 based on these indexes. 
#Another work is reordering songs by ascii order of file names rahter than integer order.
#For data generated from experimental results(usually use python to write to csv file), we don't 

idx = fread("index.csv")
mcrnn_eva = fread("mcrnn_eva_10fold.csv")
dnn_eva = fread("dnn_eva_10fold.csv")


mcrnn_r2= mcrnn_eva %>% select(a_r2:avg_r2,fold)
mcrnn_rmse= mcrnn_eva %>% select(a_rmse:fold)


#====R2 scrore barplot===========
colnames(mcrnn_r2)[1:3] = c("arousal","valence","average")
mcrnn_r2 = mcrnn_r2 %>%
  gather("type","value",arousal:average)

mcrnn_r2[,2]<-factor(mcrnn_r2[,2],levels=c("arousal","valence","average"),ordered=FALSE)
mcrnn_r2[,1]<-factor(mcrnn_r2[,1],levels=c("F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"),ordered=FALSE)

r2exp=expression(paste(R^2," Score"))
ggplot(mcrnn_r2, aes(x=fold,y=value, fill=type,group=type))+
  geom_bar(stat = "identity",position = position_dodge(0.8))+
  coord_cartesian(ylim = c(-0.15, 0.75))+
  xlab("Fold")+
  ylab(r2exp)+
  scale_fill_discrete(name="MCRNN")+
  theme_bw()+
  theme(legend.position=c(0.9, 0.88))


#====RMSE barplot=============
colnames(mcrnn_rmse)[1:3] = c("arousal","valence","average")
mcrnn_rmse = mcrnn_rmse %>%
  gather("type","value",arousal:average)

mcrnn_rmse[,2]<-factor(mcrnn_rmse[,2],levels=c("arousal","valence","average"),ordered=FALSE)
mcrnn_rmse[,1]<-factor(mcrnn_rmse[,1],levels=c("F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"),ordered=FALSE)

ggplot(mcrnn_rmse, aes(x=fold,y=value, fill=type, group=type))+
  geom_bar(stat = "identity",position = "dodge")+
  coord_cartesian(ylim = c(0, 0.4))+
  xlab("Fold")+
  ylab("RMSE")+
  scale_fill_discrete(name="DNN")+
  theme_bw()+
  theme(legend.position=c(0.9, 0.88))


#====The distribution of static emotion for all songs or fold-level songs====

songsav = fread("./1000songs_annotations/static_annotations.csv")
songsav = songsav[songsav$song_id %in% idx$index,]   #filter 744 to 705.

songsav$mean_arousal = (songsav$mean_arousal-1)/8*2-1
songsav$mean_valence = (songsav$mean_valence-1)/8*2-1
songsav = songsav %>% arrange(as.character(song_id))  

#check sample count in each quadrant
temp = songsav[songsav$mean_arousal>0 & songsav$mean_valence>0,]
#plot(songsav[,c("mean_valence","mean_arousal")],kind = 0)


plot.new()
#plot.window(xlim=c(-2,2), ylim=c(-2,2))
plot(c(-1.1,1.1), c(-1.1,1.1), type='n', asp=1,main = "Emotion Dimensional Model")
draw.circle(0, 0, 1, nv = 1000, border = NULL, col = NA, lty = 1, lwd = 1)
arrows(c(-1.1,0),c(0,-1.1),c(1.1,0),c(0,1.1), length=0.1)
text(x=0.22,y=1.1, "Arousal", font=2)
text(x=1.2,y=0.1, "Valence", font=2)
text(x=1.15,y=-0.05, "positive" , cex=.7, color="grey", font=3)
text(x=-1.15,y=-0.05, "negative" , cex=.7, color="grey", font=3)
text(x=-0.15,y=1.05, "active" , cex=.7, color="grey", font=3)
text(x=-0.15,y=-1.05, "inactive" , cex=.7, color="grey", font=3)

#visualize for one cross-validation fold or all songs
#points(songsav$mean_valence,songsav$mean_arousal,pch=16)
points(songsav$mean_valence[631:700],songsav$mean_arousal[631:700],pch=16)
points(songsav$mean_valence[1:70],songsav$mean_arousal[1:70],pch=20, col='red')
points(songsav$mean_valence[701:705],songsav$mean_arousal[701:705],pch=20, col=alpha('grey',0.8))
points(songsav$mean_valence[71:630],songsav$mean_arousal[71:630],pch=20, col=alpha('grey',0.8))

#====visualize standard deviation===========
a_std = fread("./1000songs_annotations/arousal_cont_std.csv")
a_std = a_std[a_std$song_id %in% idx$index,]
a_std = a_std %>% arrange(as.character(song_id))

a_std$idx = 1:nrow(a_std)
a_std$fold = a_std$idx %/% 70 +1

a_std1= a_std %>% select(-sample_15000ms)

a_std1 = a_std1 %>% mutate(st_mean = rowMeans(select(a_std1,starts_with("sample_"))),st_std = apply(select(a_std1,starts_with("sample_")),1,sd))
a_std2 = group_by(a_std1, fold)
a_std3 = summarise(a_std2, st_fold_mean = mean(st_mean),st_fold_st = sd(st_std))

#=====fold-level time-series visual=========

library(ggplot2)
avtrue = fread("./av_true_f3.csv")
avbase = fread("./av_base_f3.csv")
avpred = fread("./av_pred_f3.csv")
avtrue$idx =1:nrow(avtrue)
avtrue$id = (avtrue$idx-1) %/% 60 +1
avtrue$ts = 0.5*(avtrue$idx - 60*(avtrue$id - 1))
colnames(avbase) = c("abase","vbase")

avall = cbind(avtrue, avbase, avpred)

colnames(avall)[c(1,2,6:9)]<- c("A-Truth","V-Truth","A-Baseline","V-Baseline","A-MCRNN","V-MCRNN")
avall_new = avall %>%
  gather(key = "type", value="value", 'A-Truth':'V-Truth', 'A-Baseline':'V-MCRNN')

#visline = c("dashed")
#viscolor = 
#avtrue %>%

#for 70 songs
ggplot(avall_new, aes(x=ts,y=value, color=type ))+
  geom_line()+
  theme_bw()+
  xlab("Time Steps")+
  ylab("Arousal(A) / Valence(V)")+
  facet_wrap(~id)

#for one song
av4one=avall_new[avall_new$id==3,]
color=c("green","red","blue","green","red","blue")
linetype=c("longdash","longdash","longdash","solid","solid","solid")
sca = 0.5 

ggplot(av4one, aes(x=ts,y=value,group=type))+
  geom_line(aes(col=type,linetype=type),size=1)+
  theme_bw()+
  xlab("Time Steps")+
  ylab("Arousal(A) / Valence(V)")+
  scale_linetype_manual(values = linetype) +
  scale_color_manual(values = color)+
  theme(legend.position = "none",
        axis.title.y=element_text(size=15),
        axis.title.x=element_text(size=15),
        axis.text=element_text(size=15))+
  coord_cartesian(ylim = c(-sca, sca))

#add legend (980*350)
ggplot(av4one, aes(x=ts,y=value,group=type))+
  geom_line(aes(col=type,linetype=type),size=1)+
  theme_bw()+
  xlab("Time Steps")+
  ylab("Arousal(A) / Valence(V)")+
  scale_linetype_manual(values = linetype) +
  scale_color_manual(values = color)+
  theme(legend.title = element_blank(),
        legend.key.width=unit(1, "cm"),
        legend.text=element_text(size=20),
        axis.title.y=element_text(size=15),
        axis.title.x=element_text(size=15),
        axis.text=element_text(size=15))+
  #guides(shape = guide_legend(override.aes = list(size=50)))+
  coord_cartesian(ylim = c(-sca, sca))
  

#====dynamic anno for one or more songs======
#avtrue = fread("./av_true.csv")
#avpred = fread("./av_pred.csv")
library(ggplot2)
avtrue = fread("./av_true_f5.csv")
avbase = fread("./av_base_f5.csv")
avpred = fread("./av_pred_f5.csv")
avtrue$idx =1:nrow(avtrue)
avtrue$id = (avtrue$idx-1) %/% 60 +1
avtrue$ts = 0.5*(avtrue$idx - 60*(avtrue$id - 1))
colnames(avbase) = c("abase","vbase")

avall = cbind(avtrue, avbase, avpred)

sca = 0.5
sca1=0.54
pstart = 1021
pend = 1080

#only figure(800*550), figure with legend(920*550)
plot.new()
par(xpd = T, mar = par()$mar + c(0,0,0,7))
#plot.window(xlim=c(-2,2), ylim=c(-2,2))asp=1,
plot(c(-sca,sca), c(-sca,sca), type='n',xlab = "Valence", ylab = "Arousal", cex.lab=1.5, cex.axis=1.5)
#draw.circle(0, 0, 1, nv = 1000, border = NULL, col = NA, lty = 1, lwd = 1)
#arrows(c(-sca1,0),c(0,-sca1),c(sca1,0),c(0,sca1), length=0.1)
lines(c(-sca1,sca1),c(0,0), type='l')
lines(c(0,0),c(-sca1,sca1), type='l')
points(avall$vtrue[pstart:pend] ,avall$atrue[pstart:pend],pch=16, col='blue')
points(avall$vpred[pstart:pend] ,avall$apred[pstart:pend],pch=16, col=alpha('red',0.7))
points(avall$vbase[pstart:pend] ,avall$abase[pstart:pend],pch=16, col=alpha('green',0.7))
#arrows(avtrue$atrue[pstart:pend] ,avtrue$vtrue[pstart:pend], avpred$apred[pstart:pend] ,avpred$vpred[pstart:pend],length=0.1, col="red")   

legend(0.55,0.2, legend=c("Baseline","MCRNN","Truth"), col=c("green","red","blue"),
      pch = c(16,16,16),bty='n',cex=1.5)
par(mar=c(5, 4, 4, 2) + 0.1)
       #, lty=1:2, cex=0.8)
#====all songs distribution=====
atrue_all = fread("./1000songs_annotations/arousal_cont_average.csv")
atrue_all = atrue_all[atrue_all$song_id %in% idx$index,]
atrue_all = atrue_all %>% arrange(as.character(song_id))
vtrue_all = fread("./1000songs_annotations/valence_cont_average.csv")
vtrue_all = vtrue_all[vtrue_all$song_id %in% idx$index,]
vtrue_all = vtrue_all %>% arrange(as.character(song_id))

atrue_all_new = atrue_all %>%
  select(-sample_15000ms)%>%
  gather(ts, avalue, sample_15500ms:sample_45000ms)
vtrue_all_new = vtrue_all %>%
  select(-sample_15000ms)%>%
  gather(ts, vvalue, sample_15500ms:sample_45000ms)

avtrue_all = cbind(atrue_all_new,vtrue_all_new$vvalue)
colnames(avtrue_all)[4] = "vvalue"

sca = 1.1
plot.new() 

plot(c(-sca,sca), c(-sca,sca),,type='n',asp=1,xlab = "", ylab = "")
#plot.window()
draw.circle(0, 0, 1, nv = 1000, border = NULL, col = NA, lty = 1, lwd = 1)
arrows(c(-sca,0),c(0,-sca),c(sca,0),c(0,sca), length=0.1)

points(avtrue_all$vvalue ,avtrue_all$avalue,pch=16, col=alpha('grey',0.5))
text(x=0.22,y=1.1, "Arousal", font=2)
text(x=1.2,y=0.1, "Valence", font=2)
text(x=1.15,y=-0.05, "positive" , cex=.7,  font=3)
text(x=-1.15,y=-0.05, "negative" , cex=.7,  font=3)
text(x=-0.15,y=1.05, "active" , cex=.7,  font=3)
text(x=-0.15,y=-1.05, "inactive" , cex=.7,  font=3)

#====The average of A-V of 60 annotation======
fold = av_stat_f7
avtrue = fread("./av_true_f7.csv")
avtrue$idx =1: nrow(avtrue)
avtrue$id = (avtrue$idx-1) %/% 60 +1

avtrue_stat = avtrue %>%
  select(-idx) %>%
  group_by(id) %>%
  summarise(mean_arousal=mean(atrue), mean_valence=mean(vtrue))

avpred = fread("./av_pred_f7.csv")
avpred$idx =1: nrow(avpred)
avpred$id = (avpred$idx-1) %/% 60 +1

avpred_stat = avpred %>%
  select(-idx) %>%
  group_by(id) %>%
  summarise(mean_arousal_pred=mean(apred), mean_valence_pred=mean(vpred))

fold = cbind(avpred_stat,avtrue_stat)
fwrite(fold, "./av_stat_f7.csv")
sca = 0.7

plot.new()
#plot.window(xlim=c(-2,2), ylim=c(-2,2))asp=1
plot(c(-sca,sca), c(-sca,sca), type='n',main = "Emotion Dimensional Model", xlab = "Valence", ylab = "Arousal")
#draw.circle(0, 0, 1, nv = 1000, border = NULL, col = NA, lty = 1, lwd = 1)
arrows(c(-1.1,0),c(0,-1.1),c(1.1,0),c(0,1.1), length=0.1)
points(fold$mean_arousal ,fold$mean_valence,pch=1,col='blue')
points(fold$mean_arousal_pred ,fold$mean_valence_pred,pch=19, col='red')
arrows(fold$mean_arousal_pred ,fold$mean_valence_pred,fold$mean_arousal ,fold$mean_valence, length=0.1, col="grey")   

