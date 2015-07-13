function [sw]=svm_train_multi_kernel(data1,data2,MyClassifparams)
idx_sda=MyClassifparams.idx_sda;
gamma=MyClassifparams.gamma;
kfold=5;
traindata=[data1;data2];
traindataHaralick=traindata(:,idx_sda(find(idx_sda<=840)));
traindataLBP=traindata(:,idx_sda(find(idx_sda>840)));
gnd=[ones(size(data1,1),1);-ones(size(data2,1),1)];
MK(:,:,1)=calckernel('rbf',gamma(1),traindataHaralick);
MK(:,:,2)=calckernel('rbf',gamma(2),traindataLBP);
sw.result=jb_2gridSearch(MK,gnd,kfold);
sw.traindataHaralick=traindataHaralick;
sw.traindataLBP=traindataLBP;

















