%% This is a demo for the XQDA metric learning, as well as the evaluation on the VIPeR database. You can run this script to reproduce our CVPR 2015 results.
% Note: this demo requires about 1.0-1.4GB of memory.
clc;
clear;
project_dir=pwd;
dataset='GRID';
data_dir=strcat(project_dir,'/data/',dataset,'/');
addpath(data_dir);


load ([data_dir, 'pytorch_result_grid_149.mat']); 
load ([data_dir, 'camIDs.mat']); 

num_gallery = 900;
num_probe =125;
num_test = 125;
num_Class=250;
num_train=125;
numFolds = 10;

ap = zeros(num_test, 1);
cmc = zeros(num_test, num_gallery);
AP2 = zeros(num_test, 1);
precision2 = zeros(num_test, 1);
feature=[feature_g;feature_p];
 feature=feature';
 featprobeA=feature(:,1026:1275);
 featprobeB=feature(:,776:1025);
 featassist=feature(:,1:775);

 camA=camID(:,1026:1275);
 camB=camID(:,776:1025);
 camssist=camID(:,1:775);
 seed = 5;
rng(seed);
    for nf = 1 : numFolds 
         
   p = randperm(num_Class);
   idxTrain_a = p(1:num_train);
   idxTrain_b= idxTrain_a;
   idxProbe=p(num_train+1 : end);
   idxGallery2=p(num_train+1 : end);
   idxassist=1:1:775;
   idxGallery=[idxGallery2 idxassist];
   
   train_a = featprobeA(:,idxTrain_a); 
   test_a = featprobeA(:,idxProbe); 
   train_b = featprobeB(:,idxTrain_b); 
   test_b2 =  featprobeB(:,idxGallery2); 
   test_b=[test_b2 featassist];
   
   queryCAM=camA(:,idxProbe); 
   galleryCAM2=camB(:,idxGallery2);
   galleryCAM=[galleryCAM2 camssist ];
       %% Euclidean
        dist_eu = pdist2(test_b', test_a','euclidean');
        for k=1:num_test
            finalScore = dist_eu(:,k);
            [sortScore sortIndex] = sort(finalScore);
             % find groudtruth index (good and junk)    
             good_index = intersect(find(idxGallery == idxProbe(k)), find(galleryCAM ~= queryCAM(k)))';
             [ap(k),  cmc(k, :),precision(k)] = compute_AP(good_index, sortIndex);
        end
        CMC(nf,:) = mean(cmc); 
        mAP(nf,:) = mean(ap); 
       
    end

meanCMC = mean(CMC,1);
meanmAP = mean(mAP);
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCMC([1,5,10,20,50]) * 100);
fprintf('%5.2f%\n', meanmAP * 100);










