%% This is a demo for the XQDA metric learning, as well as the evaluation on the VIPeR database. You can run this script to reproduce our CVPR 2015 results.
% Note: this demo requires about 1.0-1.4GB of memory.
clc;
clear;
project_dir=pwd;
dataset='VIPeR';
data_dir=strcat(project_dir,'/data/',dataset,'/');
addpath(data_dir);


load ([data_dir, 'pytorch_result_viper_49.mat']); 
load ([data_dir, 'camIDs.mat']); 
 
num_gallery = 316;
num_test = 316;
numFolds = 10;
num_Class=632;

ap = zeros(num_gallery, 1);
cmc = zeros(num_gallery, num_gallery);
% 
% feature_a=feature(1:num_Class,:);
% feature_b=feature(num_Class+1:end,:);
% % 
feature_a=feature_a'; 
feature_b=feature_b'; 
camA=camID(1:num_Class,:)';
camB = camID(num_Class + 1 : end, :)';
seed = 25;
rng(seed);
 for nf = 1 : numFolds 
          p=randperm(num_Class);
          idxProbe=p(num_test+1:end);
          idxGallery=p(num_test+1:end);
    
         test_a = feature_a(:,idxProbe); 
         test_b = feature_b(:,idxGallery);
         queryCAM=camA(:,idxProbe); 
         galleryCAM=camB(:,idxGallery);
       %% Euclidean
        dist_eu = pdist2(test_a', test_b','euclidean');
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
        
      

    











