%% This is a demo for the XQDA metric learning, as well as the evaluation on the VIPeR database. You can run this script to reproduce our CVPR 2015 results.
% Note: this demo requires about 1.0-1.4GB of memory.
clc;
clear;
project_dir=pwd;
dataset='PRID2011';
data_dir=strcat(project_dir,'/data/',dataset,'/');
addpath(data_dir);
seed = 10;
rng(seed);

load ([data_dir, 'pytorch_result_prid_149.mat']); 
load ([data_dir, 'camIDs.mat']);
load ([data_dir, 'labels_py.mat']);

 
num_gallery = 649;
num_test = 100;
numFolds = 10;
num_common=200;
ap = zeros(num_test, 1);
cmc = zeros(num_test, num_gallery);

camA=camID(1:385,:)';
camB = camID(385 + 1 : end, :)';
    for nf = 1 : numFolds 
         p=randperm(num_common);
         idxProbe=gID_a(p(1:num_test));
         idxGallery0 = gID_b(p(1:num_test));
         idxGallery1 =gID_b(201:end);
         idxGallery=[idxGallery0;idxGallery1];
   
        
        
        test_a = feature_a(idxProbe,:);
        test_b = feature_b(idxGallery,:);
   
        queryCAM=camA(:,idxProbe); 
        galleryCAM=camB(:,idxGallery);
      
       %% Euclidean
        dist_eu = pdist2(test_b, test_a,'euclidean');
         for k=1:num_test
            finalScore = dist_eu(:,k);
            [sortScore sortIndex] = sort(finalScore);
             % find groudtruth index (good and junk)
            good_index = intersect(find(idxGallery == idxProbe(k)), find(galleryCAM ~= queryCAM(k)))'; 
            [ap(k),  cmc(k, :)] = compute_AP(good_index, sortIndex);% see compute_AP
        end
        CMC(nf,:) = mean(cmc);
        mAP(nf,:) = mean(ap);
       
    end
meanCMC = mean(CMC,1);
meanmAP = mean(mAP);    

fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCMC([1,5,10,20,50]) * 100);
fprintf('%5.2f%\n', meanmAP * 100);















