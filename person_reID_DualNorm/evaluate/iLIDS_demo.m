%% This is a demo for the XQDA metric learning, as well as the evaluation on the VIPeR database. You can run this script to reproduce our CVPR 2015 results.
% Note: this demo requires about 1.0-1.4GB of memory.
clc;
clear;
project_dir=pwd;
dataset='iLIDS';
data_dir=strcat(project_dir,'/data/',dataset,'/');
addpath(data_dir);

seed = 5;
rng(seed);

numClass = 119;
numFolds = 10;
numRanks = 25;
gsize=60;%num of test people
psize=numClass-gsize;


load([data_dir, 'pytorch_result_ilds_149.mat']);  
load([data_dir, 'labels.mat']);  
load([data_dir, 'camIDs.mat']);

feature=double(feature);
numPerson=length(unique(gID));
[u,v]=size(feature);

galFea=[];
probFea=[];
galID=[];
probID=[];
galcamID=[];
probcamID=[];
ID_TEMP=[];
camID=camID';


 for i=1:numPerson
          a(i,:)= sum(gID(:)==i); 
           st=1+length(ID_TEMP);
           ID_temp=gID(st:st+a(i,:)-1);
           ID_TEMP=[ID_TEMP;ID_temp];
           
           q= randperm(length(ID_temp));
           
          galFea_temp=[feature(st+q(1)-1,:)];
          probFea_temp=[feature(st+q(2)-1,:)];
          galFea=[galFea;galFea_temp];
          probFea=[probFea;probFea_temp];
           
          galID_temp=[gID(st+q(1)-1,:)];
          probID_temp=[gID(st+q(2)-1,:)];
          galID=[galID;galID_temp];
          probID=[probID;probID_temp];
          
          galcamID_temp=[camID(st+q(1)-1,:)];
          probcamID_temp=[camID(st+q(2)-1,:)];
          galcamID=[galcamID;galcamID_temp];
          probcamID=[probcamID;probcamID_temp];
 end

    for nf = 1 : numFolds 
          
        p = randperm(numPerson);     
     
         idxProbe=galID(p(psize+1:end));
         idxGallery=probID(p(psize+1:end));
    
         test_a = probFea(p(psize+1:end),:); 
         test_b = galFea(p(psize+1:end),:);
         queryCAM=probcamID(p(psize+1:end),:); 
         galleryCAM=galcamID(p(psize+1:end),:);
        
        
        dist_eu = pdist2(test_a, test_b,'euclidean');
        for k=1:gsize
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
      
   
    











