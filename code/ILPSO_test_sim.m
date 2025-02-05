%% Clear environment variables
clc
clear

%% Import the parameters of the trained neural network
data_truth1=load('groundtruth_11_1.mat');   
%% Import groundtruth file
data_truth=data_truth1.('data_truth');

%% Obtain normalized information in training samples
data_trains1=load('data\0.2_4x_11.1\sim1_4_4100_trains.txt');
data_trains2=load('data\0.3_4x_11.1\sim1_4_4100_trains.txt');
data_trains3=load('data\0.4_4x_11.1\sim1_4_4100_trains.txt');
data_trains4=load('data\0.2_6x_11.1\sim1_6_6100_trains.txt');
data_trains5=load('data\0.3_6x_11.1\sim1_6_6100_trains.txt');
data_trains6=load('data\0.4_6x_11.1\sim1_6_6100_trains.txt');

data_trains=[data_trains1;data_trains2;data_trains3;data_trains4;data_trains5;data_trains6];
column=[2,3,4,5,6];
[m1,n1] = size(data_trains);
[m3,n3]=size(data_truth);

truthLines=m3;
trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);
gdatatruth(1:truthLines,:)=data_truth(1:truthLines,:);
column3=[1,2];
groundtruth=data_truth(:,column3); 
gtRev=fliplr(groundtruth(:,2)'); 

%Random sorting
k=rand(1,trainLines);
[m,n]=sort(k);

%Get input and output data
ginput=gdata(:,column);
goutput1 =gdata(:,7);

goutput=zeros(trainLines,4);
for i=1:trainLines
    switch goutput1(i)
        case 0
            goutput(i,:)=[1 0 0 0];
        case 1
            goutput(i,:)=[0 1 0 0];
        case 2
            goutput(i,:)=[0 0 1 0];
	case 3
            goutput(i,:)=[0 0 0 1];
    end
end

%Find the training data and prediction data
ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';

%Normalization
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);

load('-mat','PSOBP');

%% 3.Load test data and count results
num=50; %Each folder contains 50 files
TP_count_sum=0;
TPFP_count_sum=0;
bound = 0;
count_bias_sum=0;
boundary=[]; 
num_boundary=0;
flag=1;
normal_sum=zeros(6,50);
gain_sum=zeros(6,50);
hemi_sum=zeros(6,50);
homo_sum=zeros(6,50);

% mkdir('data\','0.2_4x_binnumber');
% mkdir('data\','0.2_4x_allbinnumber');
mkdir('data\','0.2_6x_binnumber');
mkdir('data\','0.2_6x_allbinnumber');
% mkdir('data\','0.3_4x_binnumber');
% mkdir('data\','0.3_4x_allbinnumber');
% mkdir('data\','0.3_6x_binnumber');
% mkdir('data\','0.3_6x_allbinnumber');
% mkdir('data\','0.4_4x_binnumber');
% mkdir('data\','0.4_4x_allbinnumber');
% mkdir('data\','0.4_6x_binnumber');
% mkdir('data\','0.4_6x_allbinnumber');

%Importing test sample data
for t=1:num
    data_tests=load(['C:\Users\Zmx\Desktop\IPSOCNV_Q\data\0.2_6x_11.1\sim', num2str(t) ,'_6_6100_trains.txt']);
    [m2,n2] = size(data_tests);
    testLines=m2;
    gdata2(1:testLines,:) = data_tests(1:m2,:);
    ginput2_bin=gdata2(:,1);
    ginput2=gdata2(:,column);
    goutput1 =gdata2(:,7);
    goutput2=zeros(testLines,4);
    for i=1:testLines
        switch goutput1(i)
            case 0
                goutput2(i,:)=[1 0 0 0];
            case 1
                goutput2(i,:)=[0 1 0 0];
            case 2
                goutput2(i,:)=[0 0 1 0];
            case 3
                goutput2(i,:)=[0 0 0 1];
        end
     end
    ginput_test=ginput2((1:testLines),:)';
    goutput_test=goutput2((1:testLines),:)';

    %Obtain normalized test samples
    inputn_test=mapminmax('apply',ginput_test,ginputps);
    %inverse normalization
    an=sim(net,inputn_test);
    BPoutput=mapminmax('reverse',an,outputps);
    %Statistical prediction error
    error=BPoutput-goutput_test;
    abs_error=abs(error);
    errorsum=sum(abs(error));

    fid=fopen(['data\0.2_6x_binnumber\sim', num2str(t) ,'_6_6100_bin_number.txt'],'wt'); 

    TP_count = 0;
    P_count = 0;
    TPFP_count = 0;
    k = 1;
    kkk = 1;
            
    for qq = 1:testLines
        if ( error(2,qq) < error(1,qq) && error(2,qq) < error(3,qq) && error(2,qq) < error(4,qq) && goutput_test(2,qq) == 1)
            fprintf(fid,'%d\t',ginput2_bin(qq));
            fprintf(fid,'gain\t1');
            fprintf(fid,'\n');
            continue;
        end
        if ( error(3,qq) < error(1,qq) && error(3,qq) < error(2,qq) && error(3,qq) < error(4,qq) && goutput_test(3,qq) == 1)
            fprintf(fid,'%d\t',ginput2_bin(qq));
            fprintf(fid,'hemi_loss\t2');
            fprintf(fid,'\n');
            continue;
        end
        if ( error(4,qq) < error(1,qq) && error(4,qq) < error(2,qq) && error(4,qq) < error(3,qq) && goutput_test(4,qq) == 1)
            fprintf(fid,'%d\t',ginput2_bin(qq));
            fprintf(fid,'homo_loss\t3');
            fprintf(fid,'\n');
            continue;
        end
   end
            
      fidall=fopen(['data\0.2_6x_allbinnumber\sim', num2str(t) ,'_6_6100_allbinnumber.txt'],'wt'); 
      u=1;
      for p=1:testLines
          if ( error(2,p) < error(1,p) && error(2,p) < error(3,p) && error(2,p) < error(4,p))
              fprintf(fidall,'%d\t',ginput2_bin(p));
              fprintf(fidall,'gain\t1');
              fprintf(fidall,'\n');
              continue;
          end
          if ( error(3,p) < error(1,p) && error(3,p) < error(2,p) && error(3,p) < error(4,p))
              fprintf(fidall,'%d\t',ginput2_bin(p));
              fprintf(fidall,'hemi_loss\t2');
              fprintf(fidall,'\n');
              continue;
          end
          if ( error(4,p) < error(1,p) && error(4,p) < error(2,p) && error(4,p) < error(3,p))
              fprintf(fidall,'%d\t',ginput2_bin(p));
              fprintf(fidall,'homo_loss\t3');
              fprintf(fidall,'\n');
              continue;
          end
      end    
      fclose(fidall);
      
      % calculate TP_count,P_count,TPFP_count,boundary
      binnumber=[];
      binnumber_tpfp=[];
            
      for q=1:testLines
          if (( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)) 
              TP_count = TP_count+1;
              binnumber(k) = ginput2_bin(q);
              k=k+1;
          end
          if ( goutput_test(2,q) == 1 || goutput_test(3,q) == 1 || goutput_test(4,q) == 1 )
              P_count = P_count+1;
              binnumber_tpfp(kkk) = ginput2_bin(q);
              kkk = kkk + 1;
          end
          if ((abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) ) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q)) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q)) )
              TPFP_count = TPFP_count + 1;
          end
      end
      fclose(fid);

      TPFP_count_sum = TPFP_count_sum + TPFP_count;
      TP_count_sum = TP_count_sum + TP_count;
      boundbias = (P_count - TP_count)./14;
      bound = bound + boundbias;
   
      binnumberRev = fliplr(binnumber);
      [m3,n3] = size(binnumber);
      jj = 1;
      bin1 = [];
      for ii = 1:n3
          if binnumber(ii) >= groundtruth(jj,1)
              bin1(jj) = binnumber(ii);
              jj = jj + 1;
          end
          if jj > 14
              break; 
          end
      end
      jjj = 1;
      bin22 = [];
      for iii = 1:n3
          if binnumberRev(iii) <= gtRev(jjj)
              bin22(jjj) = binnumberRev(iii);
              jjj = jjj + 1;
          end
          if jjj > 14
              break; 
          end
      end
      bin2 = fliplr(bin22);
      [m5,n5] = size(bin1);
      [m6,n6] = size(bin2);
      if(n5 < 14 || n6 < 14)
          continue; 
      end
      bin = [bin1;bin2]'; 
      [m4,n4] = size(bin);
      c_bias1 = [];
      c_bias2 = [];
      for rr = 1:m4
          c_bias1(rr) = bin(rr,1) - groundtruth(rr,1);
          c_bias2(rr) = groundtruth(rr,2) - bin(rr,2);
      end
      c_bias = sum(c_bias1) + sum(c_bias2);
      count_bias = c_bias./14;
      if count_bias > 50
          continue;
      end
      boundary(t)=count_bias;
      num_boundary = num_boundary + 1;
      count_bias_sum = count_bias_sum + count_bias;
      binnumber_fp = setdiff(binnumber_tpfp,binnumber);
end


TP_count_avg = TP_count_sum / num;
TPFP_count_avg = TPFP_count_sum / num;
%Calculating the sensitivity
sensitivity = TP_count_avg / P_count;
%Calculating the precision
precision = TP_count_avg / TPFP_count_avg;
%Calculating theF1-score
F1_score = (2 * sensitivity * precision)/(sensitivity + precision);
boundary_bias = bound / num;
count_bias_avg = count_bias_sum / num_boundary;
TPR = sensitivity;
FPTN_count = m2 - P_count;
FP_count = TPFP_count - TP_count;
FPR=FP_count/FPTN_count; 
        
disp('sensitivity:');
disp(sensitivity);
disp('precision:');
disp(precision);
disp('F1-score:');
disp(F1_score);
disp('all_bound_bias:');
disp(boundary_bias);
disp('14_bound_bias:');
disp(count_bias_avg);
disp('bound_bias:');
disp(boundary);
disp('TPR:');
disp(TPR);
disp('FPR:');
disp(FPR);
