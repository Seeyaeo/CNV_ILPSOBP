%%����������
clc
clear

for l=38:40

data_trains1=load('data\0.2_4x_11.1\sim1_4_4100_trains.txt');
data_trains2=load('data\0.3_4x_11.1\sim1_4_4100_trains.txt');
data_trains3=load('data\0.4_4x_11.1\sim1_4_4100_trains.txt');
data_trains4=load('data\0.2_6x_11.1\sim1_6_6100_trains.txt');
data_trains5=load('data\0.3_6x_11.1\sim1_6_6100_trains.txt');
data_trains6=load('data\0.4_6x_11.1\sim1_6_6100_trains.txt');

data_trains=[data_trains1;data_trains2;data_trains3;data_trains4;data_trains5;data_trains6];
column=[2,3,4,5,6];
[m1,n1] = size(data_trains);

trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);

%��1��trainLines���������
k=rand(1,trainLines);
[m,n]=sort(k);
%�õ������������
ginput=gdata(:,column);
% goutput1 =gdata(:,6);
goutput1 =gdata(:,7);
%�����һά�����ά
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

%�ҳ�ѵ�����ݺ�Ԥ������
ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';

%ѡ����������������ݹ�һ��
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);

%%��������
load('-mat','PSOBP');

%column=[2,3,4,5];
num=1;
TP_count_sum=0;
TPFP_count_sum=0;

data_tests=load(['data\na192',num2str(l),'_11.1.txt']);

[m2,n2] = size(data_tests);
testLines=m2;
gdata2(1:testLines,:) = data_tests(1:m2,:);
ginput2_bin=gdata2(:,1);
ginput2=gdata2(:,column);
%goutput1 =gdata2(:,6);
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
%% BP����Ԥ��
%Ԥ�����ݹ�һ��
inputn_test=mapminmax('apply',ginput_test,ginputps);

%����Ԥ�����
an=sim(net,inputn_test);

%�����������һ��
BPoutput=mapminmax('reverse',an,outputps);

%Ԥ�����
error=BPoutput-goutput_test;
abs_error=abs(error);

%% �������
%fid=fopen(['NA192',num2str(l),'bin.txt'],'wt'); %��bin_number����д���ļ�
fid=fopen(['data\','NA192',num2str(l),'bin.txt'],'wt'); %��bin_number����д���ļ�
TP_count=0;
P_count=0;
TPFP_count=0;
%k=1;
for q=1:testLines
    if ( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));
        fprintf(fid,'gain\t1');
        fprintf(fid,'\n');
        %binnumber(k)=ginput2_bin(q);
        %k=k+1;
    end
    if ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));%��bin_numberд���ļ�
        fprintf(fid,'hemi_loss\t2');
        fprintf(fid,'\n');
        %binnumber(k)=ginput2_bin(q);
        %k=k+1;
    end
    if ( abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1)
        fprintf(fid,'%d\t',ginput2_bin(q));%��bin_numberд���ļ�
        fprintf(fid,'homo_loss\t3');
        fprintf(fid,'\n');
        %binnumber(k)=ginput2_bin(q);
        %k=k+1;

    end
    if (( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_test(2,q) == 1) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_test(3,q) == 1) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_test(4,q) == 1))     
        TP_count = TP_count + 1;
    end
    if ( goutput_test(2,q) == 1 || goutput_test(3,q) == 1 || goutput_test(4,q) == 1 )
        P_count = P_count + 1;
    end
    if ( (abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) ) )
        TPFP_count = TPFP_count + 1;
    end
end
fclose(fid);

% TPFP_count_sum = TPFP_count_sum + TPFP_count;
% TP_count_sum = TP_count_sum + TP_count;

% TP_count_avg = TP_count_sum/num;
% TPFP_count_avg = TPFP_count_sum/num;
% sensitivity = TP_count_avg/P_count;
% precision = TP_count_avg/TPFP_count_avg;
sensitivity = TP_count/P_count;
precision = TP_count/TPFP_count;
F1_score = (2 * sensitivity * precision)/(sensitivity + precision);
disp(['NA192',num2str(l),'-sensitivity:']);
disp(sensitivity);
disp(['NA192',num2str(l),'-precision:']);
disp(precision);
disp(['NA192',num2str(l),'-F1-score:']);
disp(F1_score);
end
