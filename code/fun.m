function error = fun(x,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
%�ú�������������Ӧ��ֵ
%x          input     ����
%inputnum   input     �����ڵ���
%outputnum  input     ������ڵ���
%net        input     ����
%ginputn     input     ѵ����������
%goutput_train    input     ѵ���������

%error      output    ������Ӧ��ֵ

%��ȡ
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
B2=B2';

%�����������
net.trainParam.epochs=20;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;
net.trainParam.show=100;
net.trainParam.showWindow=0;
 
%����Ȩֵ��ֵ
net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%����ѵ��
net=train(net,ginputn,goutput_train);

an=sim(net,ginputn);

%�����������һ��
BPout=mapminmax('reverse',an,outputps);

BPout=abs(BPout);
abs_error=abs(BPout-goutput_train);
[mmm,nnn]=size(goutput_train);
error_1=[1,nnn];
for q=1:nnn
     if (( abs_error(1,q) < abs_error(2,q) && abs_error(1,q) < abs_error(3,q) && abs_error(1,q) < abs_error(4,q) && goutput_train(1,q) == 1)||( abs_error(2,q) < abs_error(1,q) && abs_error(2,q) < abs_error(3,q) && abs_error(2,q) < abs_error(4,q) && goutput_train(2,q) == 1) || ( abs_error(3,q) < abs_error(1,q) && abs_error(3,q) < abs_error(2,q) && abs_error(3,q) < abs_error(4,q) && goutput_train(3,q) == 1) || (abs_error(4,q) < abs_error(1,q) && abs_error(4,q) < abs_error(2,q) && abs_error(4,q) < abs_error(3,q) && goutput_train(4,q) == 1)) 
         error_1(1,q)=0.0001;
     else
         error_1(1,q)=10;
     end
end
error=sum(error_1);





% error=sum(abs(an-goutput_train));

% ��ά�������ҳ������С����һά������ͳ�ʼ�ı�ǩһ�£�������=0.000001��
% error=sum(abs(an-goutput_train));