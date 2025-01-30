function error = fun(x,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
%x          input 
%inputnum   input     
%outputnum  input   
%net        input    
%ginputn     input     
%goutput_train    input     
%error      output    

w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
B2=B2';

net.trainParam.epochs=20;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;
net.trainParam.show=100;
net.trainParam.showWindow=0;
 
net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

net=train(net,ginputn,goutput_train);

an=sim(net,ginputn);

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