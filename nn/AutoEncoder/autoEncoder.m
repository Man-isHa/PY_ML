
%Input layer has 63 units and to compress the data according to 
%the hidden layer dimension (50 in my case).  
%The total number of weights are 2*63*50 = 6300
%There are 6300 unknowns and just 10 examples hence the model 
%does not get trained well irrespective of the number of the epochs
%Good learning rate for minimum cost can be 0.01 as set
%model1 - for backpropagation and updating weights
%loss   - for calculating loss after every updation
%model  - for forward propagation
%myClass- as a handle for the weights


X = load('alphabet.mat');
hdim = 50;
x_train = zeros(10,63);
Wih=rand(50,63); 
Who=rand(63,50);
bo=rand(63,1);
bh=rand(50,1);
for i = 1:10
    x_train(i,:) = reshape(X.(char(64+i)),1,63);
end
hl = myClass(Wih,bh,Who,bo);
epoch = 0;
while epoch <100
    x_train = x_train(randperm(size(x_train,1)),:);
    for i = 1:10 
    model1(x_train,x_train(i,:),hl,hdim);
    end
    disp(epoch);
    epoch = epoch +1;
end



