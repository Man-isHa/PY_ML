clear all;
close all;
a=linspace(0,10);
data=sin(a);
%a=[1:size(data,2)];
ans=ones(1,size(data,2));
x=[ans;a;(a.^2)/5;(a.^(3))/50;a.^(1/2);a.^(5)/10000];

th=[0,0,0,0,0,0];
h=th*x;
for k=1:10000
comp=[0,0,0,0,0,0];
for j=1:6
for i=1:size(data,2)
comp(j)=comp(j)+(h(i)-data(i))*(x(j,i));
end
end
al=.01;
for j=1:6
th(j)=th(j)-al*comp(j)/(size(data,2));
end
h=th*x;     


end
plot(data);
hold on;
plot(h);