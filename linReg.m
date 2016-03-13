a=[2.6228,2.9125,3.1390,4.2952,4.9918,4.6468,5.4008,6.3853,6.7494,7.3864]
x=[1:10];
th0=0;th1=1;
h=th0+th1.*x;
for i=1:100
comp=0;comp1=0;
for i=1:10
comp=comp+h(i)-a(i);
comp1=comp1+(h(i)-a(i))*x(i);
end
al=0.001;
th0=th0-al*comp/10;
th1=th1-al*comp1/10;
h=th0+th1.*x;
scatter(x,a);
hold on;
plot(h);
end


