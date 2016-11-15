function l = loss(x,hl,hdim)
e = zeros(10,1);
for i = 1:10
    out = model(x(i,:),hl,hdim);
    e(i)= sumsqr(out - transpose(x(i,:)));
end
 l = sum(e)/20 ;
end