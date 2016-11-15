function out = model1(x_train,x,hl,hdim)
ep = 0.01;
h = sigmf(hl.Wih * transpose(x) + hl.bh , [1,0]);
out = sigmf(hl.Who * h + hl.bo , [1,0]);
err = out - transpose(x);
dwho = (err * transpose(h));
dbo =  err ;
dwih = (h.*(1-h).*(transpose(hl.Who) * (out.*(1-out).*err) )) * x ;
dbh = (h.*(1-h).*(transpose(hl.Who) * (out.*(1-out).*err) )) ;
hl.Who = hl.Who - ep* dwho;
hl.Wih = hl.Wih - ep* dwih;
hl.bh = hl.bh - ep* dbh;
hl.bo = hl.bo - ep* dbo;
disp(loss(x_train,hl,hdim));
end