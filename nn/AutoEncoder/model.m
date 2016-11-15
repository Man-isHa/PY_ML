function out = model(x,hl,hdim)
h = sigmf(((hl.Wih * transpose(x)) + hl.bh) , [1,0]);
out = sigmf(hl.Who * h + hl.bo , [1,0]);
end
