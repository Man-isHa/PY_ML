classdef myClass < handle
	properties
		Wih=rand(50,63), Who=rand(63,50), bo=rand(63,1), bh=rand(50,1);
	end
	methods
		function h = myClass(Wih,bh,Who,bo)
		  h.Wih = Wih;
          h.bh = bh;
          h.Who = Who;
          h.bo = bo;
		end
	end
end
