# include <iostream>
# include <stdlib.h>    
# include <vector>
# include <random>
# include <algorithm>
using namespace std;

class RBM
{
	
	public:
		int i;
		int num_visible;
		int num_hidden;
		double learning_rate,error;
		std::vector<std::vector<double> > weights,update,new_data,hidden_probs,hidden_states,n_d,wts; 
		std::vector<std::vector<double> > pos_associations, r_hidden_probs, r_data, rt_data, neg_associations;
		
		void init_weights(void)
		{
			weights.resize(num_visible+1, std::vector<double>(num_hidden+1, 0.0));
			std::uniform_real_distribution<double> d(0.0f, 1.0f); 
			std::mt19937 mt; 
			for (i=0;i<=num_visible;i++)
				std::generate(weights[i].begin(),weights[i].end(),[&] { return d(mt); });
			std::fill(weights[0].begin(), weights[0].end(), 0.0);
  			for (i=1;i<=num_visible;i++)
				weights[i][0]=0.0;
			
		}
		
		std::vector<std::vector<double> > augment(std::vector<std::vector<double> > data)
		{
			std::vector<std::vector<double> > new_data;
			new_data.resize(data.size(),std::vector<double>(data[0].size()+1,1.0));
			for(int i=0;i<data.size();i++)
				for (int j=0;j<data[0].size();j++)
					new_data[i][j+1] = data[i][j];
			return new_data;
		
		}
		
		std::vector<std::vector<double> > transpose(std::vector<std::vector<double> > a)
		{
			std::vector<std::vector<double> > t;
			t.resize(a[0].size(),std::vector<double>(a.size(),0.0));
			for (int i=0;i < a.size(); i++){
				for(int j=0; j<a[0].size(); j++)
					t[j][i]=a[i][j];}
			return t;
		}


		std::vector<std::vector<double> > mul(std::vector<std::vector<double> > data,std::vector<std::vector<double> > weights)
		{
			std::vector<std::vector<double> > prod;
			prod.resize(data.size(),std::vector<double>(weights[0].size(),0.0));
			for (int i = 0; i < data.size(); i++) {
        			for (int j = 0; j < weights[0].size(); j++) {
					for (int k=0; k <data[0].size(); k++)
        	    			    prod[i][j] += data[i][k]*weights[k][j]; 
            		
        			 	}
    				}
    			return prod;

		}


		std::vector<std::vector<double> > sum(std::vector<std::vector<double> > a,std::vector<std::vector<double> > b)
		{
			std::vector<std::vector<double> > s;
			s.resize(a.size(),std::vector<double>(a[0].size(),0.0));
			for (int i = 0; i < a.size(); i++) {
        			for (int j = 0; j < a[0].size(); j++) {
					s[i][j] = a[i][j] + b[i][j]; }
	    			}
	    		return s;

		}	
		
		std::vector<std::vector<double> > sigm(std::vector<std::vector<double> > a)
		{
			std::vector<std::vector<double> > sig;
			sig.resize(a.size(),std::vector<double>(a[0].size(),0.0));
			for (int i=0;i < a.size(); i++){
				for(int j=0; j<a[0].size(); j++)
					sig[i][j] = 1/(1+exp(-a[i][j]));}
			return sig;
		}


		std::vector<std::vector<double> > rnd(std::vector<std::vector<double> > a)
		{
			std::vector<std::vector<double> > r_a,r;
			r_a.resize(a.size(),std::vector<double>(a[0].size(),0.0));
			std::mt19937 mt;	
  			std::uniform_real_distribution<double> d(0.0, 1.0);
			r.resize(a.size(),std::vector<double>(a[0].size(),0.0));
			for (int i=0;i<r.size();i++)
				std::generate(r[i].begin(),r[i].end(),[&] { return d(mt); });
			for (int i=0;i < a.size(); i++){
				for(int j=0; j<a[0].size(); j++){
					if (a[i][j]>r[i][j])
						r_a[i][j] = 1;}}
			return r_a;

		}
               
		std::vector<std::vector<double> > bias(std::vector<std::vector<double> > a)
		{
			for (int i=0; i<a.size();i++)
				a[i][0] = 1;
			return a;
		}

		double sub_2(std::vector<std::vector<double> > d1,std::vector<std::vector<double> > d2)
		{
			double err = 0.0;
			for (int i = 0; i < d1.size(); i++) {
        			for (int j = 0; j < d1[0].size(); j++){
					err += pow((d1[i][j] - d2[i][j]),2);}
				
			}
			return err;

		}


		std::vector<std::vector<double> > cd(std::vector<std::vector<double> > data,int iter)
		{
			std::vector<std::vector<double> > u;
			u.resize(weights.size(),std::vector<double>(weights[0].size(),0.0));
			new_data = augment(data);
			hidden_probs = sigm(mul (new_data , weights));
			hidden_states = bias(rnd(hidden_probs));
			n_d = transpose(new_data);
			pos_associations = mul(n_d, hidden_states);
			r_hidden_probs = hidden_probs;
			wts = transpose(weights);
			for(i=0;i<iter;i++){
				hidden_states = rnd(r_hidden_probs);
				r_data= rnd(sigm(mul(hidden_states,wts)));
				r_data= bias(r_data);
       				r_hidden_probs = sigm(mul(r_data, weights));
			}
			rt_data = transpose(r_data);
                	neg_associations = mul(rt_data, rnd(r_hidden_probs));
			for (int i=0; i<weights.size(); i++)
				for (int j=0; j<weights[0].size(); j++)
					u[i][j] = (pos_associations[i][j] - neg_associations[i][j]) / data.size();
                	return u; 
		}
		
		void train(std::vector<std::vector<double> >data, int epochs, double learning_rate = 0.1)
		{
			init_weights();			
			update.resize(num_visible+1, std::vector<double>(num_hidden+1, 0.0));
			for (int e=0;e<epochs;e++)
			{
				update = cd(data,1);
				for(i=0;i<update.size();i++){
     					std::transform(update[i].begin(), update[i].end(), update[i].begin(),
					std::bind1st(std::multiplies<double>(),learning_rate));}
				weights = sum(weights,update);
				hidden_probs = sigm(mul (new_data , weights));
				hidden_states = rnd(hidden_probs);
				r_data= rnd(bias(sigm(mul(hidden_states, wts))));
				error = sub_2(new_data,r_data);
				cout<<"epoch : "<<e<<"\t"<<error<<endl;
			}
		}
		
		void daydream(std::vector<std::vector<double> > data, int iter)
		{	
			std::vector<std::vector<double> > d;
			d = augment(data);
			hidden_probs = sigm(mul (d , weights));
			hidden_states = rnd(hidden_probs);
			for(int i=0;i<iter;i++){
					hidden_states = rnd(hidden_probs);
					r_data= rnd(sigm(mul(hidden_states,wts)));
					r_data= bias(r_data);
					for(int j=0 ;j<r_data[0].size();j++)
						cout<<r_data[1][j]<<"\t";
					cout<<endl;
       					hidden_probs = sigm(mul(r_data, weights));
			}
			
	
		}
		
};


int main()
{
	vector<vector<double> >data,d;
	d.resize(2, vector<double>(6, 1.0));	 
	RBM r;
	r.num_visible = 6;
	r.num_hidden = 4;
	int d1[2][6] = {{1,0,1,0,1,0},{0,0,0,0,0,0}};
	d[0].assign (d1[0],d1[0]+6);	
	d[1].assign (d1[1],d1[1]+6);
        int data1[6][6] = {{1,1,1,0,0,0},{1,0,1,0,0,0},{1,1,1,0,0,0},{0,0,1,1,1,0},{0,0,1,1,0,0},{0,0,1,1,1,0}};
  	data.resize(6, vector<double>(6, 1.0));
  	for(int i=0;i<6;i++)
     		data[i].assign (data1[i],data1[i]+6);
        r.train(data,5000);
	r.daydream(d,3);
	return 0;
}








