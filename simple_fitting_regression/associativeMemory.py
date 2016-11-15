% The inputs are not orthogonal therefore I have used another kind of
% Perfect recall occurs with normalised and orthogonal inputs.
% recall of subtacting the weights from the test case to find its match.



x = load('number.mat');
p = transpose(x.input_data);
y = x.output_data;
W = zeros(15,10);
for i = 1:10
    W = W + (p(i,:)'* y(i,:));
end
t = transpose(x.test_data);
W=transpose(W);
for j =1:5
for i= 1:10
    if not(any(W(i,:)-t(j,:)));
        disp(i);
        disp(j);
    end
end
end
