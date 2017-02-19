unitargetCode=zeros(64,871824);
for i=1:1:871824
	target=unitarget(i);
	index =find(lbgCodebook==target);
	code  =zeros(64,1);
	code(index)=1;
	unitargetCode(:,i)=code;
end
