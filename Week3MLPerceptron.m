v = [-1,0,1;0,1,1];
w = [1,0,1;-1,1,1];
z = [0;0;1];
x = [0;1;1];
d = [1;0];
lr = 0.1;

for i = 1:50
    %Outputs/forward pass
    u = v*x;
    for i = 1:2
        z(i) = g(u(i));
    end
    a = w*z;
    for i = 1:2
        y(i) = g(a(i));
    end 
    %Error calculation
    yE(1) = -(d(1) - y(1)); 
    yE(2,1) = -(d(2) - y(2));
    E = (yE(1) + yE(2))^2;
    zE = w(:,1:2)'*yE;
    %Weight updates
    for j = 1:2
       for k = 1:2
          w(k,j) = w(k,j) -lr*yE(k)*z(j); 
       end
    end
    for j = 1:2
       for k = 1:2
          v(k,j) = v(k,j) -lr*zE(j)*x(i);
       end
    end
    %w bias weights
    w(1,3) = w(1,3) -lr*yE(2);
    w(2,3) = w(2,3) -lr*yE(1);
    %v bias weights
    v(1,3) = v(1,3) -lr*zE(2);
    v(2,3) = v(2,3) -lr*zE(1);
    

end




