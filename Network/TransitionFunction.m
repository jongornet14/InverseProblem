function [z] = TransitionFunction(x,y)

z = normpdf(x-y,0,1);

end
