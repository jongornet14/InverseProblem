function makeFigure(P)

figure
subplot(3,2,1)
plot(P.z,P.Xdist,'b')
xlabel('$$\{x_i\}$$ Space','Interpreter','latex');ylabel('$$P(x_i)$$','Interpreter','latex');title('$$\{x_i\}$$ Distribution','Interpreter','latex')
subplot(3,2,2)
plot(P.z,P.Ydist,'r')
xlabel('$$\{y_j\}$$ Space','Interpreter','latex');ylabel('$$P(y_j)$$','Interpreter','latex');title('$$\{y_j\}$$ Distribution','Interpreter','latex')

subplot(3,2,3)
plot(P.z,P.p_y_x,'b')
xlabel('$$x$$ Space','Interpreter','latex');ylabel('$$p(y|x)$$','Interpreter','latex');title('Initiatial $$p(y|x)$$','Interpreter','latex')
subplot(3,2,5)
plot(P.z,P.p_y_x_,'b')
xlabel('$$x$$ Space','Interpreter','latex');ylabel('$$p^*(y|x)$$','Interpreter','latex');title('Adjusted $$p^*(y|x)$$','Interpreter','latex')

subplot(3,2,4)
plot(P.z,P.YdistPred0,'b')
xlabel('$$y$$ Space','Interpreter','latex');ylabel('$$\rho_1(y)$$','Interpreter','latex');title('Initiatial $$\rho_1(y)$$','Interpreter','latex')
subplot(3,2,6)
plot(P.z,P.YdistPred,'b')
xlabel('$$y$$ Space','Interpreter','latex');ylabel('$$\rho_1^*(y)$$','Interpreter','latex');title('Adjusted $$\rho_1^*(y)$$','Interpreter','latex')

end