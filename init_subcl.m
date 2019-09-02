function [W0,H0,ClusterCounter,init_err,elapsed] = init_subcl(A,distances,ra,alpha)
%
% This function implements the Subtractive Clustering initialization 
% algorithm described in [1] for Nonnegative Matrix Factorization Algorithms.
%
% [W0,H0,init_err,elapsed] = SubCl(A,distances,ra);
%
%INPUT: 
%  A=              Data Matrix (R+_mxn) (L2 normalization of columns of X
%  is required)
%  distances=      Distance matrix (R+_nxn). It is obtained using "dist(A)" 
%                  Matlab command. It evaluates euclidean distances among
%                  data vectors
%  ra=             Radius. it has a value between 0 and Inf
%                  and specifies the size of the cluster in each of the
%                  data dimensions
%
%OUTPUT:       
%  W0=             Initial Basis Matrix (R+_mxk)
%  H0=             Initial Encoding Matrix (R+_kxn)
%  init_err=       Reconstruction error, based on mean squared error measure
%  elapsed=        Time spent to compute the function

%
% References:
% 
% [1] Gabriella Casalino, Nicoletta Del Buono, Corrado Mencar,
%     Subtractive clustering for seeding non-negative matrix factorizations,
%     Information Sciences, Volume 257, 1 February 2014, Pages 369-387, 
%     ISSN 0020-0255, http://dx.doi.org/10.1016/j.ins.2013.05.038.
%     (http://www.sciencedirect.com/science/article/pii/S0020025513004349)
% [2] G. Casalino, N. Del Buono, C. Mencar (2011) 
%     Subtractive Initialization of Nonnegative Matrix Factorizations for Document Clustering,
%     188-195. In Fuzzy Logic and Applications (WILF 2011), Pages 188-195, 
%     Lecture Notes in Computer Science, A. Fanelli,  W. Pedrycz, A. Petrosino, 
%     Springer Berlin Heidelberg,ISBN: 978-3-642-23712-6, ISSN: 0302-9743, 
%     doi: 10.1007/978-3-642-23713-3_24.
%     
%     This code is kindly provided by the authors for research purposes.
%     
%     For questions or comments please send an e-mail to 
%     Gabriella Casalino (gabriella.casalino@uniba.it)
%
fprintf('Initialization by Subtractive Clustering is running! \n');
tic;

colA=size(A,2);
ra_frac=4/(ra^2);

D=sum(exp(-ra_frac*distances.^2))';

Centroid=zeros(colA,2);
[Dcl,icl]=max(D); 
Centroid(1,1)=Dcl;
Centroid(1,2)=icl;
ClusterCounter=1;


rb=alpha*ra; %1.25
rb_frac=4/(rb^2);
while 1
    ClusterCounter=ClusterCounter+1;
    D=D- (Dcl*exp(-rb_frac*(distances(:,icl)).^2));
    [Dcl,icl]=max(D);
    Centroid(ClusterCounter,1)=Dcl;
    Centroid(ClusterCounter,2)=icl;
    if ((Centroid(ClusterCounter,1)<0.15*(Centroid(1,1)))==1);
        break;
    end
end


 W0=A(:,Centroid(1:ClusterCounter,2));


sigma=1/(2*ra_frac);
for i=1:ClusterCounter
    for j=1:size(A,2)
    H0(i,j)=exp(-1/2*(sum(((A(:,j)-W0(:,i)).^ 2)./sigma)));
    end
end

NormFactor=sum(H0);

for j=1:size(A,2)
    H0(:,j)=H0(:,j)/NormFactor(j);
end
elapsed = toc;


init_err=(0.5* norm((A-W0*H0),'fro')^2)/colA;


fprintf('Initialization by Subtractive Clustering terminated! \n');

end


