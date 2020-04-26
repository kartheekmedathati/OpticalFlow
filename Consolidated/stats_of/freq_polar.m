%%%%

load alley_1_summarystats.mat
figure,imagesc(log(cumulativestats))

clear X Y C

tmp=cumulativenergy{49,49};
tmp=Cumulativenergy{20,80};
[ori,vel,scale]=size(tmp);
for s=1:scale
    C(:,s)=sum(tmp(:,:,s),2);
end
C(ori+1,:)=1;
r = (1:(scale))'/scale;
%%%
theta = pi*(-ori/2:ori/2)/(ori/2);%-pi/2;
X = r*cos(theta);
Y = r*sin(theta);

figure,pcolor(X,Y,C')%,colormap gray
axis equal tight,axis off

%%%%%%%

[xq,yq]=meshgrid(-1:0.1:1);
X=X';
X(end,:)=[];
Y=Y';
Y(end,:)=[];
for ii=1:vel
    tmp2=squeeze(tmp(:,ii,:));
    vq = griddata(X(:),Y(:),tmp2(:),xq,yq);
    figure,subplot(121),imagesc(vq),axis square
    subplot(122),pcolor(X,Y,tmp2),axis square
    VQ(:,:,ii)=vq;
end

MAX=max(max(max(VQ)));
figure,isosurface(VQ,MAX/2),grid on,xlabel('W_x'),ylabel('W_y'),zlabel('W_t'),
