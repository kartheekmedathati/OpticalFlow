function EV1 = ctf_population_energy(II,n_scales,th,th2,vel,n_filters)

n_frames = size(II,3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change Image Size for Pyramid Construction %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[sy1 sx1 st] = size(II);

fac = 2^(n_scales-1);

sy2 = ceil(sy1 ./ fac) .* fac; % target resolution
sx2 = ceil(sx1 ./ fac) .* fac; % target resolution

II = [ II ; repmat(II(end,:,:),[sy2-sy1 1 1]) ]; % replicate border row
II = [ II repmat(II(:,end,:),[1 sx2-sx1 1]) ]; % replicate border column


%%%%%%%%%%%%%%%%%
% Image Pyramid %
%%%%%%%%%%%%%%%%%


[II] = image_pyramid(II,n_frames,n_scales);


%%%%%%%%%%%%%%%%%%%%%%%%%
% Level 1 full velocity %
%%%%%%%%%%%%%%%%%%%%%%%%%

F = filt_gabor_space(II{1},n_filters);
F = filt_gabor_time(F,vel);

Ev1tmp= V1(F,II{1},th,th2,vel);

for jj=1:n_scales-1
    Ev1tmp=expandV1(Ev1tmp);
end

Ev1tmp(end-(sy2-sy1-1):end,:,:,:) = [];
Ev1tmp(:,end-(sx2-sx1-1):end,:,:) = [];

EV1{1}=Ev1tmp;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coarse-to-fine Estimation and Merging %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for scale = 2:n_scales
    
    
    F = filt_gabor_space(II{scale},n_filters);
    F = filt_gabor_time(F,vel);
    
    Ev1tmp= V1(F,II{scale},th,th2,vel);
        
    for jj=1:n_scales-scale
        Ev1tmp=expandV1(Ev1tmp);
    end
    
    Ev1tmp(end-(sy2-sy1-1):end,:,:,:) = [];
    Ev1tmp(:,end-(sx2-sx1-1):end,:,:) = [];

    EV1{scale}=Ev1tmp;
    
    clear F Ev1tmp
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [II,x_pix,y_pix] = image_pyramid(II,n_frames,n_scales)


[sy, sx, st] = size(II);

x_pix = cell(1,n_scales);
y_pix = cell(1,n_scales);

[ x_pix{n_scales}, y_pix{n_scales} ] = meshgrid(1:sx,1:sy);

lpf = [1 4 6 4 1]/16;

tmp = II;
II = cell(1,n_scales);
II{n_scales} = tmp;

for scale = n_scales-1:-1:1
    for frame = 1:n_frames
        tmp(:,:,frame) = conv2b(conv2b(tmp(:,:,frame),lpf),lpf');
    end
    [Ny, Nx, dummy] = size(tmp);
    
    tmp = tmp(1:2:Ny,1:2:Nx,:);
    II{scale} = tmp;
    x_pix{scale} = x_pix{scale+1}(1:2:Ny,1:2:Nx);
    y_pix{scale} = y_pix{scale+1}(1:2:Ny,1:2:Nx);
end



%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%


function Eout = expandV1(E)


sy = size(E,1);
sx = size(E,2);

[X Y] = meshgrid(1:(sx-1)/(2*sx-1):sx, ...
    1:(sy-1)/(2*sy-1):sy);

n_orient = size(E,3);
n_vel = size(E,4);

% Repeat edge pixel for border handling

for ii=1:n_orient
    for jj=1:n_vel
        Etmp=E(:,:,ii,jj);
        Etmp = [ Etmp Etmp(:,end,:) ];
        Etmp = [ Etmp ; Etmp(end,:,:) ];
        Etmp = bilin_interp(Etmp,X,Y);
        Eout(:,:,ii,jj)=Etmp;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [E_v1]= V1(F,II,th,th2,vel)

n_vel=size(F,2);


[sy, sx, n_orient]=size(F{1,1});
%E=zeros(sy,sx,n_orient,n_vel);
E_v1=zeros(sy,sx,n_orient,n_vel);



for n_v=1:n_vel
    E_v1(:,:,:,n_v)= sqrt(F{1,n_v}.^2+F{2,n_v}.^2);%%E
end

% % E=E.^0.5;
% % 
% % E=E/max(max(max(max(E))));
% % mask=E>th;
% % E=E.*mask;
% % 
% % %%%normalization: tmp matrix used later
% % tmp=zeros(sy,sx,n_vel);
% % for v=1:n_vel
% %     for o=1:n_orient
% %         tmp(:,:,v)=tmp(:,:,v)+E(:,:,o,v);
% %     end
% % end
% % 
% % 
% % filt = fspecial('gaussian', [5 5], 10/6);
% % filt=filt./sum(sum(fspecial('gaussian', [5 5], 10/6)));
% % 
% % 
% % for v=1:n_vel
% %     for o=1:n_orient
% %         E_v1(:,:,o,v)=E(:,:,o,v)./(tmp(:,:,v)+1e-9);%normalization
% %         E_v1(:,:,o,v) = conv2(E_v1(:,:,o,v), filt, 'same');%V1 spatial pooling
% %         mask=ones(sy,sx);
% %         mask(tmp(:,:,v)<th2)=NaN;%threshold (unreliable pixels)
% %         E_v1(:,:,o,v)=E_v1(:,:,o,v).*mask;
% %     end
% % end
% % 
% % 
if sy<20 && (~isequal(isnan(E_v1),zeros(size(E_v1))) )
    for v=1:n_vel
        for o=1:n_orient
            Os(:,:)=E_v1(:,:,o,v);
            Os=myfillin_pp(Os);
            E_v1(:,:,o,v)=Os(:,:);
        end
    end
end

if sy>=20
    for v=1:n_vel
        for o=1:n_orient
            Os(:,:)=E_v1(:,:,o,v);
            Os(1:5,:,:)=NaN; Os((end-4):end,:,:)=NaN;  Os(:,1:5,:)=NaN; Os(:,(end-4):end,:)=NaN;
            Os=myfillin_pp(Os);
            E_v1(:,:,o,v)=Os(:,:);
        end
    end
end

% % 
% % E_v1=1*E_v1/max(max(max(max(E_v1))));%exponential gain


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function IF = filt_gabor_space(I,n_filters)

[sy,sx,n_frames] = size(I);

IF{1} = zeros(sy,sx,n_filters,n_frames);
IF{2} = zeros(sy,sx,n_filters,n_frames);


w=5;
f0=1/3.8;
sigma_s=1.2*sqrt(2*log(2)) ./ (2*pi*(f0/3));


[X,Y] = meshgrid(-w:w,-w:w);
theta=0:2*pi/n_filters:(2*pi-2*pi/n_filters);

G = exp(-(X.^2+Y.^2)/(2*sigma_s^2));

for ii=1:length(theta)
    XT=cos(theta(ii))*X+sin(theta(ii))*Y;
    GC=G.*cos(2*pi*f0*XT);
    GCB{ii}=GC-sum(sum(GC))/(2*w+1)^2;%DC
    GS=G.*sin(2*pi*f0*XT);
    GSB{ii}=GS-sum(sum(GS))/(2*w+1)^2;%DC
end


for frame = 1:n_frames
    
    for ii=1:n_filters/2
        even=conv2b(I(:,:,frame),GCB{ii});
        odd=conv2b(I(:,:,frame),GSB{ii});
        
        IF{1}(:,:,ii,frame) = even;
        IF{1}(:,:,ii+n_filters/2,frame) = even;
        
        IF{2}(:,:,ii,frame) = odd;
        IF{2}(:,:,ii+n_filters/2,frame) = -odd;
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function IF = filt_gabor_time(F,v)


n_vel=size(v,2);
[sy, sx, n_orient, n_frames]=size(F{1});

for n_v=1:n_vel
    
    w0=-v(n_v)/3.8;
    
    
    %%%EXP
    t=0:n_frames-1;
    f0=1/3.8;
    B=f0/2.5;
    sigma = sqrt(2*log(2)) ./ (2*pi*B);
    g = exp(-t ./ (2.*sigma.^2));
    Fts = g.*sin(2*pi*w0*t);
    Ftc = g.*cos(2*pi*w0*t);
    Ftc=Ftc';
    Fts=Fts';
    
    
    %%% GABOR
    %             t=0:n_frames-1;
    %             f0=1/3.8;
    %             sigma=1.2*sqrt(2*log(2)) ./ (2*pi*(f0/3));
    %             g = exp(-t.^2 ./ (2.*sigma.^2));
    %             Fts = g.*sin(2*pi*w0*t);
    %             Ftc = g.*cos(2*pi*w0*t);
    %
    %             Ftc=Ftc';
    %             Fts=Fts';
    
    
    
    %%%Adelson
    %     kk=[-3 -2.5 -1.5  0.5  1.5 2.5 3];
    %     t=0:n_frames-1;
    %     k=kk(n_v);
    %     if k<0
    %         k=-k;
    %         Fts = ((k*t).^5).*exp(-k*t).*(1/factorial(5) - ((k*t).^2)/factorial(5+2));
    %     else
    %         Fts = -((k*t).^5).*exp(-k*t).*(1/factorial(5) - ((k*t).^2)/factorial(5+2));
    %     end
    %     Ftc =((k*t).^3).*exp(-k*t).*(1/factorial(3) - ((k*t).^2)/factorial(3+2));
    %     Ftc=Ftc';
    %     Fts=Fts';
    
    
    G_even_tmp=F{1};
    G_odd_tmp=F{2};
    
    G_even3d=zeros(sy,sx,n_orient);
    G_odd3d=zeros(sy,sx,n_orient);
    
    for orient=1:n_orient
        for i=1:sy
            
            G_even3d(i,:,orient) = (conv2(squeeze(G_even_tmp(i,:,orient,:))',Ftc,'valid')-conv2(squeeze(G_odd_tmp(i,:,orient,:))',Fts,'valid'))';
            G_odd3d(i,:,orient) = (conv2(squeeze(G_even_tmp(i,:,orient,:))',Fts,'valid')+conv2(squeeze(G_odd_tmp(i,:,orient,:))',Ftc,'valid'))';
            
        end
    end
    
    IF{1,n_v}= squeeze(G_even3d);
    IF{2,n_v} = squeeze(G_odd3d);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function imf=conv2b(im, ker)
[nky,nkx]=size(ker);
sh='valid';
Bx=(nkx-1)/2;
By=(nky-1)/2;
im=putborde(im,Bx,By);
imf=conv2(im,ker,sh);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function imb=putborde(im,Nx,Ny)

[sy,sx]=size(im);
imb=zeros(sy+2*Ny,sx+2*Nx);
imb(1+Ny:sy+Ny,1+Nx:sx+Nx)=im;

for k=1:Nx
    imb(Ny+1:sy+Ny,k)=im(:,1);
    imb(Ny+1:sy+Ny,k+sx+Nx)=im(:,sx);
end
for k=1:Ny
    imb(k,Nx+1:sx+Nx)=im(1,:);
    imb(k+sy+Ny,Nx+1:sx+Nx)=im(sy,:);
end



function I2 = bilin_interp(I1,X,Y)
% function I2 = bilin_interp(I1,X,Y)
%
% Arbitrary image rewarping using bilinear interpolation
%
% X (column) and Y (row) (both floating point) are the source locations,
% used to fill the respective pixels

[nY1,nX1,rem] = size(I1); % source size
[nY2,nX2] = size(X); % target size

s = size(I1);
s(1:2) = [];

I2 = NaN.*zeros([ nY2 nX2 s ]);

for r = 1:rem
    
    for x = 1:nX2
        for y = 1:nY2
            
            % Pixel warping (2x2 group)
            
            x_w = floor(X(y,x));
            y_w = floor(Y(y,x));
            
            % Check validity
            
            if ( (x_w>0) && (x_w<nX1) && (y_w>0) && (y_w<nY1) )
                
                xs = X(y,x) - x_w;
                min_xs = 1-xs;
                ys = Y(y,x) - y_w;
                min_ys = 1-ys;
                
                w_00 = min_xs*min_ys;  % w_xy
                w_10 = xs*min_ys;
                w_01 = min_xs*ys;
                w_11 = xs*ys;
                
                I2(y,x,r) = w_00*I1(y_w,x_w,r) + w_10*I1(y_w,x_w+1,r) + ...
                    w_01*I1(y_w+1,x_w,r) + w_11*I1(y_w+1,x_w+1,r);
                
            end
        end
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function O=myfillin_pp(Oin)
%2nd formula@p

filt = fspecial('gaussian', [15 15], 15/6);
filt=filt./sum(sum(fspecial('gaussian', [15 15], 15/6)));

tmp=Oin(:,:); tmp2=Oin(:,:);
tmp2(isnan(tmp2))=0;
masknonan=~isnan(tmp);
trueborder = bwmorph(masknonan,'remove');
tmp((trueborder)==0)=0;
num=conv2b(tmp, filt);
den=conv2b(trueborder, filt);
den(den==0)=1;
O1filled=(num./den).*(~masknonan) + tmp2.*masknonan;
clear tmp tmp2 masknonan trueborder num den




O(:,:)=O1filled;





