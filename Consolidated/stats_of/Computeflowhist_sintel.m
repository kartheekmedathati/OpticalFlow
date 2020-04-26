%% 14 May 2016
%  FlowDescritization
%  FlowDescritizatin = (Vx, Vy)
%
% Computes a 2D histogram of velocities
% The first and last columns and rows denote the values that lie outside
% the limits of the velocity domain considered

%%
function [flowstats,cenergy] = Computeflowhist_sintel(flowmat, II, vxbincenters,vybincenters,binwidth)

[Vx, Vy] = meshgrid(vxbincenters,vybincenters);
flowstats = zeros(size(Vx));
cenergy=cell(size(Vx));

Vx_l = Vx(:);
Vy_l = Vy(:);

flowmat_x = flowmat(:,:,1);
flowmat_y = flowmat(:,:,2);

E=run_model(II);
scale=length(E);
[~,~,ori,vel]=size(E{1});
for j=1:length(cenergy(:)), cenergy{j}=zeros(ori,vel,scale);end

for i=1:size(Vx_l,1)
    temp_x_mask = (flowmat_x> (Vx_l(i)-binwidth)).*(flowmat_x<= (Vx_l(i)+binwidth));
    temp_y_mask = (flowmat_y> (Vy_l(i)-binwidth)).*(flowmat_y<= (Vy_l(i)+binwidth));
    
    region_support = temp_x_mask.*temp_y_mask;
    flowstats(i) = sum(sum(region_support));
    
    [rr,cc]=find(region_support);
    TMPE=zeros(ori,vel,scale);
    ES=zeros(ori,vel,scale);
    for j=1:length(rr)
        for s=1:scale
            tmp=squeeze(E{s}(rr(j),cc(j),:,:));
            TMPE(:,:,s)=tmp;
        end
        ES=ES+TMPE;
    end
    if ~isempty(rr), cenergy{i}=ES/length(rr);end
end
flowstats = reshape(flowstats,size(Vx));
return

function E=run_model(II)
n_scales =5;
%nc_min =4;
th=1e-4;%1e-3;
th2=1e-3;%5e-2
n_filters=12;
vel=[-0.9 -0.6 -0.4 0 0.4 0.6 0.9];
%D=21;%speed directions

E = ctf_population_energy(II,n_scales,th,th2,vel,n_filters);

return