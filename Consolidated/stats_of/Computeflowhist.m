%% 14 May 2016 
%  FlowDescritization
%  FlowDescritizatin = (Vx, Vy)
% 
% Computes a 2D histogram of velocities 
% The first and last columns and rows denote the values that lie outside
% the limits of the velocity domain considered

%%
function flowstats = Computeflowhist(flowmat, vxbincenters,vybincenters,binwidth)

[Vx, Vy] = meshgrid(vxbincenters,vybincenters);
flowstats = zeros(size(Vx));

Vx_l = Vx(:);
Vy_l = Vy(:);

flowmat_x = flowmat(:,:,1);
flowmat_y = flowmat(:,:,2);

parfor i=1:size(Vx_l,1)
    temp_x_mask = (flowmat_x> (Vx_l(i)-binwidth)).*(flowmat_x<= (Vx_l(i)+binwidth));
    temp_y_mask = (flowmat_y> (Vy_l(i)-binwidth)).*(flowmat_y<= (Vy_l(i)+binwidth));
   
    region_support = temp_x_mask.*temp_y_mask;
    flowstats(i) = sum(sum(region_support));
end
flowstats = reshape(flowstats,size(Vx));
return