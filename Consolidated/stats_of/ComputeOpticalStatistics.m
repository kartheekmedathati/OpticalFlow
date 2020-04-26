%% 14 May 2016 
%  FlowDescritization
%  FlowDescritizatin = (Vx, Vy)
% 

MPI_Sintel_alley = './Sintel/cave_2_flow/'
StatsDir = './Cave2_Stats/'
fnames = dir(strcat(MPI_Sintel_alley,'*.flo'));

cumulativestats = zeros(101,101);
for i=1:size(fnames,1)
    
    cfname = fnames(i).name;
    flowmat = readFlowFile(strcat(MPI_Sintel_alley,cfname));
    vflow = flowToColor(flowmat);
    vxbins = -50:1:50;
    vybins = 50:-1:-50;
    flwstats = Computeflowhist(flowmat,vxbins,vybins,0.5);
    cumulativestats = cumulativestats + flwstats;
    tsum = sum(sum(flwstats));
    tmap = flwstats/tsum;
    resfilename = strcat(StatsDir,cfname(1:end-4),'.mat');
    resfilenamev = strcat(StatsDir,cfname(1:end-4),'.png');
    save(resfilename,'flwstats','-v7');
    imwrite(mat2gray(tmap>0),resfilenamev); 
    figure(1),subplot(1,2,1),imshow(vflow);
    figure(1),subplot(1,2,2), imshow(flwstats);
    pause(1);
end

save('Cave2_summarystats.mat','cumulativestats','-v7');