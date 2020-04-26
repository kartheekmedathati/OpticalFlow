%% 14 May 2016 
%  FlowDescritization
%  FlowDescritizatin = (Vx, Vy)
% 

MPI_Sintel_alley = './eval-gray-allframes/';
StatsDir = './Middlebury_Stats/';
fnames=importdata(strcat('./eval-gray-allframes/','names.txt'));


cumulativestats = zeros(51,51);
cumulativenergy=cell(51,51);
for i=1: (size(fnames,1)-1)
    
    cfname = fnames{i};
    %flowmat = readFlowFile(strcat(MPI_Sintel_alley,cfname));
    tmpot=load(strcat('./eval-gray-allframes/',cfname,'_gt.mat'));
    flowmat=tmpot.O_t;
    load(strcat('./Middlebury_energy/',cfname,'_energy.mat'));

    vflow = flowToColor(flowmat);
    vxbins = -25:1:25;
    vybins = 25:-1:-25;
    [flwstats,cenergy] = Computeflowhist_energy(flowmat,E,vxbins,vybins,0.5);
    cumulativestats = cumulativestats + flwstats;
    if i==1, cumulativenergy=cenergy; else
        for j=1:length(cumulativenergy(:)), cumulativenergy{j}=cumulativenergy{j}+cenergy{j};end
    end
    tsum = sum(sum(flwstats));
    tmap = flwstats/tsum;
    resfilename = strcat(StatsDir,cfname(1:end-4),'.mat');
    resfilenamev = strcat(StatsDir,cfname(1:end-4),'.png');
    save(resfilename,'flwstats','-v7');
    imwrite(mat2gray(tmap>0),resfilenamev); 
    figure(1),subplot(1,2,1),imshow(vflow);
    figure(1),subplot(1,2,2), imagesc(flwstats);
    disp(i)
    pause(1);
end

save('Middlebury_summarystats.mat','cumulativestats','cumulativenergy','-v7');