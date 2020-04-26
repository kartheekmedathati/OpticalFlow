%% 14 May 2016
%  FlowDescritization
%  FlowDescritizatin = (Vx, Vy)
%

MPI_Sintel_Dir = 'I:/data/MPI-Sintel-complete/training/flow/';
MPI_Sintel_Diri = 'I:/data/MPI-Sintel-complete/training/final/';
ResultsDir = 'I:/data/stats_sintel/';


directory_list = dir(MPI_Sintel_Dir);
directory_list = directory_list(3:end); % Getting rid of '.','..','.ds'
directory_listi = dir(MPI_Sintel_Diri);
directory_listi = directory_listi(3:end); % Getting rid of '.','..','.ds'

parfor k=10:length(directory_list)
    
    
    current_dir = directory_list(k).name;
    current_diri = directory_listi(k).name;
    fnames = dir(strcat(MPI_Sintel_Dir,directory_list(k).name,'/','*.flo'));
    fnamesi = dir(strcat(MPI_Sintel_Diri,directory_listi(k).name,'/','*.png'));
    
    vxbins = -50:1:50;
    vybins = 50:-1:-50;
    cumulativestats = zeros(101,101);
    cumulativenergy=cell(101,101);
    nenergy = zeros(101,101);
    
    %%%setup for par
    cfnamei = fnamesi(1).name; imgf = strcat(MPI_Sintel_Diri,directory_listi(k).name,'/',cfnamei);
    imgtmp=imread(imgf); [rrimg,ccimg,~]=size(imgtmp);
    II=zeros(rrimg,ccimg,5);
    %%%
    for i=(1:size(fnames,1)-4)
        cfname = fnames(i).name;
        flowmat = readFlowFile(strcat(MPI_Sintel_Dir,directory_list(k).name,'/',cfname));
        vflow = flowToColor(flowmat);
        %%%of
        for jj=i:(i+4)
            cfnamei = fnamesi(jj).name; imgf = strcat(MPI_Sintel_Diri,directory_listi(k).name,'/',cfnamei);
            II(:,:,jj-i+1)= double(rgb2gray(imread(imgf)))/255;
        end
        %%%

        [flwstats,cenergy] = Computeflowhist_sintel(flowmat,II,vxbins,vybins,0.5);
        cumulativestats = cumulativestats + flwstats;
        tsum = sum(sum(flwstats));
        tmap = flwstats/tsum;
        %%%of
        if i==1
            cumulativenergy=cenergy; 
            for jj=1:length(cumulativenergy(:))
                if sum(cumulativenergy{jj}(:))~=0, nenergy(jj)=nenergy(jj)+1;end
            end
        else
            for jj=1:length(cumulativenergy(:))
                cumulativenergy{jj}=cumulativenergy{jj}+cenergy{jj};
                if sum(cumulativenergy{jj}(:))~=0, nenergy(jj)=nenergy(jj)+1;end
            end
        end
        %%%

        Current_Res_Dir = strcat(ResultsDir,current_dir);
        
        if ~exist(Current_Res_Dir, 'dir')
            mkdir(Current_Res_Dir);
        end
        
        resfilename = strcat(Current_Res_Dir,'/',cfname(1:end-4),'.mat');
        save_loc_stats(resfilename,flwstats,cenergy)
        %save(resfilename,'flwstats','cenergy','-v7');
        %imwrite(mat2gray(tmap>0),resfilenamev);
        
        %     figure(1),subplot(1,2,1),imshow(vflow);
        %     figure(1),subplot(1,2,2), imshow(flwstats);
        %     pause(1);
    end
    
    save_glob_stats(strcat(Current_Res_Dir,'/',current_dir,'_summarystats.mat'),cumulativestats,cumulativenergy,nenergy)
    %save(strcat(Current_Res_Dir,'/',current_dir,'_summarystats.mat'),'cumulativestats','cumulativenergy','nenergy','-v7');
    
end;

