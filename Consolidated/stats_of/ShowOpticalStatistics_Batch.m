%% 14 May 2016
%  FlowDescritization
%  FlowDescritizatin = (Vx, Vy)
%

MPI_Sintel_Dir = 'I:/data/MPI-Sintel-complete/training/flow/';
%MPI_Sintel_Diri = 'I:/data/MPI-Sintel-complete/training/final/';
ResultsDir = 'I:/data/stats_sintel/';


directory_list = dir(MPI_Sintel_Dir);
directory_list = directory_list(3:end); % Getting rid of '.','..','.ds'
% directory_listi = dir(MPI_Sintel_Diri);
% directory_listi = directory_listi(3:end); % Getting rid of '.','..','.ds'

Cumulativestats = zeros(101,101);
Cumulativenergy=cell(101,101);
Nenergy = zeros(101,101);

for k=1:length(directory_list)
    
    current_dir = directory_list(k).name;
    
    Current_Res_Dir = strcat(ResultsDir,current_dir);
    load(strcat(Current_Res_Dir,'/',current_dir,'_summarystats.mat'))
    
    Cumulativestats = Cumulativestats + cumulativestats;
    if k==1
        Cumulativenergy=cumulativenergy;
%         for jj=1:length(cumulativenergy(:))
%             if sum(cumulativenergy{jj}(:))~=0, Nenergy(jj)=nenergy(jj)+1;end
%         end   
    else
        for jj=1:length(Cumulativenergy(:))
            Cumulativenergy{jj}=Cumulativenergy{jj}+cumulativenergy{jj};
            %if sum(cumulativenergy{jj}(:))~=0, nenergy(jj)=nenergy(jj)+1;end
        end
    end
    %Cumulativenergy=Cumulativenergy + cumulativenergy;
    Nenergy = Nenergy +nenergy;
    figure,imagesc(log(cumulativestats))
end;
figure,imagesc(log(Cumulativestats))
figure,imagesc((Cumulativestats))
