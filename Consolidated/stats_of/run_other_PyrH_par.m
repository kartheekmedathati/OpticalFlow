%run other(training) of all Middlebury

n_scales =5;
nc_min =4;
th=1e-4;%1e-3;
th2=1e-3;%5e-2
n_filters=12;
vel=[-0.9 -0.6 -0.4 0 0.4 0.6 0.9];
%D=21;%speed directions

nns=[n_scales 7 6 7];
thh=[1e-4,1e-4,5e-3,5e-3];

dname='./Middlebury_energy/';

mypath='./eval-gray-allframes/';
names=importdata(strcat(mypath,'names.txt'));
dim=length(names);
for ii=1:1
    n_scales=nns(ii);
    th=thh(ii);
    %jjv=[7];
    for hh=1:dim
        %jj=jjv(hh);
        jj=hh;
        tmpI=load(strcat(mypath,names{jj},'.mat'));
        I=tmpI.I;       
        tic
        %O =ctf_population_flow(I(:,:,2:6),n_scales,th,th2,vel,n_filters,D);
        E = ctf_population_energy(I(:,:,2:6),n_scales,th,th2,vel,n_filters);
        toc
  
        str=sprintf('_energy.mat');
        saveof(strcat(dname,names{jj},str), E)
        %save(strcat(dname,names{jj},str),'EV1')
    end
end

