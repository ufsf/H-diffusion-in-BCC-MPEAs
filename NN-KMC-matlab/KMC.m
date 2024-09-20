clc;
clear;
%%
tic
imat = 2;
temp = [300:100:1200 2000 3500 5000];
% temp = 300;

Size = 20;
MSD_max = 1e3;
for i = 1:length(temp)
    res = CalcD(imat,temp(i),MSD_max,Size);
    save([num2str(temp(i)),'-',num2str(imat),'.mat'],'res');
end
toc