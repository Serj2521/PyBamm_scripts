clc
clearvars -except c_p Cmax
c_p_t=c_p'; % Concentration from SPM matlab, for this program "SPM_positive_electrode_Finite_Difference_Volume.m"  should be run before

load('output_DFN.mat')
load('output_SPM.mat')
Max_surf_concentration_DFN = max(c_p_DFN(end,:))
t_max_C_DFN=t_DFN(find(c_p_DFN(end,:)==Max_surf_concentration_DFN,1))
Max_surf_concentration_SPM = max(c_p_SPM(end,:))
t_max_C_SPM=t_SPM(find(c_p_SPM(end,:)==Max_surf_concentration_SPM,1))
T_finrest_SPM=t_SPM(find(t_SPM>=t_max_C_SPM+1,1));
k_end=find(t_SPM>=t_max_C_SPM+1,1);


for i=1:k_end
    k(i)=find(0:length(c_p_t(1,:))-2>=t_SPM(i)*3600,1);
    c_p_surfrep(:,i) = c_p_t(:,k(i));
end



Err_FVM_Py=100*abs(c_p_surfrep-c_p_SPM(:,1:k_end))/Cmax;
Mean_Err_FVM_Py=mean(Err_FVM_Py,"all"); % Mean absolute error

R = 5.22e-6;             % Particle Radius [m]
R_p=linspace(0,R,length(c_p_DFN(:,1)));

subplot(1,2,1);
surf(t_SPM(1:k_end),R_p,c_p_SPM(:,1:k_end),'facecolor','b','facealpha',0.3,'edgecolor',[1 0 1])
    grid('on');
    xlabel('Distance [m]','interpreter','latex'); ylabel('Time [h]','interpreter','latex'); zlabel('Concentration [mol/m3]','interpreter','latex');
    title('Time vs R Positive particle Concentration [mol/m3]','interpreter','latex');

subplot(1,2,2); 
    colormap turbo
    imagesc(t_SPM(1:k_end),R_p,Err_FVM_Py) %Error between FVM & Pybamm
    colorbar
    xlabel('Time t [h]','interpreter','latex')
    ylabel('Distance x[m]','interpreter','latex')
    title('Absolute Err. distribution (%)')

    fprintf('Mean error between FVM & Pybamm concentration results is %1.7f percent\n',Mean_Err_FVM_Py)