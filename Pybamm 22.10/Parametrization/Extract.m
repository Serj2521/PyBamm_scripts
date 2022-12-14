clc
clear all

load('output_DFN.mat')
load('output_SPM.mat')
Max_surf_concentration_DFN = max(c_p_DFN(end,:))
Max_surf_concentration_SPM = max(c_p_SPM(end,:))

R = 5.22e-6;             % Particle Radius [m]
R_p=linspace(0,R,length(c_p_DFN(:,1)));

surf(t_SPM,R_p,c_p_SPM,'facecolor','b','facealpha',0.3,'edgecolor',[1 0 1])
    grid('on');
    xlabel('Distance [m]','interpreter','latex'); ylabel('Time [h]','interpreter','latex'); zlabel('Concentration [mol/m3]','interpreter','latex');
    title('Time vs R Positive particle Concentration [mol/m3]','interpreter','latex');