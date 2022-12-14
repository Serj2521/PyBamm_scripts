clc
clear all

load('output.mat')
R = 5.22e-6;             % Particle Radius [m]
Nr = 20;                 % Number of shells radially
R_p=linspace(0,R,Nr);