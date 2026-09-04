
% Requirement:
%   functionRlocalscattering.m must be on the MATLAB path.

clear; clc; close all;
rng(42);

%% ========================================================================
%  PART 1: SYSTEM PARAMETERS
% =========================================================================

M        = 10;     % Number of APs
K        = 20;     % Number of UEs
J        = K;      
Nt       = 16;     % Number of antennas per AP
tau_p    = 8;      % Pilot length; tau_p < J means underdetermined random pilot matrix
tau_c    = 200;

nTopoTotal      = 50;
nTrainTopo      = 40;
nTestTopo       = 10;
nSamplesPerTopo = 50;

squareLength = 1000;

% Noise parameters
B_hz         = 20e6;
noiseFigure  = 7;
noiseVardBm  = -174 + 10*log10(B_hz) + noiseFigure;
noiseVar_W   = db2pow_local(noiseVardBm - 30);
noiseVar_mW  = db2pow_local(noiseVardBm);

% Per-symbol pilot transmit-power levels.

pilotPowerLevels_dBm = -10:5:10;
nPilotPowerLevels    = length(pilotPowerLevels_dBm);

if nTrainTopo + nTestTopo ~= nTopoTotal
    error('nTrainTopo + nTestTopo must equal nTopoTotal.');
end

if mod(nSamplesPerTopo, nPilotPowerLevels) ~= 0
    error('nSamplesPerTopo must be divisible by the number of pilot-power levels.');
end

nSamplesPerPilotPower = nSamplesPerTopo / nPilotPowerLevels;
pilotPowerLevels_mW    = db2pow_local(pilotPowerLevels_dBm);
pilotBlockEnergyLevels_mW = tau_p * pilotPowerLevels_mW;

% Rician setup
useRandomLoSPhase = true;
pilotAssignMode   = 'globalRandomP';
pilotMatrixMode   = 'complexGaussianUnitNorm';

% Rician parameters
K_ref_dB      = 5;       
K_min_dB      = NaN;    
K_max_dB      = NaN;     
minLinkDist_m = 30;

% Local scattering model parameters
ASD_varphi_deg = 10;
ASD_theta_deg  = 10;
ASD_deg        = ASD_varphi_deg;
ASD_varphi     = ASD_varphi_deg * pi/180;
ASD_theta      = ASD_theta_deg  * pi/180;
antennaSpacing = 1/2;

% Pathloss/shadowing model
pathlossModel = 'Cost231_Walfish_Ikegami_LoS_RicianProject';
APheight      = 12.5;
UEheight      = 1.5;
distVertical  = APheight - UEheight;
sigma_sf      = 8;
decorr        = 100;
alpha_pl      = 26;
constantTerm  = -30.18;

% Output options
datasetRoot = 'dataset';
cleanOutput = true;

%% ========================================================================
%  PART 2: BASIC INFORMATION
% =========================================================================

if tau_p >= J
    warning('tau_p >= J: random pilot matrix is not underdetermined.');
end

fprintf('=== Cell-Free Rician Dataset Generation ===\n');
fprintf('System        : M=%d, J=%d, Nt=%d, tau_p=%d\n', M, J, Nt, tau_p);
fprintf('Topologies    : train=%d, test=%d\n', nTrainTopo, nTestTopo);
fprintf('Samples       : %d per topology, %d per pilot-power level\n', ...
    nSamplesPerTopo, nSamplesPerPilotPower);
fprintf('Pilot power [dBm]: [%s]\n', num2str(pilotPowerLevels_dBm));
fprintf('Output        : %s\n\n', fullfile(pwd, datasetRoot));

%% ========================================================================
%  PART 3: CREATE DATASET FOLDERS
% =========================================================================

if cleanOutput && exist(datasetRoot, 'dir')
    rmdir(datasetRoot, 's');
end

mkdir_if_missing(datasetRoot);

splitNames = {'train','test'};
for ss = 1:numel(splitNames)
    splitName = splitNames{ss};

    mkdir_if_missing(fullfile(datasetRoot, splitName));
    mkdir_if_missing(fullfile(datasetRoot, splitName, 'channel'));
    mkdir_if_missing(fullfile(datasetRoot, splitName, 'channel', 'data'));
    mkdir_if_missing(fullfile(datasetRoot, splitName, 'channel', 'label'));
    mkdir_if_missing(fullfile(datasetRoot, splitName, 'channel', 'pilot'));
    mkdir_if_missing(fullfile(datasetRoot, splitName, 'beam'));
    mkdir_if_missing(fullfile(datasetRoot, splitName, 'beam', 'data'));
    mkdir_if_missing(fullfile(datasetRoot, splitName, 'beam', 'others'));
end

%% ========================================================================
%  PART 4: GLOBAL RANDOM PILOT MATRIX
% =========================================================================

% IMPORTANT:
%   Phi_user is generated ONCE here and then reused for every topology,
%   every AP, every pilot-power level, and both train/test splits.
%   Size: [tau_p x J]. Since tau_p < J, the observation is still rank-limited,
%   but users no longer share exactly identical one-hot pilots.

[Phi_user, pilotColumnNorms, pilotGram_complex, ...
    G_mix_complex, G_abs, G_mix_realimag] = ...
    generate_global_random_pilot_matrix_local(tau_p, J, pilotMatrixMode);

Phi_pool     = Phi_user;          
Phi_complex  = Phi_user;
Phi_real     = [real(Phi_user), imag(Phi_user)];
Phi_realimag = cat(3, real(Phi_user), imag(Phi_user));
P_right      = Phi_user.';        % [J x tau_p], used in the normalized pilot model
P_matrix     = conj(Phi_user);    % [tau_p x J], used for matched filtering Y*conj(Phi_user)

G_complex  = G_mix_complex;
G_realimag = G_mix_realimag;

if norm(G_mix_complex - P_right * P_matrix, 'fro') > 1e-12
    error('Pilot mixing-matrix construction is inconsistent.');
end

pilotIndex       = (1:J).';
pilotReuseMatrix = eye(J);
Pset             = 1:J;
maxReuse         = 1;
meanReuse        = 1;
pilotGramRank    = rank(G_complex, 1e-10);

offDiagMask = ~eye(J);
offDiagAbs  = abs(G_complex(offDiagMask));
maxAbsPilotCorr  = max(offDiagAbs);
meanAbsPilotCorr = mean(offDiagAbs);

pilotInfoTrainDir = fullfile(datasetRoot, 'train', 'channel', 'pilot');
pilotInfoTestDir  = fullfile(datasetRoot, 'test',  'channel', 'pilot');

ticTotal = tic;

%% ========================================================================
%  PART 6: GENERATE TOPOLOGIES AND SAMPLES
% =========================================================================

for topoID = 1:nTopoTotal

    if topoID <= nTrainTopo
        splitName = 'train';
    else
        splitName = 'test';
    end

    channelDataDir  = fullfile(datasetRoot, splitName, 'channel', 'data');
    channelLabelDir = fullfile(datasetRoot, splitName, 'channel', 'label');
    beamDataDir     = fullfile(datasetRoot, splitName, 'beam', 'data');
    beamOthersDir   = fullfile(datasetRoot, splitName, 'beam', 'others');

    ticTopo = tic;

    %% --------------------------------------------------------------------
    % Generate AP/UE topology with wrap-around
    %% --------------------------------------------------------------------

    APpos = (rand(M,1) + 1i*rand(M,1)) * squareLength;
    UEpos = (rand(J,1) + 1i*rand(J,1)) * squareLength;

    APpos_real = [real(APpos), imag(APpos)];
    UEpos_real = [real(UEpos), imag(UEpos)];

    [APposWrapped, wrapLocations] = wrap_ap_positions_local(APpos, squareLength);
    [distances, whichWrap] = compute_wrapped_distances_local(APposWrapped, UEpos, distVertical);

    theta_mj = zeros(M, J);
    elev_mj  = zeros(M, J);

    for m = 1:M
        for j = 1:J
            AP_w = APposWrapped(m, whichWrap(m,j));
            theta_mj(m,j) = angle(UEpos(j) - AP_w);
            elev_mj(m,j)  = asin(min(max(distVertical / distances(m,j), -1), 1));
        end
    end

    distancesForPL   = max(distances, minLinkDist_m);
    distHorizontal_mj = sqrt(max(distancesForPL.^2 - distVertical^2, 0));

    %% --------------------------------------------------------------------
    % Large-scale fading
    %% --------------------------------------------------------------------

    [beta_mj, gainOverNoisedB, gainNoShadowdB] = generate_large_scale_fading_local( ...
        M, J, UEpos, wrapLocations, distancesForPL, sigma_sf, decorr, ...
        constantTerm, alpha_pl, noiseVardBm);

    %% --------------------------------------------------------------------
    % Global random pilot matrix metadata
    %% --------------------------------------------------------------------


    masterAP = zeros(J,1);
    for j = 1:J
        [~, masterAP(j)] = max(beta_mj(:,j));
    end

    save(fullfile(pilotInfoTrainDir, 'pilot_info.mat'), ...
        'Phi_pool', 'Phi_user', 'Phi_complex', 'Phi_real', 'Phi_realimag', ...
        'P_matrix', 'P_right', ...
        'pilotGram_complex', 'G_mix_complex', 'G_mix_realimag', ...
        'G_complex', 'G_realimag', 'G_abs', ...
        'pilotIndex', 'pilotReuseMatrix', 'Pset', 'maxReuse', 'meanReuse', ...
        'pilotColumnNorms', 'pilotGramRank', 'maxAbsPilotCorr', 'meanAbsPilotCorr', ...
        'M', 'J', 'Nt', 'tau_p', ...
        'pilotPowerLevels_dBm', 'pilotPowerLevels_mW', 'pilotBlockEnergyLevels_mW', ...
        'nPilotPowerLevels', 'nSamplesPerPilotPower', ...
        'pilotAssignMode', 'pilotMatrixMode', '-v7');

    save(fullfile(pilotInfoTestDir, 'pilot_info.mat'), ...
        'Phi_pool', 'Phi_user', 'Phi_complex', 'Phi_real', 'Phi_realimag', ...
        'P_matrix', 'P_right', ...
        'pilotGram_complex', 'G_mix_complex', 'G_mix_realimag', ...
        'G_complex', 'G_realimag', 'G_abs', ...
        'pilotIndex', 'pilotReuseMatrix', 'Pset', 'maxReuse', 'meanReuse', ...
        'pilotColumnNorms', 'pilotGramRank', 'maxAbsPilotCorr', 'meanAbsPilotCorr', ...
        'M', 'J', 'Nt', 'tau_p', ...
        'pilotPowerLevels_dBm', 'pilotPowerLevels_mW', 'pilotBlockEnergyLevels_mW', ...
        'nPilotPowerLevels', 'nSamplesPerPilotPower', ...
        'pilotAssignMode', 'pilotMatrixMode', '-v7');

    %% --------------------------------------------------------------------
    % Pathloss slope check, without saving figures
    %% --------------------------------------------------------------------

    x_pathloss = log10(distancesForPL(:));
    y_pathloss = gainOverNoisedB(:);
    y_pathloss_noshadow = gainNoShadowdB(:);

    pathlossFit = polyfit(x_pathloss, y_pathloss, 1);
    pathlossSlope_est = pathlossFit(1);
    pathlossIntercept_est = pathlossFit(2);

    pathlossFit_noshadow = polyfit(x_pathloss, y_pathloss_noshadow, 1);
    pathlossSlope_noshadow = pathlossFit_noshadow(1);

    %% --------------------------------------------------------------------
    % Steering vectors A
    %% --------------------------------------------------------------------

    n_ant = (0:Nt-1).';
    A = zeros(Nt, M, J, 'like', 1+1i);

    for m = 1:M
        for j = 1:J
            A(:,m,j) = exp(1i * 2*pi*antennaSpacing * n_ant * ...
                sin(theta_mj(m,j)) * cos(elev_mj(m,j)));
        end
    end

    A_real_topology = cat(1, real(A), imag(A));

    dataA_all = zeros(Nt, J, 2, M);
    for apID = 1:M
        A_complex = reshape(A(:,apID,:), Nt, J);
        dataA_all(:,:,:,apID) = cat(3, real(A_complex), imag(A_complex));
    end

    %% --------------------------------------------------------------------
    % Rician channel statistics
    %% --------------------------------------------------------------------

    [H_mean, R_cov, R_sqrt, kappa_mj, kappa_mj_dB, rho_los_mj, rho_nlos_mj] = ...
        compute_rician_statistics_local( ...
            M, J, Nt, distHorizontal_mj, A, beta_mj, ...
            theta_mj, elev_mj, ASD_varphi, ASD_theta, antennaSpacing);

    pLoS_mj = ones(M, J);
    LoS_mj  = true(M, J);

    H_mean_real = cat(1, real(H_mean), imag(H_mean));

    %% --------------------------------------------------------------------
    % Save topology metadata
    %% --------------------------------------------------------------------

    topologyFile = fullfile(beamOthersDir, sprintf('topology_%02d.mat', topoID));

    save(topologyFile, ...
        'APpos', 'UEpos', 'APpos_real', 'UEpos_real', ...
        'beta_mj', 'gainOverNoisedB', 'gainNoShadowdB', ...
        'theta_mj', 'elev_mj', 'distances', 'distancesForPL', 'distHorizontal_mj', ...
        'A_real_topology', ...
        'Phi_pool', 'Phi_user', 'Phi_complex', 'Phi_real', 'Phi_realimag', ...
        'P_matrix', 'P_right', ...
        'pilotGram_complex', 'G_mix_complex', 'G_mix_realimag', ...
        'G_complex', 'G_realimag', 'G_abs', ...
        'pilotIndex', 'pilotReuseMatrix', 'Pset', 'maxReuse', 'meanReuse', 'masterAP', ...
        'H_mean_real', ...
        'pLoS_mj', 'LoS_mj', 'kappa_mj', 'kappa_mj_dB', ...
        'rho_los_mj', 'rho_nlos_mj', 'R_cov', ...
        'K_ref_dB', 'K_min_dB', 'K_max_dB', 'ASD_deg', 'ASD_varphi_deg', 'ASD_theta_deg', ...
        'ASD_varphi', 'ASD_theta', 'antennaSpacing', 'minLinkDist_m', ...
        'M', 'J', 'Nt', 'tau_p', 'tau_c', ...
        'pilotPowerLevels_dBm', 'pilotPowerLevels_mW', 'pilotBlockEnergyLevels_mW', ...
        'nPilotPowerLevels', 'nSamplesPerPilotPower', ...
        'noiseVardBm', 'noiseVar_mW', 'noiseVar_W', ...
        'alpha_pl', 'constantTerm', 'sigma_sf', 'decorr', 'distVertical', ...
        'APheight', 'UEheight', 'pathlossModel', 'pilotAssignMode', 'pilotMatrixMode', 'useRandomLoSPhase', ...
        'pathlossSlope_est', 'pathlossIntercept_est', 'pathlossSlope_noshadow', ...
        '-v7');

    %% --------------------------------------------------------------------
    % Generate samples
    %% --------------------------------------------------------------------

    generate_samples_local( ...
        topoID, M, J, Nt, tau_p, nSamplesPerPilotPower, nPilotPowerLevels, ...
        pilotPowerLevels_dBm, pilotPowerLevels_mW, pilotBlockEnergyLevels_mW, ...
        Phi_user, P_matrix, P_right, H_mean, R_sqrt, ...
        pilotIndex, pilotReuseMatrix, G_mix_complex, G_mix_realimag, ...
        dataA_all, useRandomLoSPhase, ...
        channelDataDir, channelLabelDir, beamDataDir);

    fprintf('Topology %02d/%02d [%s] completed in %.1fs\n', ...
        topoID, nTopoTotal, upper(splitName), toc(ticTopo));
end

%% ========================================================================
%  PART 7: COMPLETION SUMMARY
% =========================================================================

trainSampleCount = nTrainTopo * M * nSamplesPerTopo;
testSampleCount = nTestTopo * M * nSamplesPerTopo;

fprintf('\n=== Dataset Generation Complete ===\n');
fprintf('Train samples: %d\n', trainSampleCount);
fprintf('Test samples : %d\n', testSampleCount);
fprintf('Tensor shape : [%d x %d x 2]\n', Nt, J);
fprintf('Output       : %s\n', fullfile(pwd, datasetRoot));
fprintf('Elapsed time : %.1fs\n', toc(ticTotal));

%% ========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function mkdir_if_missing(pathName)
    if ~exist(pathName, 'dir')
        mkdir(pathName);
    end
end

function y = db2pow_local(x)
    y = 10.^(x/10);
end

function [APposWrapped, wrapLocations] = wrap_ap_positions_local(APpos, squareLength)
    M = length(APpos);

    wrapH = repmat([-squareLength 0 squareLength], 3, 1);
    wrapV = wrapH.';
    wrapLocations = wrapH(:).' + 1i*wrapV(:).';

    APposWrapped = repmat(APpos, [1 9]) + repmat(wrapLocations, [M 1]);
end

function [distances, whichWrap] = compute_wrapped_distances_local(APposWrapped, UEpos, distVertical)
    [M, ~] = size(APposWrapped);
    J = length(UEpos);

    distances = zeros(M, J);
    whichWrap = zeros(M, J);

    for j = 1:J
        [d2D, wIdx] = min(abs(APposWrapped - UEpos(j)), [], 2);
        distances(:,j) = sqrt(distVertical^2 + d2D.^2);
        whichWrap(:,j) = wIdx;
    end
end

function [beta_mj, gainOverNoisedB, gainNoShadowdB] = generate_large_scale_fading_local( ...
    M, J, UEpos, wrapLocations, distancesForPL, sigma_sf, decorr, ...
    constantTerm, alpha_pl, noiseVardBm)

    gainOverNoisedB = zeros(M, J);
    gainNoShadowdB  = zeros(M, J);

    shadowCorrMatrix = sigma_sf^2 * ones(J, J);
    shadowReal       = zeros(J, M);

    for j = 1:J
        if j == 1
            meanSF = 0;
            stdSF  = sigma_sf;
            newcol = [];
        else
            shortDist = zeros(j-1, 1);
            for i = 1:j-1
                shortDist(i) = min(abs(UEpos(j) - UEpos(i) + wrapLocations));
            end

            newcol = sigma_sf^2 * 2.^(-shortDist / decorr);
            term1  = newcol' / shadowCorrMatrix(1:j-1, 1:j-1);

            meanSF = term1 * shadowReal(1:j-1, :);
            stdSF  = sqrt(max(sigma_sf^2 - term1 * newcol, 0));
        end

        shadowing = meanSF + stdSF * randn(1, M);

        betaLoS_dB = constantTerm - alpha_pl * log10(distancesForPL(:,j));
        gainNoShadowdB(:,j) = betaLoS_dB - noiseVardBm;
        gainOverNoisedB(:,j) = gainNoShadowdB(:,j) + shadowing.';

        shadowCorrMatrix(1:j-1, j) = newcol;
        shadowCorrMatrix(j, 1:j-1) = newcol';
        shadowReal(j,:) = shadowing;
    end

    beta_mj = db2pow_local(gainOverNoisedB);
end


function [Phi_user, pilotColumnNorms, pilotGram_complex, ...
    G_mix_complex, G_abs, G_mix_realimag] = ...
    generate_global_random_pilot_matrix_local(tau_p, J, pilotMatrixMode)

    switch lower(pilotMatrixMode)
        case 'complexgaussianunitnorm'
            Phi_user = sqrt(0.5) * (randn(tau_p, J) + 1i*randn(tau_p, J));
        case 'realgaussianunitnorm'
            Phi_user = randn(tau_p, J);
        case 'qpskunitnorm'
            bitsI = 2*(rand(tau_p, J) > 0.5) - 1;
            bitsQ = 2*(rand(tau_p, J) > 0.5) - 1;
            Phi_user = (bitsI + 1i*bitsQ) / sqrt(2);
        otherwise
            error('Unsupported pilotMatrixMode: %s', pilotMatrixMode);
    end

    % Normalize each user pilot to unit energy so random pilot amplitudes do
    % not create artificial user-power differences.
    for j = 1:J
        colNorm = norm(Phi_user(:,j));
        if colNorm < eps
            error('Generated a near-zero pilot column. Try another random seed.');
        end
        Phi_user(:,j) = Phi_user(:,j) / colNorm;
    end

    pilotColumnNorms = sqrt(sum(abs(Phi_user).^2, 1));
    pilotGram_complex = Phi_user' * Phi_user;             % Phi^H * Phi
    G_mix_complex     = Phi_user.' * conj(Phi_user);      % Phi^T * Phi*
    G_abs             = abs(G_mix_complex);
    G_mix_realimag    = cat(3, real(G_mix_complex), imag(G_mix_complex));
end

function [H_mean, R_cov, R_sqrt, kappa_mj, kappa_mj_dB, rho_los_mj, rho_nlos_mj] = ...
    compute_rician_statistics_local( ...
    M, J, Nt, distHorizontal_mj, A, beta_mj, ...
    theta_mj, elev_mj, ASD_varphi, ASD_theta, antennaSpacing)

    H_mean = zeros(Nt, M, J, 'like', 1+1i);
    R_cov  = zeros(Nt, Nt, M, J, 'like', 1+1i);
    R_sqrt = zeros(Nt, Nt, M, J, 'like', 1+1i);

    kappa_mj    = zeros(M, J);
    kappa_mj_dB = zeros(M, J);
    rho_los_mj  = zeros(M, J);
    rho_nlos_mj = zeros(M, J);

    for m = 1:M
        for j = 1:J
            d_horiz = distHorizontal_mj(m,j);

            % Rician-project style distance-dependent K-factor.
            kappa_mj(m,j)    = 10^(1.3 - 0.003*d_horiz);
            kappa_mj_dB(m,j) = 10*log10(kappa_mj(m,j));

            rho_los_mj(m,j)  = beta_mj(m,j) * kappa_mj(m,j) / (kappa_mj(m,j) + 1);
            rho_nlos_mj(m,j) = beta_mj(m,j) / (kappa_mj(m,j) + 1);

            R_unit = functionRlocalscattering( ...
                Nt, theta_mj(m,j), elev_mj(m,j), ...
                ASD_varphi, ASD_theta, antennaSpacing);

            R_unit = (R_unit + R_unit') / 2;

            H_mean(:,m,j) = sqrt(rho_los_mj(m,j)) * A(:,m,j);

            R_cov(:,:,m,j) = rho_nlos_mj(m,j) * R_unit;
            R_cov(:,:,m,j) = (R_cov(:,:,m,j) + R_cov(:,:,m,j)') / 2;

            R_sqrt(:,:,m,j) = hermitian_sqrt_psd_local(R_cov(:,:,m,j), false);
        end
    end
end

function generate_samples_local( ...
    topoID, M, J, Nt, tau_p, nSamplesPerPilotPower, nPilotPowerLevels, ...
    pilotPowerLevels_dBm, pilotPowerLevels_mW, pilotBlockEnergyLevels_mW, ...
    Phi_user, P_matrix, P_right, H_mean, R_sqrt, ...
    pilotIndex, pilotReuseMatrix, G_mix_complex, G_mix_realimag, ...
    dataA_all, useRandomLoSPhase, ...
    channelDataDir, channelLabelDir, beamDataDir)

    % Compatibility aliases now point to the actual matched-filter mixing
    % matrix Phi^T*Phi*, not to the conventional Gram matrix Phi^H*Phi.
    G_complex  = G_mix_complex;
    G_realimag = G_mix_realimag;

    for powerIdx = 1:nPilotPowerLevels
        pilotPower_dBm      = pilotPowerLevels_dBm(powerIdx);
        pilotPower_mW       = pilotPowerLevels_mW(powerIdx);
        pilotBlockEnergy_mW = pilotBlockEnergyLevels_mW(powerIdx);

        for idxID = 1:nSamplesPerPilotPower

            %% -------------------------------------------------------------
            % Generate Rician channel H_s [Nt x M x J]
            %% -------------------------------------------------------------

            H_s = zeros(Nt, M, J, 'like', 1+1i);

            if useRandomLoSPhase
                phaseLoS_mj = exp(1i * (-pi + 2*pi*rand(M,J)));
            else
                phaseLoS_mj = ones(M,J);
            end

            for m = 1:M
                for j = 1:J
                    w = sqrt(0.5) * (randn(Nt,1) + 1i*randn(Nt,1));
                    h_nlos = R_sqrt(:,:,m,j) * w;
                    h_los  = H_mean(:,m,j) * phaseLoS_mj(m,j);

                    H_s(:,m,j) = h_los + h_nlos;
                end
            end

            %% -------------------------------------------------------------
            % Generate received pilot signal Y_s [Nt x tau_p x M]
            %   Y_m = sqrt(tau_p*pilotPower_mW) * H_m * P_right + W_m
            %% -------------------------------------------------------------

            Y_s = zeros(Nt, tau_p, M, 'like', 1+1i);

            for m = 1:M
                H_m = reshape(H_s(:,m,:), Nt, J);
                sig = H_m * P_right;

                noise = sqrt(0.5) * (randn(Nt, tau_p) + 1i*randn(Nt, tau_p));
                Y_s(:,:,m) = sqrt(pilotBlockEnergy_mW) * sig + noise;
            end

            %% -------------------------------------------------------------
            % Save AP-level samples
            %% -------------------------------------------------------------

            for apID = 1:M
                H_complex = reshape(H_s(:,apID,:), Nt, J);
                Y_complex = Y_s(:,:,apID);


                Psi_corr_complex = (1/sqrt(pilotBlockEnergy_mW)) * Y_complex * P_matrix;

                dataP  = cat(3, real(Psi_corr_complex), imag(Psi_corr_complex));
                labelH = cat(3, real(H_complex), imag(H_complex));
                dataA  = dataA_all(:,:,:,apID);

                dataY = cat(3, real(Y_complex), imag(Y_complex));
                Y_herm_complex = Y_complex';
                dataY_herm = cat(3, real(Y_herm_complex), imag(Y_herm_complex));

                P_file = fullfile(channelDataDir, ...
                    sprintf('dataP_topo_%02d_ap_%02d_ppow_dBm_%g_idx_%03d.mat', ...
                    topoID, apID, pilotPower_dBm, idxID));

                H_file = fullfile(channelLabelDir, ...
                    sprintf('labelH_topo_%02d_ap_%02d_ppow_dBm_%g_idx_%03d.mat', ...
                    topoID, apID, pilotPower_dBm, idxID));

                A_file = fullfile(beamDataDir, ...
                    sprintf('dataA_topo_%02d_ap_%02d_ppow_dBm_%g_idx_%03d.mat', ...
                    topoID, apID, pilotPower_dBm, idxID));

                phaseLoS_ap = phaseLoS_mj(apID,:); 

                save(P_file, ...
                    'dataP', 'dataY', 'dataY_herm', ...
                    'Phi_user', 'P_matrix', 'P_right', ...
                    'G_mix_complex', 'G_mix_realimag', ...
                    'G_complex', 'G_realimag', ...
                    'pilotPower_dBm', 'pilotPower_mW', 'pilotBlockEnergy_mW', ...
                    'topoID', 'apID', 'idxID', ...
                    'pilotIndex', 'pilotReuseMatrix', '-v7');

                save(H_file, 'labelH', ...
                    'pilotPower_dBm', 'pilotPower_mW', 'pilotBlockEnergy_mW', ...
                    'topoID', 'apID', 'idxID', 'phaseLoS_ap', 'useRandomLoSPhase', '-v7');

                save(A_file, 'dataA', ...
                    'pilotPower_dBm', 'pilotPower_mW', 'pilotBlockEnergy_mW', ...
                    'topoID', 'apID', 'idxID', '-v7');

            end
        end
    end
end

function A = hermitian_sqrt_psd_local(R, inverseFlag)
    if nargin < 2
        inverseFlag = false;
    end

    R = (R + R') / 2;

    [V, D] = eig(R);
    lambda = real(diag(D));
    lambda(lambda < 0) = 0;

    epsVal = 1e-12;
    if inverseFlag
        lambdaSafe = max(lambda, epsVal);
        A = V * diag(1 ./ sqrt(lambdaSafe)) * V';
    else
        A = V * diag(sqrt(lambda)) * V';
    end

    A = (A + A') / 2;
end
