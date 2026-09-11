# ARBNet

Cell-free integrated sensing and communication (CF-ISAC) is fundamentally dependent on precise channel state information (CSI) for effective network-wide coordination. However, the practical attainment of CSI is often compromised due to non-orthogonal pilot acquisition and limitations encountered in the access point (AP) to central processing unit (CPU) fronthaul, prior to its application in coordinated beamforming. This paper introduces ARBNet, a comprehensive framework that seamlessly integrates the processes of CSI acquisition, fronthaul-constrained reconstruction, and coordinated beamforming throughout the entire CSI processing chain.

In the initial phase, a hybrid estimator utilizing fast Fourier transform and state-space techniques employs depthwise-separable residual blocks (DSRBs) for refinement at the antenna domain, while bidirectional selective state-space blocks (BSSBs) effectively capture dependencies within the beam domain through pilot-conditioned modulation methods. Subsequently, the APs are tasked with retaining and quantizing sparse beam-domain coefficients, while support-aware pre-activation residual blocks (PARBs) carry out CSI reconstruction at the central processing unit. The following stage involves the implementation of message-passing blocks (MPBs), which facilitate the learning of AP-device coordination and device weight parameters in the context of structured regularized zero-forcing beamforming. Unlike existing designs that operate on either directly available CSI or separately defined CSI errors, ARBNet conducts coordinated transmissions based on the CSI representation resulting from the preceding acquisition and fronthaul reconstruction stages.

Extensive simulations substantiate that ARBNet significantly reduces the normalized mean-square errors (NMSEs) for estimation and reconstruction to -6.620 and -5.733 dB, respectively, while enhancing the minimum communication signal-to-interference-plus-noise ratio (SINR) by 23.641 dB relative to the end-to-end baseline. Furthermore, it sustains a mean sensing signal-to-noise ratio (SNR) of 11.875 dB.

If there is any error or topic that needs to be discussed, please contact [Truong-Thinh Le](https://github.com/KrynStackk) at [letruongthinh1712@gmail.com](mailto:letruongthinh1712@gmail.com).

## Architecture

### Device-Centric CF-ISAC System Architecture

<p align="center">
  <img src="figs/system.png" width="495">
</p>

### Overview of ARBNet Architecture

<p align="center">
  <img src="figs/all.png" width="100%">
</p>
