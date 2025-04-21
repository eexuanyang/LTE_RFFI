# LTE_RFFI
This project sets up an LTE device radio frequency fingerprint identification system using deep learning techniques. The LTE uplink signals are collected from ten different LTE devices using a USRP N210 in different locations. The sampling rate of the USRP is 25 MHz. The received signals are resampled to 30.72 MHz in Matlab. Then, the signals are processed and saved in the MAT file form. The signals processed by the proposed WL method in our paper are included in the dataset. The corresponding dataset can be found in [LTE_RFF_IDENTIFICATION_DATASET](https://ieee-dataport.org/documents/lterffidentificationdataset).
#  Citation
If the part of the dataset/codes contributes to your project, please cite:

```
[1] X. Yang and D. Li, "LED-RFF: LTE DMRS-Based Channel Robust Radio Frequency Fingerprint Identification Scheme," in IEEE Transactions on Information Forensics and Security, vol. 19, pp. 1855-1869, 2024, doi: 10.1109/TIFS.2023.3343079.
```
# Note
The neural network [InceptionTime](https://github.com/hfawaz/InceptionTime) is adopted and modified in this project.
