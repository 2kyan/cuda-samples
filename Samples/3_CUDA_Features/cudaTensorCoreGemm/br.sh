#!/bin/bash
HNVCC_OP=dump hnvcc cudaTensorCoreGemm_rr.cu -arch=sm_70 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm_rr.cu -arch=sm_75 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm_rr.cu -arch=sm_80 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm_rr.cu -arch=sm_86 -I ../../../Common/
dsass dump.cudaTensorCoreGemm_rr.sm_70.cubin
dsass dump.cudaTensorCoreGemm_rr.sm_75.cubin
dsass dump.cudaTensorCoreGemm_rr.sm_80.cubin
dsass dump.cudaTensorCoreGemm_rr.sm_86.cubin
HNVCC_OP=dump hnvcc cudaTensorCoreGemm_rc.cu -arch=sm_70 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm_rc.cu -arch=sm_75 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm_rc.cu -arch=sm_80 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm_rc.cu -arch=sm_86 -I ../../../Common/
dsass dump.cudaTensorCoreGemm_rc.sm_70.cubin
dsass dump.cudaTensorCoreGemm_rc.sm_75.cubin
dsass dump.cudaTensorCoreGemm_rc.sm_80.cubin
dsass dump.cudaTensorCoreGemm_rc.sm_86.cubin
