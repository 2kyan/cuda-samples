#!/bin/bash
HNVCC_OP=dump hnvcc cudaTensorCoreGemm-full-rr.cu -arch=sm_70 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm-full-rr.cu -arch=sm_75 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm-full-rr.cu -arch=sm_80 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm-full-rr.cu -arch=sm_86 -I ../../../Common/
dsass dump.cudaTensorCoreGemm-full-rr.sm_70.cubin
dsass dump.cudaTensorCoreGemm-full-rr.sm_75.cubin
dsass dump.cudaTensorCoreGemm-full-rr.sm_80.cubin
dsass dump.cudaTensorCoreGemm-full-rr.sm_86.cubin
HNVCC_OP=dump hnvcc cudaTensorCoreGemm-full-rc.cu -arch=sm_70 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm-full-rc.cu -arch=sm_75 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm-full-rc.cu -arch=sm_80 -I ../../../Common/
HNVCC_OP=dump hnvcc cudaTensorCoreGemm-full-rc.cu -arch=sm_86 -I ../../../Common/
dsass dump.cudaTensorCoreGemm-full-rc.sm_70.cubin
dsass dump.cudaTensorCoreGemm-full-rc.sm_75.cubin
dsass dump.cudaTensorCoreGemm-full-rc.sm_80.cubin
dsass dump.cudaTensorCoreGemm-full-rc.sm_86.cubin
