

import sys
sys.path.append("../")

#########################################################################

import AdditionalUtils.CreateMeshTensor as CreateMeshTensor
import CustomTFOperators.ChamferLoss as ChamferLoss
import os
import numpy as np
import AdditionalUtils.OBJReader as OBJReader

#########################################################################

# ddcMeshes   = '/HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/results/tensorboardLogDeepDynamicCharacters/3709x3756x3760/snapshot_iter_359999/3773_train_100_18900/final.meshes'
# oursMeshes  = '/HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/results/tensorboardLogDeepDynamicCharacters/3709x3756x3760x4104x4150/snapshot_iter_359999/4157_test_0_18900/final.meshes'

ddcMeshes   = '/HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/results/tensorboardLogDeepDynamicCharacters/3709x3756x3760/snapshot_iter_359999/4159_test_0_7400/final.meshes'
oursMeshes  = '/HPS/RTMPC4/work/DeepDynamicCharacters/Dataset/VladNew/results/tensorboardLogDeepDynamicCharacters/3709x3756x3760x4104x4150/snapshot_iter_359999/4158_test_0_7400/final.meshes'

startFrame  = 100
# endFrame    = 18800
endFrame    = 7300
gtMeshBasePath = '/HPS/RLData1/work/Vlad/testing/recon'

stepSize = 10

#########################################################################

print('Read meshes ...', flush=True)

ddcMeshes = CreateMeshTensor.createMeshSequenceTensor(inputMeshesFile=ddcMeshes, startFrame=startFrame, numSamples=endFrame-startFrame, scale=0.001)
oursMeshes = CreateMeshTensor.createMeshSequenceTensor(inputMeshesFile=oursMeshes, startFrame=startFrame, numSamples=endFrame-startFrame, scale=0.001)

totalChamferDDC   = 0.0
totalChamferOurs  = 0.0

totalHausdorffDDC   = 0.0
totalHausdorffOurs  = 0.0

print('Start loop ...', flush=True)

for f in range(startFrame, endFrame, stepSize):

    gtMeshFile = os.path.join(gtMeshBasePath, str(f).zfill(6), 'model.obj')

    gtMesh = OBJReader.OBJReader(gtMeshFile, readMTLFlag=False, segWeightsFlag=False, computeAdjacencyFlag=False, computePerFaceTextureCoordinatedFlag=False, verbose=True)
    vertexCoordinates = np.array(gtMesh.vertexCoordinates, dtype=np.float32) * 0.001

    ddcMesh = ddcMeshes [f - startFrame]
    ourMesh = oursMeshes[f - startFrame]

    chamferDDC, hausDorffDDC   = ChamferLoss.chamferHausdorffMetric(ddcMesh, vertexCoordinates)
    chamferOurs, hausDorffOurs = ChamferLoss.chamferHausdorffMetric(ourMesh, vertexCoordinates)

    totalChamferDDC = totalChamferDDC + chamferDDC
    totalChamferOurs = totalChamferOurs + chamferOurs

    totalHausdorffDDC = totalHausdorffDDC + hausDorffDDC
    totalHausdorffOurs = totalHausdorffOurs + hausDorffOurs

    numSamples = (f - startFrame) / stepSize

    print(f,     '{:.6f}'.format(totalChamferOurs/float(numSamples)),   '{:.6f}'.format(totalChamferDDC/float(numSamples)),
          '~~~', '{:.6f}'.format(totalHausdorffOurs/float(numSamples)), '{:.6f}'.format(totalHausdorffDDC/float(numSamples)), flush=True)

numSamples = (endFrame-startFrame) / stepSize

print('Average Chamfer error: ', '{:.6f}'.format(totalChamferOurs /float(numSamples)), '{:.6f}'.format(totalChamferDDC /float(numSamples)), flush=True)
print('Average Haudorff error: ', '{:.6f}'.format(totalHausdorffOurs /float(numSamples)), '{:.6f}'.format(totalHausdorffDDC /float(numSamples)), flush=True)

#########################################################################