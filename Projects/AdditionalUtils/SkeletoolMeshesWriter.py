

def writeSkeletoolMeshesFile(outputFileName='', numberOfMeshes=0, basePath='', meshName='', startFrame=0):

    # open output file
    outputMeshesFile = open(outputFileName, 'w')

    # write skeletool version
    outputMeshesFile.write('Skeletool Meshes file v1.0\n')

    # write number of frames
    outputMeshesFile.write('frames ' + str(numberOfMeshes) +'\n')

    # write basepath
    outputMeshesFile.write('basepath ' + basePath + '\n')

    # write mesh name
    outputMeshesFile.write('trimeshes ' + meshName + '\n')

    # write start frame
    outputMeshesFile.write('format f ' + str(startFrame) +'\n')
